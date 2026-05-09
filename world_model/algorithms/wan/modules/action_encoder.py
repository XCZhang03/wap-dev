import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.geometry_utils import CameraPose
from .vae import CausalConv3d, RMS_norm

class CausalConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self._padding = 2 * padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self._padding, 0))  # pad left side only
        return super().forward(x)


class ActionEncoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, AdaLN_proj=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        # this conv collapses 4 timesteps → 1
        self.temporal_down = nn.Sequential(
            CausalConv1D(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CausalConv1D(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        if AdaLN_proj:
            self.proj = nn.Sequential(
                torch.nn.Linear(hidden_dim, 6*hidden_dim),
                torch.nn.SiLU(),
            )

    def encode_chunk(self, x):
        """
        Convenience wrapper kept for API compatibility; runs the
        temporal downsampling conv on the full sequence.

        x: (B, T, D)
        returns: (B, T_out, D)
        """
        x = x.transpose(1, 2)      # (B, D, T)
        x = self.temporal_down(x)  # (B, D, T_out)
        x = x.transpose(1, 2)      # (B, T_out, D)
        return x

    def forward(self, x):
        """
        x: (B, 1+T, D)
        returns: (B, 1+T//4, D)
        """
        B, T, D = x.shape
        # Encode the full sequence and apply a single causal
        # temporal downsampling conv across time. This matches the
        # behavior of running CausalConv1D over the entire sequence
        # (with stride=4) rather than emulating chunked streaming.
        x = self.encode(x)                  # (B, T, D)
        x = self.encode_chunk(x)            # (B, T_out, D)
        if hasattr(self, 'proj'):
            x_ = self.proj(x).unflatten(-1, (6, self.hidden_dim))
            return x, x_
        return x

class PatchEncoder3D(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim,
        dim=64,
    ):
        """
        3D patch encoder for camera pose encoding, following similar logic as VAE encoder.
        """  
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.token_dim = token_dim
        self.input_proj = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(input_dim, dim, 3, stride=(2, 2)),
        )
        self.resamples = nn.ModuleList([nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        ) for _ in range(2)])
        self.time_convs = nn.ModuleList([nn.Sequential(
            CausalConv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=(3,3,3),
                stride=(2,1,1),
                padding=(1,1,1)
            ),
            RMS_norm(dim, images=False),
            nn.SiLU()
        ) for _ in range(2)])
        self.patch_embed = nn.Conv3d(dim, dim, kernel_size=(1,2,2), stride=(1,2,2))
        self.adaln_proj = nn.Sequential(nn.Linear(dim, 6*token_dim), nn.SiLU())

    @torch.enable_grad()
    def forward(self, x):
        t = x.shape[1]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.input_proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        for time_conv, resample in zip(self.time_convs, self.resamples):
            x = time_conv(x)
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = resample(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        x = self.patch_embed(x)
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = self.adaln_proj(x).unflatten(-1, (6, self.token_dim))
        return x
    
class CameraPoseEncoder(nn.Module):
    def __init__(
            self,
            dim,
            cond_mode='global',
            num_cams=4,
            normalization='none'
    ):
        super().__init__()
        self.cond_mode = cond_mode
        self.num_cams = num_cams
        self.normalization = normalization
        assert cond_mode in ['global', 'ray', 'ray-encoding']
        match cond_mode:
            case "global":
                self.conditioning_dim = 12
            case "ray" | "plucker":
                self.conditioning_dim = 6
            case "ray-encoding":
                self.conditioning_dim = 180
            case _:
                raise ValueError(
                    f"Unknown camera pose conditioning type: {cond_mode}"
                )
        match cond_mode:
            case "global":
                self.encoder = ActionEncoder(
                    action_dim=self.conditioning_dim,
                    hidden_dim=dim // self.num_cams,
                    AdaLN_proj=True,
                )
            case "ray" | "ray-encoding" | "plucker":
                self.encoder = PatchEncoder3D(
                    input_dim=self.conditioning_dim,
                    token_dim=dim,
                )

    def forward(self, raw_camera_poses, video_metadata):
        dtype = next(self.encoder.parameters()).dtype
        B, T, num_cams, _ = raw_camera_poses.shape
        assert num_cams == self.num_cams, f"Expected {self.num_cams} cameras, got {num_cams}"

        height = video_metadata['height'][0]
        width = video_metadata['width'][0]
        assert height == width, "Only supports square images for now"
        if num_cams == 1:
            resolution = width
        elif num_cams == 4:
            resolution = width // 2
        else:
            raise ValueError("Only supports 1 or 4 cameras for now")
        
        conds = []            # used for 'global' or 1-camera ray modes
        conds_panel = None   # used for 4-camera ray/plucker modes

        for cam_id in range(num_cams):
            camera_poses = CameraPose.from_vectors(raw_camera_poses[:, :, cam_id, :])  # (B, T, pose_dim)
            match self.normalization:
                case 'none':
                    pass
                case 'first':
                    camera_poses.normalize_by_first()
                case 'mean':
                    camera_poses.normalize_by_mean()
            match self.cond_mode:
                case "global":
                    camera_poses_cond =  camera_poses.extrinsics(flatten=True)
                case "ray" | "ray-encoding" | "plucker":
                    rays = camera_poses.rays(resolution=resolution)
                    if self.cond_mode == "ray-encoding":
                        rays = rays.to_pos_encoding()[0]
                    else:
                        rays = rays.to_tensor(
                            use_plucker=self.cond_mode == "plucker"
                        )
                    camera_poses_cond = rearrange(rays, "b t h w c -> b t c h w")
            camera_poses_cond = camera_poses_cond.to(dtype=dtype)
            del camera_poses

            if self.cond_mode in ['ray', 'ray-encoding', 'plucker'] and num_cams == 4:
                # Build the 2x2 camera mosaic incrementally to avoid
                # holding all four camera tensors plus a separate
                # concatenated result in memory.
                if conds_panel is None:
                    Bc, Tc, Cc, Hc, Wc = camera_poses_cond.shape
                    conds_panel = camera_poses_cond.new_empty(Bc, Tc, Cc, Hc * 2, Wc * 2)

                _, _, _, Hc, Wc = camera_poses_cond.shape
                if cam_id == 0:
                    conds_panel[:, :, :, :Hc, :Wc] = camera_poses_cond
                elif cam_id == 1:
                    conds_panel[:, :, :, :Hc, Wc:] = camera_poses_cond
                elif cam_id == 2:
                    conds_panel[:, :, :, Hc:, :Wc] = camera_poses_cond
                elif cam_id == 3:
                    conds_panel[:, :, :, Hc:, Wc:] = camera_poses_cond
                del camera_poses_cond
            else:
                conds.append(camera_poses_cond)

        # we need to ensure that the last dimension does not change when aggregating the cameras
        if self.cond_mode in ['ray', 'ray-encoding', 'plucker']:
            if num_cams == 1:
                conds = conds[0]
            elif num_cams == 4:
                assert conds_panel is not None
                conds = conds_panel
            else:
                raise ValueError("Only supports 1 or 4 cameras for now")
            return self.encode(conds)
        elif self.cond_mode == 'global':
            conds = [self.encode(c)[1] for c in conds]
            conds = torch.cat(conds, dim=-1)
            return conds

    def encode(self, conds):
        assert conds.shape[2] == self.conditioning_dim, f"Expected conditioning dim {self.conditioning_dim}, got {conds.shape[2]}"
        conds_hidden_states = self.encoder(conds)
        return conds_hidden_states

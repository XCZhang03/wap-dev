from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
import threading
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

from .video_base import VideoDataset


class CondVideoDataset(VideoDataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__(cfg, split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return self.load_record(record)

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # Load video data - either raw or preprocessed latents
        videos, conds, low_dim_conds, camera_poses = self._load_video_cond(record)
        # images = videos[:1].clone() if self.image_to_video else None
        image_latents, video_latents = None, None
        video_metadata = {
            "num_frames": videos.shape[0],
            "height": videos.shape[2],
            "width": videos.shape[3],
        }

        # Load prompt data - either raw or preprocessed embeddings
        prompts = self.id_token
        prompt_embeds = None
        prompt_embed_len = None
        negative_prompt_embeds = None
        negative_prompt_embed_len = None
        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(record)
            negative_prompt_embeds, negative_prompt_embed_len = self._load_prompt_embed(record, negative=True)


        has_bbox, bbox_render = self._render_bbox(record)

        output = {
            "videos": videos,
            "conds": conds,
            # "low_dim_conds": low_dim_conds,
            # "camera_poses": camera_poses,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
            "video_path": str(self.data_root / record["video_path"]),
        }

        if prompts is not None:
            output["prompts"] = prompts
        # if images is not None:
        #     output["images"] = images
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        if negative_prompt_embeds is not None:
            output["negative_prompt_embeds"] = negative_prompt_embeds
            output["negative_prompt_embed_len"] = negative_prompt_embed_len
        if image_latents is not None:
            output["image_latents"] = image_latents
        if video_latents is not None:
            output["video_latents"] = video_latents

        return output
    
    def pad_actions(self, actions: torch.Tensor):
        import torch.nn.functional as F
        action_dim = self.cfg.get('action_dim', 7)
        if actions.shape[-1] > action_dim:
            raise ValueError(f"Action dimension {actions.shape[-1]} is larger than expected {action_dim}.")
        elif actions.shape[-1] < action_dim:
            if actions.shape[-1] == 7:
                actions = F.pad(actions, (0, action_dim - actions.shape[-1]), "constant", 0)
            elif actions.shape[-1] == 14 or actions.shape[-1] == 24:
                half_actions = actions.chunk(2, dim=-1)
                half_actions = [F.pad(half_action, (0, action_dim // 2 - half_action.shape[-1]), "constant", 0) for half_action in half_actions]
                actions = torch.cat(half_actions, dim=-1)
            else:
                raise NotImplementedError(f"Padding for action dimension {actions.shape[-1]} is not implemented.")
        # print("[DEBUG] [Action shape]", actions.shape)
        return actions
                



    def _load_video_cond(self, record: Dict[str, Any]) -> torch.Tensor:
        """
        Given a record, return a tensor of shape (n_frames, 3, H, W)
        """

        video_path = self.data_root / record["video_path"]
        cond_path = self.data_root / record['cond_path']
        # low_dim_path = self.data_root / record['low_dim_path']
        video_reader = decord.VideoReader(uri=video_path.as_posix())
        cond_reader = decord.VideoReader(uri=cond_path.as_posix())
        # low_dim_cond_reader = np.load(low_dim_path.as_posix(), allow_pickle=True)

        n_frames = len(video_reader)
        assert n_frames == len(cond_reader), "Video and cond length mismatch"
        start = record.get("trim_start", 0)
        end = record.get("trim_end", n_frames)
        indices = self._temporal_sample(end - start, record["fps"])
        indices = list(start + indices)
        frames = video_reader.get_batch(indices)
        cond_frames = cond_reader.get_batch(indices)
        # low_dim_conds = self.pad_actions(torch.from_numpy(low_dim_cond_reader['actions'][indices]).float()).contiguous()
        # camera_poses = self.load_camera_pose(low_dim_cond_reader)[indices]  # (n_frames, num_cams, pose_dim)
        low_dim_conds = None
        camera_poses = None
        # do some padding
        if len(frames) != self.n_frames:
            raise ValueError(
                f"Expected {len(frames)=} to be equal to {self.n_frames=}."
            )

        # crop if specified in the record
        if "crop_top" in record and "crop_bottom" in record:
            frames = frames[:, record["crop_top"] : record["crop_bottom"]]
            cond_frames = cond_frames[:, record["crop_top"] : record["crop_bottom"]]
        if "crop_left" in record and "crop_right" in record:
            frames = frames[:, :, record["crop_left"] : record["crop_right"]]
            cond_frames = cond_frames[:, :, record["crop_left"] : record["crop_right"]]

        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0
        cond_frames = cond_frames.float().permute(0, 3, 1, 2).contiguous() / 255.0
        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
            cond_frames = self.no_augment_transforms(cond_frames)
        else:
            frames = self.augment_transforms(frames)
            cond_frames = self.augment_transforms(cond_frames)
        frames = self.img_normalize(frames)
        cond_frames = self.img_normalize(cond_frames)

        return frames, cond_frames, low_dim_conds, camera_poses

    def load_camera_pose(self, low_dim_cond_reader):
        def pose2vec(intrinsics, extrinsics):
            '''
            Docstring for pose2vec
            
            :param intrinsics: 3x3 intrinsics matrix
            :param extrinsics: 3x4 extrinsics matrix
            '''
            # intrinsics
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]
            W = px * 2
            H = py * 2
            intrinsics_vec = torch.tensor([fx / W, fy / H, px / W, py / H])
            # extrinsics
            RT = rearrange(torch.tensor(extrinsics)[:3], 'i j -> (i j)', i=3, j=4)

            # concat
            pose = torch.cat([intrinsics_vec, RT], dim=0)

            # # DEBUG
            # from utils.geometry_utils import CameraPose
            # cam_pose = CameraPose.from_vectors(pose.unsqueeze(0).unsqueeze(0))
            # extrinsics_debug = cam_pose.extrinsics().numpy()
            # intrinsics_debug = cam_pose.intrinsics().numpy() * np.array([[W], [H], [1]])
            # assert np.allclose(intrinsics_debug, intrinsics)
            # assert np.allclose(extrinsics_debug, extrinsics[:3])

            return pose
            
        camera_names = low_dim_cond_reader['panel_order']
        camera_poses = {cam_name: [] for cam_name in camera_names}
        for cam_pose in low_dim_cond_reader['camera_poses']:
            for cam_name in camera_names:
                intrinsics = cam_pose[cam_name]['intrinsics']
                extrinsics = cam_pose[cam_name]['extrinsics']
                pose_vec = pose2vec(intrinsics, extrinsics)
                camera_poses[cam_name].append(pose_vec)
        for cam_name in camera_names:
            camera_poses[cam_name] = torch.stack(camera_poses[cam_name], dim=0)

        return torch.stack([camera_poses[cam_name] for cam_name in camera_names], dim=1)  # (num_frames, num_cams, pose_dim)

        
        

    


    def download(self):
        """
        Automatically download the dataset to self.data_root. Optional.
        """
        raise NotImplementedError(
            "Automatic download not implemented for this dataset."
        )

from typing import Dict, Tuple, Union
import copy
import torch
import numpy as np
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import robosuite.utils.transform_utils as T

def get_rot_rep(key: str) -> str:
    if key.endswith('mat'):
        return 'matrix'
    elif key.endswith('quat'):
        return 'quaternion'
    elif key.endswith('ori'):
        return 'axis_angle'
    else:
        raise RuntimeError(f"Unsupported rotation representation for key: {key}")


class LowdimIDMObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rotation_rep: str = 'quaternion',
            use_delta: bool = True,
        ):
        """
        Assumes low_dim input: B,D
        """
        super().__init__()

        self.use_delta = use_delta

        pos_keys = dict()
        rot_keys = dict()
        qpos_keys = dict()
        key_shape_map = dict()

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'qpos')
            key_shape_map[key] = shape
            if type == 'qpos':
                assert key.endswith('qpos'), f"Expected qpos key to end with 'qpos', got {key}"
                qpos_keys[key] = shape
            elif type == 'pos':
                assert key.endswith('pos'), f"Expected pos key to end with 'pos', got {key}"
                pos_keys[key] = shape
            elif type == 'rot':
                # raise NotImplementedError("Rotation type is not supported in InverseDynamicsStateEncoder.")
                rot_keys[key] = shape
                assert get_rot_rep(key) == rotation_rep == 'quaternion', \
                    "Only quaternion rotation representation is supported currently."
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        pos_keys = sorted(pos_keys)
        rot_keys = sorted(rot_keys)
        qpos_keys = sorted(qpos_keys)

        self.shape_meta = shape_meta
        self.pos_keys = pos_keys
        self.rot_keys = rot_keys
        self.qpos_keys = qpos_keys
        self.key_shape_map = key_shape_map

    def get_zero_obs(self, batch_size: int=1)  -> Dict[str, torch.Tensor]:
        zero_obs = dict()
        for key in self.key_shape_map:
            shape = self.key_shape_map[key]
            if key in self.rot_keys:
                zero_obs[key] = torch.tensor([[0,0,0,1]] * batch_size, device=self.device, dtype=self.dtype)
            else:
                zero_obs[key] = torch.zeros((batch_size,)+shape, device=self.device, dtype=self.dtype)
        return zero_obs

    def forward(self, obs_dict, delta_obs_dict=None):
        batch_size = None
        features = list()
        
        # process lowdim input
        for key in self.key_shape_map.keys():
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        cond_dict = self.get_zero_obs(batch_size)
        cond_dict.update(delta_obs_dict)

        if not self.use_delta:
            for key in self.key_shape_map.keys():
                if key in self.rot_keys:
                    cond_dict[key] = torch.tensor(np.stack([
                        T.quat_multiply(
                            cond_dict[key][i].tolist(),
                            obs_dict[key][i].tolist()
                        ) for i in range(cond_dict[key].shape[0])
                    ], axis=0)).to(device=self.device, dtype=self.dtype)
                else:
                    cond_dict[key] = cond_dict[key] + obs_dict[key]

        for key in self.key_shape_map.keys():
            data = cond_dict[key]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_output = self.forward(self.get_zero_obs(batch_size=1), self.get_zero_obs(batch_size=1))
        output_shape = example_output.shape[1:]
        return output_shape

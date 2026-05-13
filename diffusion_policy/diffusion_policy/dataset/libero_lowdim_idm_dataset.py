from typing import Dict, List, Any, Optional
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from diffusion_policy.common.libero_utils import get_env_details, update_demo_keys, LANG_EMBED_CACHE_FILE
register_codecs()

import robosuite.utils.transform_utils as T

class LIBEROLowdimIDMDataset(BaseLowdimDataset):
    def __init__(self,
            shape_meta: dict,
            benchmark_name: str='libero_90',
            task_indices: str='1',
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            num_demos_per_task=None,
        ):
        if abs_action:
            # currently abs action not supported by libero
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        env_details = get_env_details(
            benchmark_name=benchmark_name,
            task_indices=task_indices,
        )
        dataset_paths = env_details['dataset_paths']
        env_metas = env_details['env_metas']

        # merge demos
        def load_demos():
            demos = {}
            index = 0
            print('Loading demos from HDF5 files.', flush=True)
            for dataset_path, env_meta in zip(dataset_paths, env_metas):
                with h5py.File(dataset_path, 'r') as file:
                    n_demos = max(0, min(len(file['data']), int(num_demos_per_task))) if num_demos_per_task is not None else len(file['data'])
                    for i in tqdm(range(n_demos), desc=f'Loading demos from {os.path.basename(dataset_path)}'):
                        demos[f'demo_{index}'] = update_demo_keys(file['data'][f'demo_{i}'], low_dim_only=True)
                        index += 1
            return demos
        
        replay_buffer = _convert_libero_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            demos=load_demos())
        print(f"Dataset has {replay_buffer.n_episodes} episodes, {replay_buffer.n_steps} steps.", flush=True)

        pos_keys = list()
        rot_keys = list()
        qpos_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'qpos')
            if type == 'qpos':
                qpos_keys.append(key)
            elif type == 'pos':
                pos_keys.append(key)
            elif type == 'rot':
                # raise NotImplementedError("Rotation type is not supported in InverseDynamicsStateEncoder.")
                rot_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.pos_keys = pos_keys
        self.rot_keys = rot_keys
        self.qpos_keys = qpos_keys
        self.lowdim_keys = sorted(pos_keys + rot_keys + qpos_keys)
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        # if self.abs_action:
        #     if stat['mean'].shape[-1] > 10:
        #         # dual arm
        #         this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
        #     else:
        #         this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
        #     if self.use_legacy_normalizer:
        #         this_normalizer = normalizer_from_stat(stat)
        # else:
        #     # already normalized
        this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            # no normalize in inverse dynamics
            this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)

        obs_dict = dict()
        delta_obs_dict = dict()

        for key in self.pos_keys + self.qpos_keys:
            obs = data[key][0]
            goal_obs = data[key][-1]
            delta_obs = goal_obs - obs
            obs_dict[key] = obs
            delta_obs_dict[key] = delta_obs
        for key in self.rot_keys:
            obs = data[key][0]
            goal_obs = data[key][-1]
            delta_obs = T.quat_distance(goal_obs, obs)
            obs_dict[key] = obs
            delta_obs_dict[key] = delta_obs
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'][:-1].astype(np.float32)),
            'delta_obs': dict_apply(delta_obs_dict, torch.from_numpy)
        }
        return torch_data


def _convert_libero_to_replay(store, shape_meta, demos, n_workers=None, 
            max_inflight_tasks=None, num_demos_per_task=None) -> ReplayBuffer:

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        lowdim_keys.append(key)
        

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)


    episode_ends = list()
    prev_end = 0
    for i in range(len(demos)):
        demo = demos[f'demo_{i}']
        episode_length = demo['actions'].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)

    # save lowdim data
    for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
        data_key = 'obs/' + key
        if key == 'action':
            data_key = 'actions'
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            this_data.append(demo[data_key][:].astype(np.float32))
        this_data = np.concatenate(this_data, axis=0)
        if key == 'action':
            # this_data = _convert_actions(
            #     raw_actions=this_data,
            #     abs_action=abs_action,
            #     rotation_transformer=rotation_transformer
            # )
            assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
        else:
            assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype
        )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

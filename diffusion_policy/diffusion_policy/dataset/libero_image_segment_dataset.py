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
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
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

from diffusion_policy.common.libero_utils import CACHE_DIR
DATA_CACHE_DIR = os.path.join(CACHE_DIR, 'data/')

class LIBEROImageSegmentDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            benchmark_name: str='libero_90',
            task_indices: str='1',
            instructions: str=None,
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


        self.lang_embed = False
        if "lang_embed" in shape_meta['obs']:
            from diffusion_policy.model.vision.model_getter import get_language_model
            self.lang_embed = True
            self.lang_encode_fn = get_language_model()

        env_details = get_env_details(
            benchmark_name=benchmark_name,
            task_indices=task_indices,
        )
        dataset_paths = env_details['dataset_paths']
        env_metas = env_details['env_metas']

        # merge demos
        def load_demos():
            def process_dataset(dataset_path):
                """Process all demos from a single dataset file."""
                dataset_demos = []
                with h5py.File(dataset_path, 'r') as file:
                    all_demo_keys = list(file['segment_data'].keys())
                    n_demos = all_demo_keys[:min(len(all_demo_keys), int(num_demos_per_task))] if num_demos_per_task is not None else all_demo_keys
                    for i, ep in enumerate(tqdm(n_demos, desc=f'Loading demos from {os.path.basename(dataset_path)}')):
                        if instructions is not None:
                            instruction_list = instructions.split('|')
                            lang = file['segment_data'][ep].attrs['language_instruction']
                            if lang not in instruction_list:
                                continue
                        updated_demo = update_demo_keys(file['segment_data'][ep])
                        if self.lang_embed:
                            lang = file['segment_data'][ep].attrs['language_instruction']
                            lang_embed_cache = dict(np.load(LANG_EMBED_CACHE_FILE)) if os.path.exists(LANG_EMBED_CACHE_FILE) else dict()
                            if lang in lang_embed_cache:
                                lang_embed = lang_embed_cache[lang]
                            else:
                                lang_embed = self.lang_encode_fn(lang)
                                # Thread-safe cache update
                                cache_lock_path = LANG_EMBED_CACHE_FILE + '.lock'
                                with FileLock(cache_lock_path):
                                    # Re-load cache in case other workers updated it
                                    current_cache = dict(np.load(LANG_EMBED_CACHE_FILE)) if os.path.exists(LANG_EMBED_CACHE_FILE) else dict()
                                    current_cache[lang] = lang_embed
                                    np.savez_compressed(LANG_EMBED_CACHE_FILE, **current_cache)
                                    lang_embed_cache = current_cache
                                assert lang_embed.shape == shape_meta['obs']['lang_embed']['shape']
                            updated_demo['obs/lang_embed'] = np.tile(
                                lang_embed[None, :], 
                                (updated_demo['actions'].shape[0], 1)
                            )
                        dataset_demos.append(updated_demo)
                return dataset_demos
            
            demos = []
            print('Loading demos from HDF5 files with parallel dataset processing.', flush=True)

            # Process datasets in parallel
            max_workers = min(len(dataset_paths), 1)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                dataset_futures = [executor.submit(process_dataset, dataset_path) for dataset_path in dataset_paths]
                for future in tqdm(concurrent.futures.as_completed(dataset_futures),
                                   total=len(dataset_futures),
                                   desc="Processing datasets"):
                    dataset_demos = future.result()
                    demos.extend(dataset_demos)
            
            return demos
        

        replay_buffer = None
        if use_cache:
            cache_zarr_path = os.path.join(DATA_CACHE_DIR, (benchmark_name + '-' + str(task_indices) + '*' + str(num_demos_per_task) + 'libero_atomic_segment' + str(instructions) +'.zarr.zip'))
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!', flush=True)
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_libero_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            demos=load_demos())
                        print(f'Saving cache to disk {cache_zarr_path}.', flush=True)
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.', flush=True)
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_libero_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                demos=load_demos())
        print(f"Dataset has {replay_buffer.n_episodes} episodes, {replay_buffer.n_steps} steps.", flush=True)

        rgb_keys = list()
        lowdim_keys = list()
        lang_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'lang':
                lang_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

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
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.lang_keys = lang_keys
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

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('embed'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
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
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_libero_to_replay(store, shape_meta, demos, n_workers=None, 
            max_inflight_tasks=None, num_demos_per_task=None) -> ReplayBuffer:

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    lang_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'lang':
            lang_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)


    episode_ends = list()
    prev_end = 0
    for i in range(len(demos)):
        demo = demos[i]
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
            demo = demos[i]
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
    
    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            print(f'Error in img_copy: {e}')
            return False
    
    with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = 'obs/' + key
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps,h,w,c),
                    chunks=(1,h,w,c),
                    compressor=this_compressor,
                    dtype=np.uint8
                )
                for episode_idx in range(len(demos)):
                    demo = demos[episode_idx]
                    hdf5_arr = demo[data_key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(img_copy, 
                                img_arr, zarr_idx, hdf5_arr, hdf5_idx))
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))

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

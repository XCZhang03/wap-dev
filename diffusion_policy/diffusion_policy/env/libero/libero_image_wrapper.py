import os
from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import OmegaConf
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

class LIBEROImageWrapper(gym.Env):
    def __init__(self, 
        env: BDDLBaseDomain,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        compute_language_embedding=False,
        ):

        self.env = env
        self.parsed_problem = env.parsed_problem
        self.language_instruction = " ".join(env.parsed_problem['language_instruction'])
        self.problem_name = env.parsed_problem['problem_name']
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith('embed'):
                min_value, max_value = -np.inf, np.inf
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

        if 'lang_embed' in self.shape_meta['obs'].keys():
            print('Language embedding observation detected.')
            from diffusion_policy.common.libero_utils import LANG_EMBED_CACHE_FILE
            def embed_lang(instruction: str) -> np.ndarray:
                lang_embed_cache = dict(np.load(LANG_EMBED_CACHE_FILE)) if os.path.exists(LANG_EMBED_CACHE_FILE) else dict()
                lang_embed = lang_embed_cache.get(instruction, None)
                if lang_embed is not None:
                    print('Loaded language embed from cache.')
                if lang_embed is None:
                    if not compute_language_embedding:
                        raise RuntimeError('Language embed not found in cache.')
                    from diffusion_policy.model.vision.model_getter import get_language_model
                    lang_encode_fn = get_language_model()
                    lang_embed = lang_encode_fn(instruction).astype(np.float32)
                    lang_embed_cache[self.language_instruction] = lang_embed
                    np.savez_compressed(LANG_EMBED_CACHE_FILE, **lang_embed_cache)
                return lang_embed
            self.embed_fn = embed_lang
            self.set_lang_embed(self.language_instruction)

    def set_lang_embed(self, instruction: str):
        self.language_instruction = instruction
        if "lang_embed" in self.shape_meta['obs'].keys():
            print(f"Setting language instruction to: {instruction}")
            self.lang_embed = self.embed_fn(instruction)
            return instruction

    def get_observation(self, raw_obs=None, return_raw=False):
        if raw_obs is None:
            raw_obs = self.env._get_observations()
        if return_raw:
            return raw_obs

        self.render_cache = raw_obs[self.render_obs_key][::-1]

        obs = dict()
        for key in self.observation_space.keys():
            if key.endswith('image'):
                obs[key] = np.moveaxis(raw_obs[key][::-1], -1, 0) / 255.0
            elif key == 'lang_embed':
                obs[key] = self.lang_embed
            else:
                obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset_to(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)
        self.env.sim.forward()
        self.env._check_success()
        self.env._post_process()
        self.env._update_observables(force=True)
        raw_obs = self.env._get_observations()
        return raw_obs

    def reset(self, **kwargs):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.reset_to(self.init_state)
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.reset_to(self.seed_state_map[seed])
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.sim.get_state().flatten()
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = self.render_cache
        return img


# def test():
if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/diffusion_policy/diffusion_policy/config/task/libero_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    from diffusion_policy.common.libero_utils import get_env_details

    env_details = get_env_details(
        benchmark_name=cfg['env_runner']['benchmark_name'],
        task_indices=cfg['env_runner']['task_indices']
    )
    env_meta = env_details['env_metas'][0]
    
    from libero.libero.envs.env_wrapper import ControlEnv
    env_kwargs = {
        "bddl_file_name": env_meta['bddl_file_name'],
        "camera_heights": shape_meta['obs']['agentview_image']['shape'][1],
        "camera_widths": shape_meta['obs']['agentview_image']['shape'][2],
        "camera_segmentations": None,
        "ignore_done": False,
        "hard_reset": False,
    }
    env = ControlEnv(**env_kwargs).env

    wrapper = LIBEROImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    from PIL import Image
    Image.fromarray(img).save('test_libero_image_wrapper.png')


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])

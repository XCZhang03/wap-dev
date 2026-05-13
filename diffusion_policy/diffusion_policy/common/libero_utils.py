import os
from typing import Dict, Any
import numpy as np
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.env_wrapper import ControlEnv
import libero.libero.utils.utils as libero_utils
import robomimic.utils.file_utils as FileUtils
import robosuite.utils.transform_utils as T
from libero.libero.envs.bddl_utils import robosuite_parse_problem

MAX_INSTRUCTION_LENGTH = 256


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ckpts")
os.makedirs(CACHE_DIR, exist_ok=True)

LANG_EMBED_CACHE_FILE = os.path.join(CACHE_DIR, 'lang_embed_cache.npz')


def parse_task_indices(task):
        """
        Parse task into a list of integer task indices.
        Supports:
          - a single integer or string like "3"
          - a comma-separated list like "1,2,5"
          - ranges with '-' like "2-5" (inclusive) or "5-2" (descending)
          - a mix like "1,3-5,7"
          - a list/tuple of ints
        Returns a deduplicated list preserving the first-seen order.
        """
        tasks = []
        if isinstance(task, (list, tuple)):
            tasks = [int(t) for t in task]
        else:
            task_str = str(task).strip()
            if ',' in task_str or '-' in task_str:
                for part in [p.strip() for p in task_str.split(',') if p.strip()]:
                    if '-' in part:
                        start_s, end_s = [x.strip() for x in part.split('-', 1)]
                        start, end = int(start_s), int(end_s)
                        if start <= end:
                            tasks.extend(range(start, end + 1))
                        else:
                            tasks.extend(range(start, end - 1, -1))
                    else:
                        tasks.append(int(part))
            else:
                tasks = [int(task_str)]

        # Deduplicate while preserving order
        seen = set()
        out = []
        for x in tasks:
            if x not in seen:
                seen.add(x)
                out.append(x)

        if not out:
            raise ValueError(f"Could not parse task: {task!r}")

        return out

def get_env_details(
        benchmark_name,
        task_indices,
):
    """
    Returns the path to the Libero Robomimic dataset and environment metadata.

    Args:
        benchmark_name (str, optional): The name of the benchmark. Defaults to "libero_90".
        task_index (int, optional): The index of the task in the benchmark. Defaults to 0.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    task_indices = parse_task_indices(task_indices)
    benchmark = get_benchmark(benchmark_name)()
    tasks = [benchmark.get_task(idx) for idx in task_indices]
    demonstration_paths = [os.path.join(get_libero_path("datasets"), benchmark.get_task_demonstration(idx)) for idx in task_indices]

    env_metas = [FileUtils.get_env_metadata_from_dataset(demo_path) for demo_path in demonstration_paths]
    for task_index, env_meta in zip(task_indices, env_metas):
        env_meta['bddl_file_name'] = benchmark.get_task_bddl_file_path(task_index)
        env_meta['parsed_problem'] = robosuite_parse_problem(env_meta['bddl_file_name'])
        env_meta['parsed_problem']['language_instruction'] = " ".join(env_meta['parsed_problem']['language_instruction'])

    return {
        "dataset_paths": demonstration_paths,
        "env_metas": env_metas,
        "tasks": tasks,
        "benchmark": benchmark,
    }

def create_env(env_meta, shape_meta, enable_render=True, empty_env=False):
    # Robosuite's hard reset causes excessive memory consumption.
    # Disabled to run more envs.
    # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
    env_kwargs = {
        "bddl_file_name": env_meta['bddl_file_name'],
        "camera_heights": shape_meta['obs']['agentview_image']['shape'][1],
        "camera_widths": shape_meta['obs']['agentview_image']['shape'][2],
        "camera_segmentations": None,
        "ignore_done": False,
        "hard_reset": False,
        "use_camera_obs": enable_render,
        "has_offscreen_renderer": enable_render,
        "has_renderer": enable_render,
        "camera_names": ["agentview", "frontview", "sideview", "birdview", "robot0_eye_in_hand", "canonical_frontview"],
    }
    env = ControlEnv(**env_kwargs).env

    if empty_env:
        import robosuite as suite
        empty_env_kwargs = env_meta['env_kwargs'].copy()
        empty_env_kwargs['env_name'] = "SingleArmEmptyEnv"
        empty_env_kwargs['hard_reset'] = False
        empty_env_kwargs['has_offscreen_renderer'] = False
        empty_env_kwargs['has_renderer'] = False
        empty_env_kwargs['use_camera_obs'] = False
        empty_env_kwargs['camera_names'] = ['agentview', 'frontview', 'sideview', 'birdview', 'canonical_frontview', 'robot0_eye_in_hand']
        empty_env_kwargs['camera_heights'] = shape_meta['obs']['agentview_image']['shape'][1]
        empty_env_kwargs['camera_widths'] = shape_meta['obs']['agentview_image']['shape'][2]
        empty_env_kwargs['robots'] = [type(robot.robot_model).__name__ for robot in env.robots]
        empty_env = suite.make(**empty_env_kwargs)
        empty_env.copy_env_model(env)
         
        return env, empty_env

    return env

def update_demo_keys(demo, low_dim_only=False) -> Dict[str, Any]:
        KEYS_MAP = {
            "agentview_rgb": "agentview_image",
            "eye_in_hand_rgb": "robot0_eye_in_hand_image",
            "joint_states": "robot0_joint_pos",
            "ee_pos": "robot0_eef_pos",
            "ee_ori": "robot0_eef_quat",
            "gripper_states": "robot0_gripper_qpos",
        }
        new_demo = {}

        obs = demo['obs']
        new_obs = {}
        for key in obs.keys():
            if key in KEYS_MAP.values():
                continue
            if key in KEYS_MAP:
                if key.endswith('rgb'):
                    if low_dim_only:
                        continue
                    new_obs[KEYS_MAP[key]] = np.array(obs[key])[:, ::-1, :, :]  # flip image
                elif key.endswith('ori'):
                    # handles orientation representation conversion if needed
                    new_obs['robot0_eef_quat'] = [T.axisangle2quat(ori) for ori in obs['ee_ori'][:]]
                    new_obs['robot0_eef_quat'] = np.array(new_obs['robot0_eef_quat'], dtype=np.float32)
                else:
                    new_obs[KEYS_MAP[key]] = np.array(obs[key])

        for key in new_obs:
            new_demo_key = 'obs/' + key
            new_demo[new_demo_key] = new_obs[key]
        
        new_demo['rewards'] = np.array(demo['rewards'])
        new_demo['actions'] = np.array(demo['actions'])
        new_demo['dones'] = np.array(demo['dones'])
        new_demo['states'] = np.array(demo['states'])

        return new_demo

if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import h5py
    cfg_path = os.path.expanduser('/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/diffusion_policy/diffusion_policy/config/task/libero_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    env_details = get_env_details(
        benchmark_name=cfg['env_runner']['benchmark_name'],
        task_indices=cfg['env_runner']['task_indices']
    )
    env_meta = env_details['env_metas'][0]

    env = create_env(env_meta, shape_meta, enable_render=True)

    f = h5py.File(env_details['dataset_paths'][0], 'r')
    demo = f['data']['demo_0']
    new_demo = update_demo_keys(demo)

    from diffusion_policy.env.libero.libero_image_wrapper import LIBEROImageWrapper
    env = LIBEROImageWrapper(
        env=env,
        shape_meta=shape_meta,
        init_state=new_demo['states'][70],
    )
    obs = env.reset()
    print(env.env.sim.get_state().flatten()-env.init_state)

    print(obs['robot0_eef_quat'])
    print(new_demo['obs/robot0_eef_quat'][:][70])

    print(obs['robot0_eef_pos'])
    print(new_demo['obs/robot0_eef_pos'][:][70])

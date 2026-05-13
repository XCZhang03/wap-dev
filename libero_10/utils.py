import os
import pathlib

import numpy as np
import robosuite.utils.transform_utils as T
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from diffusion_policy.common.dp_utils import embed_lang, load_checkpoint, resize_with_pad, convert_to_uint8

CAMERA_NAMES = ["agentview", "birdview", "robot0_eye_in_hand", "sideview"]
LIBERO_ENV_RESOLUTION = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

policy = None
cfg = None
idm = None
idm_cfg = None
idm_2 = None
idm_2_cfg = None
subtask_embeddings = None

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_names": CAMERA_NAMES}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _get_empty_env(task, env):
    """Create a headless 'empty' environment that shares the physics model with *env*.

    Used for camera-info queries and world-model simulation without full
    rendering overhead.
    """
    import robosuite
    dataset_file = os.path.join(get_libero_path("datasets"), f"{task.problem_folder}/{task.name}_demo.hdf5")
    import h5py
    import json
    f = h5py.File(dataset_file, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    empty_env_kwargs = env_meta['env_kwargs'].copy()
    empty_env_kwargs['env_name'] = "SingleArmEmptyEnv"
    empty_env_kwargs['hard_reset'] = False
    empty_env_kwargs['ignore_done'] = True
    empty_env_kwargs['has_offscreen_renderer'] = False
    empty_env_kwargs['has_renderer'] = False
    empty_env_kwargs['use_camera_obs'] = False
    empty_env_kwargs['camera_names'] = CAMERA_NAMES
    empty_env_kwargs['camera_heights'] = LIBERO_ENV_RESOLUTION
    empty_env_kwargs['camera_widths'] = LIBERO_ENV_RESOLUTION
    empty_env_kwargs['robots'] = [type(robot.robot_model).__name__ for robot in env.robots]
    empty_env = robosuite.make(**empty_env_kwargs)
    empty_env.copy_env_model(env)
    return empty_env


checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/world-action-planner/diffusion_policy/ckpts/libero_90/dp.ckpt"
idm_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/world-action-planner/diffusion_policy/ckpts/libero_90/idm_long.ckpt"
idm_2_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/world-action-planner/diffusion_policy/ckpts/libero_90/idm_short.ckpt"


def _load_policy():
    global policy, cfg
    if policy is None:
        policy, cfg = load_checkpoint(checkpoint_path)
        policy = policy.to(DEVICE)
    return policy, cfg

def to_torch(image):
    image = resize_with_pad(image, 128, 128)
    return np.moveaxis(image[::-1], -1, 0) / 255.0


def policy_fn(obs, subtask_id=0):
    policy, cfg = _load_policy()
    np_obs_dict = dict(obs)
    if "lang_embed" in cfg.shape_meta.obs:
        if subtask_embeddings is None:
            raise RuntimeError("subtask_embeddings must be set before calling policy_fn.")
        np_obs_dict["lang_embed"] = subtask_embeddings[subtask_id]
    obs_keys = cfg.shape_meta.obs.keys()
    np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
    np_obs_dict = {k: to_torch(v) if "image" in k else v for k, v in np_obs_dict.items()}
    obs_dict = {k: torch.from_numpy(v).to(DEVICE).unsqueeze(0).unsqueeze(0) for k, v in np_obs_dict.items()}
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)
    np_action_dict = {k: v.cpu().numpy() for k, v in action_dict.items()}
    action = np_action_dict['action_pred'][0]
    return action


def _load_idm():
    global idm, idm_cfg
    if idm is None:
        idm, idm_cfg = load_checkpoint(idm_checkpoint_path)
        idm = idm.to(DEVICE)
    return idm, idm_cfg


def idm_fn(obs, target_pos, target_quat=None):
    idm, idm_cfg = _load_idm()
    np_obs_dict = dict(obs)
    obs_keys = idm_cfg.shape_meta.obs.keys()
    np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
    delta_obs_dict = {"robot0_eef_pos": target_pos - obs['robot0_eef_pos']}
    if target_quat is not None:
        delta_obs_dict['robot0_eef_quat'] = T.quat_distance(target_quat, obs['robot0_eef_quat'])
    obs_dict = {k: torch.from_numpy(v).to(DEVICE).unsqueeze(0) for k, v in np_obs_dict.items()}
    delta_obs_dict = {k: torch.from_numpy(v).to(DEVICE).unsqueeze(0) for k, v in delta_obs_dict.items()}
    with torch.no_grad():
        action_dict = idm.predict_action(obs_dict, delta_obs_dict)
    np_pred_action = action_dict['action_pred'].cpu().numpy()[0]
    return np_pred_action


def _load_idm_2():
    global idm_2, idm_2_cfg
    if idm_2 is None:
        idm_2, idm_2_cfg = load_checkpoint(idm_2_checkpoint_path)
        idm_2 = idm_2.to(DEVICE)
    return idm_2, idm_2_cfg


def idm_fn_2(obs, target_pos, target_quat=None):
    idm_2, idm_2_cfg = _load_idm_2()
    np_obs_dict = dict(obs)
    obs_keys = idm_2_cfg.shape_meta.obs.keys()
    np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
    delta_obs_dict = {"robot0_eef_pos": target_pos - obs['robot0_eef_pos']}
    if target_quat is not None:
        delta_obs_dict['robot0_eef_quat'] = T.quat_distance(target_quat, obs['robot0_eef_quat'])
    obs_dict = {k: torch.from_numpy(v).to(DEVICE).unsqueeze(0) for k, v in np_obs_dict.items()}
    delta_obs_dict = {k: torch.from_numpy(v).to(DEVICE).unsqueeze(0) for k, v in delta_obs_dict.items()}
    with torch.no_grad():
        action_dict = idm_2.predict_action(obs_dict, delta_obs_dict)
    np_pred_action = action_dict['action_pred'].cpu().numpy()[0]
    return np_pred_action

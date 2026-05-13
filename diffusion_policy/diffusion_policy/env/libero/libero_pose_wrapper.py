import os
from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from diffusion_policy.env.libero.libero_image_wrapper import LIBEROImageWrapper
from robosuite.environments.manipulation.empty_env import SingleArmEmptyEnv

class LIBEROPoseWrapper(LIBEROImageWrapper):
    def __init__(self,
        empty_env: SingleArmEmptyEnv,
        env: BDDLBaseDomain,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        **kwargs
        ):

        super().__init__(
            env=env,
            shape_meta=shape_meta,
            init_state=init_state,
            render_obs_key=render_obs_key,
            **kwargs
        )
        self.empty_env = empty_env

    @property
    def unwrapped(self):
        return self.env

    def set_robot(self, robot_state=None):
        return self.empty_env.copy_robot_state(env=self.env if robot_state is None else None, robot_state=robot_state)
    
    def get_robot_state(self):
        return self.empty_env.get_robot_state(env=self.env)
    
    def get_simulation_robot_state(self):
        return self.empty_env.get_robot_state()
    
    def to_torch(self, image):
        image = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0  # HWC to CHW
        return image
    
    def to_PIL(self, image):
        image = (np.moveaxis(image, 0, -1) * 255.0).astype(np.uint8)  # CHW to HWC
        return image

    def render_simulation_pose(self):
        action_poses = {}
        for cam_name, cam_info in self.empty_env.get_camera_info(self.env).items():
            pose_image = self.empty_env.plot_pose(cam_info['camera_transform'], height=cam_info['camera_height'], width=cam_info['camera_width'])
            action_poses[f"{cam_name}_image"] = self.to_torch(pose_image)
        self.pose_render_cache = action_poses[self.render_obs_key]
        return action_poses

    def simulation_step(self, action):
        raw_obs, reward, done, info = self.empty_env.step(action)
        action_poses = self.render_simulation_pose()
        raw_obs.update(action_poses)
        return raw_obs

    def reset(self, **kwargs):
        self.empty_env.reset()
        returns = super().reset()
        self.set_robot()
        return returns

    def sim_render(self):
        if self.pose_render_cache is None:
            self.render_simulation_pose()
        img = self.pose_render_cache
        img = self.to_PIL(img)
        return img

    def get_camera_info(self):
        return self.empty_env.get_camera_info(self.env)

    
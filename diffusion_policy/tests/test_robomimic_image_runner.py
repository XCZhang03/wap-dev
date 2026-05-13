import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.env_runner.libero_image_runner import LIBEROImageRunner

def test():
    import os
    from omegaconf import OmegaConf
    # cfg_path = os.path.expanduser('diffusion_policy/config/task/lift_image_abs.yaml')
    cfg_path = os.path.expanduser('diffusion_policy/config/task/libero_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    cfg['n_obs_steps'] = 1
    cfg['n_action_steps'] = 1
    cfg['past_action_visible'] = False
    runner_cfg = cfg['env_runner']
    # runner_cfg['dataset_path'] = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
    runner_cfg['benchmark_name'] = "libero_90"
    runner_cfg['task_indices'] = "1"
    runner_cfg['n_train'] = 1
    runner_cfg['n_test'] = 1
    runner_cfg['max_steps'] = 100
    runner_cfg['n_envs'] = 2
    runner_cfg['abs_action'] = False
    del runner_cfg['_target_']
    runner = LIBEROImageRunner(
        **runner_cfg, 
        output_dir='data/scratch/test')

    # import pdb; pdb.set_trace()

    self = runner
    env = self.env
    env.seed(seeds=self.env_seeds)
    env.call_each('run_dill_function', 
        args_list=[(x,) for x in self.env_init_fn_dills[:self.env.num_envs]])
    obs = env.reset()
    env.set_lang_embed('grasp the bowl')
    
    # for i in range(10):
    #     actions = env.action_space.sample() * 0.1
    #     # env_actions = self.undo_transform_action(actions)
    #     env_actions = actions
    #     _ = env.step(env_actions)
    #     _ = env.call_each("simulation_step", args_list=[(env_action[0],) for env_action in env_actions])
    #     print(env.render())
    #     from PIL import Image
    #     poses = env.call("sim_render")
    #     Image.fromarray(poses[0]).save("test_render_pose_0.png")
    #     Image.fromarray(poses[1]).save("test_render_pose_1.png")
    #     # breakpoint()

if __name__ == '__main__':
    test()

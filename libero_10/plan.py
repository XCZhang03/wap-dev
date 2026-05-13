import contextlib
import math
import os
import pathlib
import shutil

import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import torch
import tqdm

import utils as planning_utils
from utils import *
from agent import VLMAgent
from agent_utils import *
from wm_client.wm_env import WMEnv

CAMERA_NAMES = ["agentview", "birdview", "robot0_eye_in_hand", "sideview"]
LIBERO_ENV_RESOLUTION = 224
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]



def run_planning(task_id=0, seed=0, wm_client=None):
    """Run one episode of VLM-guided MPC with candidate-search optimisation.

    Phases:
        1. Execute diffusion policy for subtask 0 until the handoff condition.
        2. Query VLM agent for a goal point and refine via WM-simulated search
           (midpoint → height → endpoint → local candidates).
        3. Resume diffusion policy for subtask 1 to completion.

    Returns:
        bool: True if the task was completed successfully.
    """
    save_dir = os.path.join("scratch_dir/planning_data/", f"task{task_id}", f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    task_suite_name = "libero_10"
    agent = VLMAgent(task_suite_name, task_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = planning_utils._get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    empty_env = planning_utils._get_empty_env(task, env)
    print(f"Task description: {task_description}")

    # -- Subtask language embeddings --
    subtasks = libero10_subtask_map[task_id]
    avail_task_suite = benchmark_dict["libero_90"]()
    subtask_embeddings = []
    subtask_descriptions = []
    for subtask_id in subtasks:
        subtask = avail_task_suite.get_task(subtask_id)
        subtask_description = subtask.language
        print(f"Subtask description: {subtask_description}")
        subtask_embedding = embed_lang(subtask_description)
        subtask_embeddings.append(subtask_embedding)
        subtask_descriptions.append(subtask_description)
    planning_utils.subtask_embeddings = subtask_embeddings

    # ========== ENVIRONMENT INITIALIZATION ==========
    wm_env = WMEnv(env, empty_env, wm_client)

    wm_env.reset()
    num_steps = 0
    subtask_id = 0
    done = False
    replay_images = []
    obs = env.set_init_state(initial_states[seed])
    for t in range(60):
        obs, reward, done, info = wm_env.step(LIBERO_DUMMY_ACTION)
    agent.start_episode(obs)

    # ========== EXECUTION PIPELINE ==========
    # Redirect all prints to log file
    log_file_path = os.path.join(save_dir, f"execution_log_task{task_id}_seed{seed}.txt")
    with open(log_file_path, 'w') as log_file:
        with contextlib.redirect_stdout(log_file):
            # Phase 1: Execute diffusion policy for subtask 0
            print("Phase 1: Executing policy")
            pbar = tqdm.tqdm(total=500, desc="Executing policy ")
            subtask_id = 0
            while not done and num_steps < 400:
                action_chunk = policy_fn(obs, subtask_id=subtask_id)[:10]
                for action in action_chunk:
                    obs, reward, done, info = wm_env.step(action)
                    replay_images.append(obs["agentview_image"][::-1])
                pbar.update(10)
                num_steps += 10
                if num_steps >= subtask_steps[task_id] and agent.verify_subtask_completion(subtask_descriptions, obs):
                    subtask_id = 1
                    break
                else:
                    agent.cache_obs(obs)
            
            # Phase 2: Execute agent Proposal
            print("Phase 2: Executing agent proposal...")
            agent.start_mpc(obs)
            agent_actions = agent.get_action_proposal()
            
            gripper_action = -1
            for action_dict in agent_actions:
                if action_dict["action"] == "RELEASE":
                    for _ in range(10):
                        obs, reward, done, info = wm_env.step(LIBERO_DUMMY_ACTION)
                        replay_images.append(obs["agentview_image"][::-1])
                    pbar.update(10)
                    num_steps += 10
                    gripper_action = -1
                if action_dict["action"] == "MOVE":
                    plot_coordinates_on_image(obs, action_dict['parameters'], os.path.join(save_dir, f"mpc_at{num_steps}.png"))
                    target_point = generate_3d_point(action_dict['parameters'], empty_env.get_camera_info())
                    target_quat = None
                    action_chunk = idm_fn(obs, target_point)
                    action_chunk = update_gripper_action(action_chunk, gripper_action)
                    break

            # Phase 3: Optimise trajectory to goal point

            # -- Step 1: Optimise approach midpoint --
            with wm_env.simulation():
                pred_obs = wm_env.simulate(action_chunk)
                wm_agent_obs = pred_obs['future_obs']
            imageio.mimwrite(os.path.join(save_dir, 'test_dp_output_wm_1.mp4'), pred_obs['WMPredictionOutput'].full_video, fps=20)
            traj_response = agent.optimize_trajectory(wm_agent_obs)
            midpoint_adjustment = optimize_trajectory(traj_response)
            midpoint_obs = wm_agent_obs[20]
            midpoint_obs['robot0_eef_pos'] += midpoint_adjustment
            action_chunk = update_gripper_action(idm_fn_2(obs, midpoint_obs['robot0_eef_pos'], midpoint_obs['robot0_eef_quat']), gripper_action)
            for action in action_chunk:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
                pbar.update(1)
                num_steps += 1
            agent.cache_obs(obs)
            action_chunk = idm_fn_2(obs, target_point, target_quat=target_quat)
            action_chunk = update_gripper_action(action_chunk, gripper_action)

            # -- Step 2: Optimise endpoint height --
            with wm_env.simulation():
                pred_obs = wm_env.simulate(action_chunk)
                wm_agent_obs = pred_obs['future_obs']
            imageio.mimwrite(os.path.join(save_dir, 'test_dp_output_wm_2.mp4'), pred_obs['WMPredictionOutput'].full_video, fps=20)
            height_response = agent.optimize_height(wm_agent_obs)
            endpoint_adjustment = np.array([0, 0, height_response]) * 0.06
            target_point += endpoint_adjustment
            action_chunk = idm_fn_2(obs, target_point, target_quat=target_quat)
            action_chunk = update_gripper_action(action_chunk, gripper_action)

            # -- Step 3: Optimise endpoint position --
            with wm_env.simulation():
                pred_obs = wm_env.simulate(action_chunk)
                wm_agent_obs = pred_obs['future_obs']
            imageio.mimwrite(os.path.join(save_dir, 'test_dp_output_wm_3.mp4'), pred_obs['WMPredictionOutput'].full_video, fps=20)
            endpoint_response = agent.optimize_endpoint(wm_agent_obs)
            target_point += optimize_endpoint(endpoint_response, scale=0.02)
            action_chunk = idm_fn_2(obs, target_point, target_quat=target_quat)
            action_chunk = update_gripper_action(action_chunk, gripper_action)

            # -- Step 4: Local candidate search --
            candidate_obs = []
            candidate_actions = []
            candidate_points = generate_candidates(target_point)
            for i, candidate_point in enumerate(candidate_points):
                with wm_env.simulation():
                    candidate_action = update_gripper_action(idm_fn_2(obs, candidate_point, target_quat), gripper_action)
                    candidate_actions.append(candidate_action)
                    pred_obs = wm_env.simulate(candidate_action)
                    next_action_chunk = policy_fn(pred_obs['future_obs'][-1], subtask_id=1)[:20]
                    next_obs = wm_env.simulate(next_action_chunk)
                    next_action_chunk = policy_fn(next_obs['future_obs'][-1], subtask_id=1)[:20]
                    next_obs = wm_env.simulate(next_action_chunk)
                imageio.mimwrite(os.path.join(save_dir, f'dp_position_round0_candidate{i}.mp4'), next_obs['WMPredictionOutput'].full_video, fps=20)
                candidate_obs.append(next_obs['future_obs'])
            wristview_ranking = agent.rank_images_wristview(candidate_obs)
            action_chunk = candidate_actions[wristview_ranking[0]]

            # -- Execute optimised trajectory --
            for action in action_chunk:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
            pbar.update(len(action_chunk))
            num_steps += len(action_chunk)
            agent.cache_obs(obs)



            # Phase 4: Resume diffusion policy for subtask 1
            print("Phase 4: Resuming policy after MPC...")
            subtask_id = 1
            while not done and num_steps < 500:
                action_chunk = policy_fn(obs, subtask_id=subtask_id)[:10]
                for action in action_chunk:
                    obs, reward, done, info = env.step(action)
                    replay_images.append(obs["agentview_image"][::-1])
                pbar.update(10)
                num_steps += 10
                agent.cache_obs(obs)
        
            pbar.close()
            print(f"Execution completed. Total steps: {num_steps}, Done: {done}")

    imageio.mimwrite(os.path.join(save_dir, f"replay_task{task_id}_seed{seed}.mp4"), replay_images, fps=20)
    
    print(f"Execution log saved to: {log_file_path}")
    print(f"Replay saved to: {os.path.join(save_dir, f'replay_task{task_id}_seed{seed}.mp4')}")

    print(f"Success {done}")
    return done


if __name__ == "__main__":

    # from wm_client.client import WMClient
    # host = "0.0.0.0"
    # port = 7880
    # wm_client = WMClient(host, port)
    wm_client = None

    for task in [4]:
        num_success = 0
        for seed in range(10):
            success = run_planning(task_id=task, seed=seed, wm_client=wm_client)
            if success:
                num_success += 1
            print(f"Number of successful runs: {num_success}/{seed+1}")
        with open(os.path.join("scratch_dir/mpc_data/test_agent_search", f"task{task}", "result.txt"), "w") as f:
            f.write(f"Success rate: {num_success}/10\n")

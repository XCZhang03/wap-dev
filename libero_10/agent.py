import h5py
import numpy as np
import os
import io
import pathlib
from PIL import Image
from google.genai import types

from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from api import call_api
from agent_utils import *

prompt_base = """- Part 0: Instruction
You are a robotics expert, and you are here given a robot manipulation task.
Please analyze the task and given images, understand the task, and provide correct action proposals or help identify the right action.
"""

prompt_proposal = """- Part 4: Action Proposal
    Now, please propose the next actions for the robot to complete the task. You should complete the task following the order in the instruction. 
    The available atomic actions are:
    1. **MOVE** Here you move the gripper to a target position, and you should point it out in the multiview state images provided below. 
    The position should be represented by x,y pixel coordinates normalized to 0-1000. 
    **REMINDER** To grasp or operate an object, move towards it.
    Examples:
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 480, "y": 320}
    }
    }
    2. **ROTATION** Here you rotate the gripper, and you return the rotation in Euler angles [delta_roll, delta_pitch, delta_yaw] in degrees. 
    ### **REMINDER** 
    The coordinate system and axis are defined as follows: from the frontview camera perspective,
    - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
    - The y axis is pointing to the right, with left being negative y and right being positive y.
    - The z axis is pointing upwards, with down being negative z and upwards being positive z.
    Example:
    {
    "action": "ROTATE",
    "parameters": {
    "delta_roll": 0,
    "delta_pitch": 15,
    "delta_yaw": 0
    }
    }
    3. **RELEASE** Here you release the object by opening the gripper.
    Example:
    {
    "action": "RELEASE",
    "parameters": {}
    }
    4. **GRASP** Here you grasp the object by closing the gripper.
    Example:
    {
    "action": "GRASP",
    "parameters": {}
    }
    **REMINDER** You should not return a single release acion in the final action list.
    Finally, return a list of actions in the order of execution.
    For example, 
    [
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 480, "y": 320}
    }
    },
    {
    "action": "ROTATION",
    "parameters": {
    "delta_roll": 0,
    "delta_pitch": 15,
    "delta_yaw": 0
    }
    },
    {
    "action": "RELEASE",
    "parameters": {}
    }
    ]
    ```
    """

class VLMAgent:
    def __init__(self, 
                 task_suite_name, 
                 task_id,
                 obs_history_interval=4
                 ):
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.get_task_description()

        self.obs_history_interval = obs_history_interval

        self.use_sideview = self.task_id in view_config['sideview']
        self.use_wristview = self.task_id in view_config['wristview']

    def get_task_description(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        task = task_suite.get_task(self.task_id)

        def _get_libero_env(task, resolution, seed):
            """Initializes and returns the LIBERO environment, along with the task description."""
            task_description = task.language
            CAMERA_NAMES = ["agentview", "birdview", "robot0_eye_in_hand", "sideview", "canonical_frontview"]
            task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_names": CAMERA_NAMES}
            env = OffScreenRenderEnv(**env_args)
            env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
            return env, task_description
        env, self.task_description = _get_libero_env(task, resolution=256, seed=0)
        demonstration_path = os.path.join(get_libero_path("datasets"), task_suite.get_task_demonstration(self.task_id))
        f = h5py.File(demonstration_path, 'r')
        demo = f['data']['demo_0']
        states = np.array(demo['states'])
        env.reset()

        # get start image
        start_obs = env.set_init_state(states[5])
        start_image_agentview = start_obs['agentview_image'][::-1]
        start_image_topview = start_obs['birdview_image'][::-1]
        # get end image
        end_obs = env.set_init_state(states[-1])
        end_image_agentview = end_obs['agentview_image'][::-1]
        end_image_topview = end_obs['birdview_image'][::-1]
        # close env
        env.close()
        del env

        start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_agentview), mime_type='image/jpeg')
        start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_topview), mime_type='image/jpeg')
        end_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_agentview), mime_type='image/jpeg')
        end_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_topview), mime_type='image/jpeg')

        prompt = f"""- Part 1: Task Description
    task instruction: {self.task_description}
        """
        self.task_prompt = [prompt_base+prompt, "Here is the frontview and topview images of the start state of demonstration", start_image_agentview_part, start_image_topview_part, "Here is the frontview and topview images of the end state of demonstration", end_image_agentview_part, end_image_topview_part]

    def start_episode(self, obs):
        self.obs_cache = []
        episode_start_image_topview = obs['birdview_image'][::-1]
        episode_start_image_agentview = obs['agentview_image'][::-1]
        episode_start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_agentview), mime_type='image/jpeg')
        episode_start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_topview), mime_type='image/jpeg')
        prompt = f"""- Part 2: Current Episode Observation
        Here are the frontview start state images of the current episode of the task in Part 1.
        Please understand how to achieve the task goal based on the task description you have already seen.
        """
        self.current_episode_prompt = [prompt, "Here is the frontview start state images of our episode", episode_start_image_agentview_part]

    def cache_obs(self, obs):
        self.obs_cache.append(types.Part.from_bytes(data=numpy_to_jpeg_bytes(obs['agentview_image'][::-1]), mime_type='image/jpeg'))
    

    def verify_subtask_completion(self, subtasks, obs):
        current_frontview_image = obs['agentview_image'][::-1]
        current_frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_frontview_image), mime_type='image/jpeg')
        prompt = f"""- Part 4: Subtask Completion Verification
        Here we have decomposed the task instruction in Part 1 into two subtasks {subtasks[0]} and {subtasks[1]}, where {subtasks[0]} should be completed before {subtasks[1]}.
        The robot has been executing actions, and I need you to verify in the last step whether the first subtask has completed and we can move on to the second subtask, or we are still in the first subtask and need to keep working on it.
        Please analyze the observation history and the current images, and verify whether the first subtask has been completed. If the first subtask has been completed and we can move on to the second subtask, please return 1. If we are still in the first subtask and need to keep working on it, please return 0. 
        return in json format, for example:
        ```json
        [0]
        ```
        ### **REMINDER** When placing an object on a plate, zoom in on the bottom of the object, and verify that it had made contact with the plate. If the object is still being held in the air, then the first subtask is not completed.
        """
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview images of the past history steps"] + self.obs_cache[::self.obs_history_interval] + ["Here is the frontview image of the current step", current_frontview_image_part] + \
            ["Reason about the subtask completion based on the task instructions and current image, and then return the result in json format."]
        response = call_api(content, thinking="low")
        print("API response for subtask completion verification:", response)
        return get_json(response)[0]

    def reflect_on_obs_history(self):
        prompt = f"""- Part 3: Observations History Reflection
        The robot has been executing actions to complete the task, and the key frames of frontview images of the robot's observations during the execution are shown below.
        """
        self.obs_history_prompt = [prompt] + self.obs_cache[::self.obs_history_interval]

    def start_mpc(self, obs):
        current_image_frontview = obs['agentview_image'][::-1]
        current_image_topview = obs['birdview_image'][::-1]
        current_image_sideview = obs['sideview_image'][::-1]
        current_image_wristview = obs['robot0_eye_in_hand_image'][::-1]
        current_image_frontview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_frontview), mime_type='image/jpeg')
        current_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_topview), mime_type='image/jpeg')
        current_image_sideview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_sideview), mime_type='image/jpeg')
        current_image_wristview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_wristview), mime_type='image/jpeg')
        self.mpc_obs_frontview = ["Here is the frontview image of the current state", current_image_frontview_part]
        self.mpc_obs = ["Here is the frontview image of the current state", current_image_frontview_part, "Here is the topview image of the current state", current_image_topview_part, "Here is the sideview image of the current state", current_image_sideview_part, "Here is the wristview image of the current state", current_image_wristview_part]
    
    def get_action_proposal(self):
        self.reflect_on_obs_history()
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt_proposal] + self.mpc_obs + \
            ["Please analyze the previous history, understand the state of the task right now, and return the proposed action sequence in json format."]
        response = call_api(content, thinking=None)
        print("API response:", response)
        output = get_json(response)
        return output

    def optimize_trajectory(self, obs_list):
        prompt = """- Part 4: Optimize trajectory
        Here we provide a robot trajectory trying to comlete the task, and I need you to optimize the trajectory to make it safer and more successful in completing the task.
        I want you to return the direction to move the gripper so that it can be collision-free and ready to complete the task.
        Return the gripper trajectory adjustments in x y z directions, and which part of the trajectory needs to be adjusted:
        The coordinate system and axis are defined as follows: from the frontview camera perspective,
        - The x axis is pointing towards the camera, with away from camera (towards the robot body) being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction.
        You should output your gripper adjustments at the **middle** of the trajectory, so the gripper is collision-free when following the adjusted trajectory.
        Please return your adjustments direction in json format, for example:
        ```json
        {
          "delta_x": 0,
          "delta_y": 0,
          "delta_z": 1
        }
        ```
         where delta_x, delta_y, delta_z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        Here are the guidelines for optimization:
        0. Refelect on the task instruction and previous history, and understand what the trajectory is trying to do, and which part of the task it is trying to complete. This will help you analyze the trajectory and find out potential problems and how to optimize.
        1. Analyze the trajectory frame by frame, and inspect whether there may be potential collisions with any object along the trajectory. 
            - Make sure that the gripper had cleared any previous objects such as cups and backet rims during the movement, especially at the start of the trajectory.
            - If there is any potential collision with rims and objects, adjust the gripper direction, so that the trajectory is fully clear of any obstacles.
            - You should lift the gripper up to avoid potential collision, with delta_z being 1.
        2. Remember you should output the deltas or adjustments of the trajectory. If the trajectory is moving towards the right direction you do not need to further enhance the movement in that direction, you only need to adjust the trajectory when there is potential problem or when the trajectory is not moving towards the right direction.
        """
        frontview_image_part_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
            frontview_image_part_list.append(frontview_image_part)
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt] + ["Here is the frontview images of the trajectory to be optimized"] + frontview_image_part_list[::2] + \
            ["Please analyze the trajectory based on the guidelines and images step by step, and then return the optimization result in json format."]
        response = call_api(content, thinking=None)
        print("API response for trajectory optimization:", response)
        return get_json(response)

    
    def optimize_height(self, obs_list):
        prompt = """- Part 4: Optimize gripper height
        Here I will give you a series of images showing the trajectory of the gripper approaching and trying to grasp an object.
        I want you to identify whether the gripper height need to be adjusted by lifting to avoid collision with the object and ensure a safe grasp.
        Please analyze following the guidelines below:
        1. First, reflect on the task instruction and previous history, and understand what the gripper is trying to do and what is the target object. This will help you analyze whether we should adjust the gripper height.
        2. The gripper should be slightly above the object top to ensure enough room for descending and grasping, which is approximately half the object height.
        3. If the gripper jaws are almost touching the object in the final images and there is potential risk of collision, you should lift the gripper by returning 1
        4. If the gripper is at a safe height, you should return 0.
        Format:
        Please return your gripper height adjustment in json format, for example:
        ```json
        {
            "z": 0
        }
        ```
        where z means you are adjusting along the z axis, with 1 being upwards (lifting) and 0 being no adjustment.
        you should return 0 or 1 in the answer.
        """
        frontview_image_part_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
            frontview_image_part_list.append(frontview_image_part)
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt] + frontview_image_part_list[-4:] + \
        ["Please analyze the trajectory based on the guidelines and images, reason carefully step by step, and then return the result in json format."]
        response = call_api(content, thinking=None)
        print("API response for gripper height optimization:", response)
        return get_json(response)['z']

    def optimize_endpoint(self, obs_list):
        obs = obs_list[-1]
        frontview_image = obs['agentview_image'][::-1]
        frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
        sideview_image = obs['sideview_image'][::-1]
        sideview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(sideview_image), mime_type='image/jpeg')
        topview_image = obs['birdview_image'][::-1]
        topview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(topview_image), mime_type='image/jpeg')
        wristview_image = obs['robot0_eye_in_hand_image'][::-1]
        wristview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(wristview_image), mime_type='image/jpeg')
        prompt = f"""- Part 4: Optimize gripper position
        Here is the gripper position of our next robot action, and I want you to look carefully and analyze the position of the gripper and the object, and optimize the gripper position following the guidelines below.
        Return the gripper trajectory adjustments in x y z directions:
        The coordinate system and axis are defined as follows: from the **frontview** camera perspective,
        - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction; if you need to move the gripper to the right, return 1 in y direction and 0 in x and z direction; if you need to move the gripper towards the camera, return 1 in x direction and 0 in y and z direction.
        ## **Guidelines**:
        0. Understand what the gripper is trying to do, based on your task instruction understandings and the history.
        1. If the gripper is about to grasp an object, the gripper should be aligned and directly above the object.
        2. Only adjust the gripper if it is clearly and **significantly** misaligned with the object, with both gripper jaws outside the object
        Zoom in on the frontview image, and see if the gripper is below or above the object. If the gripper is clearly below and can not grasp, return 1 in z direction. Also, check whether the gripper is **clearly** to the left or right of the object with both **jaws** outside the object, and return the adjustment in y direction.
        {"Zoom in on the wristview image to see if the object is between the jaws of the gripper. If not, move the gripper so that it is above the object and well aligned with the object. For the wrist view image, the left in wristview is the right from the frontview which is +y, so if the object is in the right of the wristview you should move left in the frontview (right in wristview) which is the -y direction, and vice versa. The up in the wristview image is towards the camera, which is +x. The gripper jaws are at the bottom edge of the wristview image, and the object should appear in the bottom part of the wristview image, so if the object is at the top in the image you should move +x and if the object is not visible you should return -x" if self.use_wristview else ""}
        {"Zoom in on the sideview image to see whether the gripper jaws is directly above the object. The left in the sideview image is towards the camera, thus +x. If the gripper jaws are positioned entirely to the right of the object, move in the +x direction, and vice versa. If the gripper partially overlaps with the object, do not adjust the x-direction." if self.use_sideview else ""}
        3. When grasping a cup you should grasp by the rim, so the gripper should be placed above the rim instead of the body center. Do not adjust the y direction unless both jaws are outside of the cup.
        Similarly, when grasping a box, if one of the grippers is above the box, do not adjust it. Adjust the direction only if both gripper jaws are outside the object.
        4. Reason carefully about the object positions, make sure you are looking at the right object, and point to them before reasoning about the spatial relationships.
        {"Cross validate you output from multiple views to ensure the correctness of the directions" if self.use_sideview or self.use_wristview else ""}
        """
        format = """
        ## **Format**:
        Please return your optimization direction in json format, for example:
        ```json
        {
        "x": 0,
        "y": -1,
        "z": 1}
        ```
        where x, y, z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        Only adjust if the gripper is clearly misaligned, such as both the jaws are beside the object. Return all 0 if the object is mostly below the object.
        """
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt+format] + ["Here is the frontview image of the current state", frontview_image_part,] + (["Here is the sideview image of the current state", sideview_image_part] if self.use_sideview else []) + (["Here is the wristview image of the current state", wristview_image_part] if self.use_wristview else []) + \
            ["Please analyze the gripper position based on the guidelines and images step by step, and then return the optimization result in json format as the example above."]
        response = call_api(content, thinking=None)
        print("API response for endpoint optimization:", response)
        return get_json(response)
    
    def verify_rotation(self, candidate_obs_list):
        prompt = """- Part 4: Verify gripper rotation
        Here I give you a set of candidate images that shows the frontview of the robot state, grasping an object with different gripper orientations. I want you to analyze the images and verify which image shows the best gripper orientation for a firm and clear grasp. Please return the id of the best image in json format, for example:
        ```json
        {
        "best_image_id": 0
        }
        ```
        where the best_image_id is the id of the image that shows the best gripper orientation ready for a firm and clear grasp, ranging from 0 to N-1, where N is the total number of candidate images.
        ## Guidelines:
        1. The position of the gripper may be mis-aligned with the target object, so only care about the rotation of the gripper.
        2. When grasping the box, the gripper should be horizontal and grasping the long edges.
        """
        frontview_image_part_list = []
        for obs_list in candidate_obs_list:
            frontview_image = obs_list[-1]['agentview_image'][::-1]
            frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
            frontview_image_part_list.append(frontview_image_part)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview images of the candidates"] + frontview_image_part_list + \
            ["Please analyze the gripper orientations in the images based on the guidelines and images step by step, and then return the verification result in json format as the example above."]
        response = call_api(content, thinking=None)
        print("API response for rotation verification:", response)
        return get_json(response)["best_image_id"]

    def rank_images_wristview(self, candidate_obs_list):
        wristview_image_part_list = []
        for obs_list in candidate_obs_list:
            wristview_image = obs_list[-1]['robot0_eye_in_hand_image'][::-1]
            wristview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(wristview_image), mime_type='image/jpeg')
            wristview_image_part_list.append(wristview_image_part)
        image_ranking_wristview = f""" - Part 4: Trajectory Images Ranking
            Now, I have a set of wristview images showing the robot trying to grasp an object. I want you to rank them from best to worst in terms of whether the grasp is firm and clear. The best image should be a clear and firm grasp at the right place.
            Here are more detailed guidelines:
            1. Reflect on the trajectory history and task instructions, and understand which object is the gripper trying to grasp.
            2. Identify the position of the jaws of the gripper in the wristview image, which is at the bottom. Verify whether the object is being grasped between the jaws clearly.
            {grasp_skills.get(self.task_id, "")}
            ## **REMINDER**:
            1. You should rank all the images following the same standard. If none of them is perfect, you should rank them by which one is closest.
            2. When ranking the later images, refer and reflect the previous candidates to rank them faithfully. For example, if the first image is blurry and the second image is clear, then the second image should be ranked higher than the first image.
            Return the ranking result in json format, for example:
            ```json
            [0, 2, 3, 1, 4, 5]
            ```
            where the ids in the list are the image ids ranked from best to worst, with the first being the best. The range of the ids should be from 0 to N-1, where N is the total number of candidate images.
            """
        content = self.task_prompt + self.current_episode_prompt + [image_ranking_wristview] + wristview_image_part_list + \
            ["Please reason carefully about the gripper and object states in the images one by one, following the guidelines step by step, and rank them from best to worst in json format as example above"]
        response = call_api(content, thinking=None)
        print("API response for wristview image ranking:", response)
        return get_json(response)

        


                                                                              

if __name__ == "__main__":
    task_suite_name = "libero_10"
    task_id = 0
    seed = 0
    
    agent = VLMAgent(task_suite_name, task_id)
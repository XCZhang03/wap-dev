import io
import json
import re
from PIL import Image
import numpy as np

view_name_mapping = {
    "frontview": "agentview",
    "topview": "birdview",
    "sideview": "sideview"
}

def numpy_to_jpeg_bytes(img_array):
    """Convert numpy image array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format='JPEG', quality=90)
    return buf.getvalue()

def get_json(response_text):
    # Use regular expression to extract the JSON part from the response text
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            output = json.loads(json_str)
            return output
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None
    else:
        print("No JSON found in the response.")
        return None


# ------------------------------------------------------------
# Helper: Triangulate a single 3D point from multiple views
# ------------------------------------------------------------
def triangulate_multiview(uv_list, P_list):
    """
    uv_list:  list of (u, v) pixel coords for each camera
    P_list:   list of 3x4 projection matrices

    Returns: 3D point in world coordinates
    """
    A = []

    for (u, v), P in zip(uv_list, P_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])

    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]          # last row
    X = X_h[:3] / X_h[3]  # convert from homogeneous
    return X

def plot_coordinates_on_image(obs, coordinates, save_path=None):
    from PIL import ImageDraw, Image
    for view in coordinates:
        if view == "frontview":
            image = obs['agentview_image'][::-1]
        elif view == "topview":
            image = obs['birdview_image'][::-1]
        elif view == "sideview":
            image = obs['sideview_image'][::-1]
        else:
            continue
        
        x = int(coordinates[view]["x"] / 1000 * image.shape[1])
        y = int(coordinates[view]["y"] / 1000 * image.shape[0])
        
        # Plot a red dot on the image at the (x, y) coordinates
        image_with_dot = Image.fromarray(image)
        draw = ImageDraw.Draw(image_with_dot)
        draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='red')
        
        # Save or display the image with the plotted coordinates
        if save_path:
            image_with_dot.save(f"{save_path}_{view}_with_coordinates.jpg")
        else:
            image_with_dot.save(f"test_dp_{view}_with_coordinates.jpg")

def generate_3d_point(coordinates, camera_info):
    uv_list = []
    p_list = []
    for view in coordinates:
        _view = view_name_mapping[view]
        x = int(coordinates[view]["x"] / 1000 * camera_info[_view]["camera_width"])
        y = int(coordinates[view]["y"] / 1000 * camera_info[_view]["camera_height"])
        uv_list.append((x, y))
        p_list.append(camera_info[_view]["camera_transform"])

    return triangulate_multiview(uv_list, p_list)


def optimize_trajectory(traj_response, scale=0.06):
    adjustment = np.array([traj_response['delta_x'] * 0.01, traj_response['delta_y'] * 0.01, traj_response['delta_z'] * scale])
    return adjustment

def optimize_endpoint(endpoint_response, scale=0.02):
    adjustment = np.array([scale * endpoint_response['x'], scale * endpoint_response['y'], scale * endpoint_response['z']])
    return adjustment


def generate_rotation_candidates():
    candidates = [[1.0, 0.0, 0.0, 0.0], None]
    return candidates

def update_gripper_action(action_chunk, gripper_action):
    if gripper_action == -1:
        action_chunk[:, -1] = -1
    return action_chunk
    

def generate_candidates(target_point):
    candidates = [target_point]
    noise = [[0.05,0,0], [-0.05,0,0], [0,0.05,0], [0,-0.05,0]]
    for n in noise:
        noised_point = target_point + np.array(n)
        candidates.append(noised_point)
    return candidates


view_config = {
    "sideview": [4, 6],
    "wristview": [7,],
}
subtask_steps = {
    0: 100,
    4: 70,
    6: 100,
    7: 110,
    8: 150,
}
libero10_subtask_map = {
    0: [50, 54],
    4: [67, 68],
    6: [72, 70],
    7: [46, 47],
}
subtask_scales = {
    0: 0.5,
    4: 1.0,
    6: 0.5,
    7: 1.0
}
grasp_skills = {
    7: "If we are grasping a box, we should have the center of the box directly between our grippers",
    4: "When grasping the yellow and white mug, we should grasp by the **white** rim instead of the **yellow** rim. If the gripper is centered at the yellow rim, the grasp is failed."
}


if __name__ == "__main__":
    pass
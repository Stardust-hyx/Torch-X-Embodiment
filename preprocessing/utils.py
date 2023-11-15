import numpy as np

DATASETS = [
    # 'fractal20220817_data',
    # 'kuka',
    # 'bridge',
    'taco_play', # 48GB, low-quality obs image (occlusion happens frequently)
    # 'jaco_play',
    'berkeley_cable_routing', # 4.7GB, low-quality language instruction ("route cable")
    'roboturk', # 45GB, low-quality language instruction ("object search", "layout laundry", "create tower")
    # 'nyu_door_opening_surprising_effectiveness',
    'viola', # 10GB, high control frequency and many redundant actions in demonstration
    'berkeley_autolab_ur5', # 76GB, high quality
    'toto', # 127GB, low-quality language instruction ("pour"), high control frequency
    # 'language_table',
]

def taco_play_preprocess(episode):
    list_obs_image = np.array([step['observation']['rgb_static'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_actions = np.array([step['action']['actions'] for step in episode])
    movement_actions = list_actions[:-1, :6]
    # -1 is closed gripper, 1 is open gripper
    gripper_actions = list_actions[:-1, 6]
    # reverse, so that -1 is open gripper, 1 is closed gripper
    gripper_actions = -gripper_actions
    return {
        'robot_and_gripper': ['Franka', 'Franka_Custom_3D_print'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def berkeley_cable_routing_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_obs_robot_state = np.array([step['observation']['robot_state'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    movement_actions = list_obs_robot_state[1:, :6] - list_obs_robot_state[:-1, :6]
    # gripper is always closed.
    gripper_actions = list_obs_robot_state[:-1, 6]
    assert all([abs(x - 0) < 0.1 for x in gripper_actions])
    gripper_actions = np.ones_like(gripper_actions)
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions
    }

def roboturk_preprocess(episode):
    list_obs_image = np.array([step['observation']['front_rgb'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = np.array([step['action']['gripper_closedness_action'] for step in episode]).squeeze(axis=-1)[:-1]
    return {
        'robot_and_gripper': ['Sawyer', 'Sawyer_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def viola_preprocess(episode):
    list_obs_image = np.array([step['observation']['agentview_rgb'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = np.array([step['action']['gripper_closedness_action'] for step in episode])[:-1]
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def berkeley_autolab_ur5_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    # 1 if gripper closing needs to be triggered from an open state, -1 if gripper opening needs to be triggered from a closed state, 0 if no change
    # covert to closedness state so that -1 is open gripper, 1 is closed gripper
    gripper_actions = []
    pre_closedness = -1.0    # assume the gripper is open in the begining
    for step in episode:
        action = step['action']['gripper_closedness_action']
        if action != 0:
            assert pre_closedness == -1 * action
            pre_closedness *= -1
        else:
            assert action == 0
        gripper_actions.append(pre_closedness)
    gripper_actions = np.array(gripper_actions)[:-1]
    
    return {
        'robot_and_gripper': ['UR5', 'Robotiq_2F-85'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def toto_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['observation']['natural_language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    assert list_instruct[0] == 'pour'
    # gripper is always closed.
    gripper_actions = np.array([step['action']['open_gripper'] for step in episode])[:-1]
    assert all([x==False for x in gripper_actions])
    gripper_actions = np.ones_like(gripper_actions, dtype=np.float32)
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

PREPROCESS_FUNCTIONS = {
    "taco_play": taco_play_preprocess,
    "berkeley_cable_routing": berkeley_cable_routing_preprocess,
    "roboturk": roboturk_preprocess,
    "viola": viola_preprocess,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_preprocess,
    "toto": toto_preprocess,
}

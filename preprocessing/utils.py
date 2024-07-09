import numpy as np

DATASETS = [
    # 'fractal20220817_data',
    # 'kuka',
    'bridge',   # 124GB, Original Version of Bridge V2 from Project Website
    'taco_play', # 48GB, low-quality obs image (occlusion happens frequently)
    # 'jaco_play',
    'berkeley_cable_routing', # 4.7GB, low-quality language instruction ("route cable")
    # 'roboturk', # 45GB, low-quality instruction ("object search", "layout laundry", "create tower")
    # 'nyu_door_opening_surprising_effectiveness',
    'viola', # 10GB, high control frequency and many redundant actions in demonstration
    'berkeley_autolab_ur5', # 76GB, high quality
    'toto', # 127GB, low-quality language instruction ("pour"), high control frequency
    # 'language_table',
    # The datasets above were used in the paper 'Open X-Embodiment: Robotic Learning Datasets and RT-X Models>'.
    "stanford_hydra_dataset_converted_externally_to_rlds", # 72.5GB, high quality
    "austin_buds_dataset_converted_externally_to_rlds", # 1.5GB, high quality
    # "nyu_franka_play_dataset_converted_externally_to_rlds", # 5.2GB, low-quality language instruction ("play with the kitchen")
    "maniskill_dataset_converted_externally_to_rlds", # 151GB, low-quality language instruction
    "austin_sailor_dataset_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "berkeley_fanuc_manipulation", # 8.9GB, high quality
    # "bc_z", # 81GB, low quality, do not use it
]


def bridge_preprocess(episode, discard_keywords=['robot', 'end effector']):
    # Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    list_obs_image = np.array([step['observation']['image_0'] for step in episode])[1:]
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    inst = list_instruct[0].numpy().decode('utf-8').lower()
    if any(x in inst for x in discard_keywords):
        return {
            'instruction': inst,
            'movement_actions': [],
        }
    list_actions = np.array([step['action'] for step in episode])[1:]
    # Relabels actions to use reached proprioceptive state
    list_pos = np.array([step["observation"]["state"][:6] for step in episode])[1:]
    movement_actions = list_pos[1:] - list_pos[:-1]
    # 0 is closed gripper, 1 is open gripper
    gripper_actions = list_actions[:-1, 6]
    # centralize and reverse, so that -1 is open gripper, 1 is closed gripper
    gripper_actions = -(gripper_actions-0.5) * 2
    return {
        'robot_and_gripper': ['WidowX', 'WidowX_Default'],
        'instruction': inst,
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

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
    # movement_actions = list_obs_robot_state[1:, :6] - list_obs_robot_state[:-1, :6]
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
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
    inst = list_instruct[0].numpy().decode('utf-8')
    if inst == 'object search':
        return {
            'instruction': inst,
            'movement_actions': [],
        }
    list_world_vector = np.array([step['action']['world_vector'] for step in episode])
    list_rotation_delta = np.array([step['action']['rotation_delta'] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = np.array([step['action']['gripper_closedness_action'] for step in episode]).squeeze(axis=-1)[:-1]
    return {
        'robot_and_gripper': ['Sawyer', 'Sawyer_Default'],
        'instruction': inst,
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

def stanford_hydra_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_actions = np.array([step['action'] for step in episode])
    movement_actions = list_actions[:-1, :6]
    # 0 is open gripper, 1 is closed gripper
    gripper_actions = list_actions[:-1, 6]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = (gripper_actions - 0.5) * 2
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def austin_buds_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    instruction = list_instruct[0].numpy().decode('utf-8')
    assert instruction == "Take the lid off the pot, put the pot on the plate, and use the tool to push to pot to the front of the table."
    instruction = "Take the lid off the pot, put the pot on the plate and use a tool to push the pot to table's front."
    list_actions = np.array([step['action'] for step in episode])
    movement_actions = list_actions[:-1, :6]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = list_actions[:-1, 6]
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': instruction,
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def nyu_franka_play_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_actions = np.array([step['action'] for step in episode])
    movement_actions = list_actions[:-1, 7:13]
    # -1 is closed gripper, 1 is open gripper
    gripper_actions = list_actions[:-1, 13]
    # reverse, so that -1 is open gripper, 1 is closed gripper
    gripper_actions = -gripper_actions
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def maniskill_preprocess(episode, discard_keywords=['goal', 'designated']):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    inst = list_instruct[0].numpy().decode('utf-8')
    if any(x in inst for x in discard_keywords):
        return {
            'instruction': inst,
            'movement_actions': [],
        }
    list_actions = np.array([step['action'] for step in episode])
    movement_actions = list_actions[:-1, :6]
    # -1 is closed gripper, 1 is open gripper
    gripper_actions = list_actions[:-1, 6]
    # reverse, so that -1 is open gripper, 1 is closed gripper
    gripper_actions = -gripper_actions
    return {
        'robot_and_gripper': ['Franka', 'Franka_Default'],
        'instruction': inst,
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def language_table_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    list_actions = np.array([step['action'] for step in episode])
    movement_actions = list_actions[:-1, :6]
    # 1 is closed gripper, -1 is open gripper
    gripper_actions = list_actions[:-1, 6]
    return {
        'robot_and_gripper': ['xArm', 'stick for pushing'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def berkeley_fanuc_manipulation_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step['language_instruction'] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    movement_actions = np.array([step['action'] for step in episode])[:-1]
    # dataset does not store gripper actions, so use gripper state info, where 1 = close, 0 = open
    gripper_actions = np.array([step['observation']['state'] for step in episode])[:-1, 6]
    # 1 is closed gripper, -1 is open gripper
    gripper_actions = (gripper_actions - 0.5) * 2
    return {
        'robot_and_gripper': ['Fanuc Mate', 'Fanuc Mate_Default'],
        'instruction': list_instruct[0].numpy().decode('utf-8'),
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }

def bc_z_preprocess(episode):
    list_obs_image = np.array([step['observation']['image'] for step in episode])
    list_instruct = [step["observation"]["natural_language_instruction"] for step in episode]
    assert all(inst == list_instruct[0] for inst in list_instruct)
    inst = list_instruct[0].numpy().decode('utf-8')
    # discard trajs containing actions not from the teleoperator
    if any(x == 1 for x in [step['observation']['present/autonomous'] for step in episode]):
        return {
            'instruction': inst,
            'movement_actions': [],
        }
    list_world_vector = np.array([step['action']['future/xyz_residual'][:3] for step in episode])
    list_rotation_delta = np.array([step['action']['future/axis_angle_residual'][:3] for step in episode])
    movement_actions = np.concatenate([list_world_vector, list_rotation_delta], axis=1)[:-1]
    # 0 is open gripper, 1 is closed gripper
    gripper_actions = np.array([step['action']['future/target_close'][:1] for step in episode]).squeeze(axis=-1)[:-1]
    # -1 is open gripper, 1 is closed gripper
    gripper_actions = (gripper_actions - 0.5) * 2
    return {
        'robot_and_gripper': ['Google', 'Google_Default'],
        'instruction': inst,
        'obs_images': list_obs_image,
        'movement_actions': movement_actions,
        'gripper_actions': gripper_actions,
    }


PREPROCESS_FUNCTIONS = {
    "bridge": bridge_preprocess,
    "taco_play": taco_play_preprocess,
    "berkeley_cable_routing": berkeley_cable_routing_preprocess,
    "roboturk": roboturk_preprocess,
    "viola": viola_preprocess,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_preprocess,
    "toto": toto_preprocess,
    "language_table": language_table_preprocess,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_preprocess,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_preprocess,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_preprocess,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_preprocess,
    "berkeley_fanuc_manipulation": berkeley_fanuc_manipulation_preprocess,
    "bc_z": bc_z_preprocess,
}

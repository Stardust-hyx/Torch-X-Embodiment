import os
import torch
from torchvision.transforms import v2
import numpy as np
import pickle, json
import argparse
from PIL import Image

import hydra
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env
from tqdm import tqdm

def save_img(img, fpath):
    img = Image.fromarray(img)
    img.save(fpath)
    return

def save_as_gif(images, path='temp.gif'):
    # Render the images as gif and save:
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)

def transform_imgs(imgs, tgt_size):
    # num_img * H * W * C (np.uint8)
    imgs = torch.from_numpy(imgs)
    imgs = imgs.permute((0, 3, 1, 2))
    # num_img * C * tgt_size * tgt_size (torch.uint8)
    imgs = v2.functional.resize(
        imgs,
        size=(tgt_size, tgt_size),
        interpolation=v2.InterpolationMode.BICUBIC,
        antialias=True
    )
    return imgs

def action_statistics(list_actions, log_f):
    num_traj = len(list_actions)
    all_actions = np.concatenate(list_actions)
    num_transition = len(all_actions)
    actions_mean = np.mean(all_actions, axis=0).tolist()
    actions_std = np.std(all_actions, axis=0)
    print(f'[num_transition]: {num_transition}', file=log_f)
    print(f'[avg_len_traj]: {num_transition//num_traj}', file=log_f)
    print(f'[mean]: {actions_mean}', file=log_f)
    print(f'[std]: {actions_std.tolist()}', file=log_f)
    actions_std = np.where(actions_std==0, 1.0, actions_std).tolist()
    return actions_mean, actions_std


def collect_old_traj(in_dir, interval):
    start, end = interval
    list_robot_obs, list_scene_obs, list_obs_image, abs_actions, rel_actions = [], [], [], [], []
    for index in range(start, end+1):
        fname = "episode_%07d.npz" % index
        fpath = os.path.join(in_dir, fname)
        d = np.load(fpath)

        list_robot_obs.append(d['robot_obs'])
        list_scene_obs.append(d['scene_obs'])
        list_obs_image.append(d['rgb_static'])
        # abs_action = d['actions']
        # abs_actions.append(abs_action)
        rel_action = d['rel_actions']
        rel_actions.append(rel_action)

    old_traj = {
        'robot_obs': np.stack(list_robot_obs),
        'scene_obs': np.stack(list_scene_obs),
        'images': np.stack(list_obs_image),
        'rel_actions': np.stack(rel_actions),
    }
    return old_traj

def rerender(env, task_oracle, traj, task, i, skip):
    if not skip:
        obs = env.reset(robot_obs=traj["robot_obs"][0], scene_obs=traj["scene_obs"][0])
        start_info = env.get_info()
        o = obs['rgb_obs']['rgb_static']
    else:
        obs, start_info, o = None, None, None

    list_obs_image = [o]
    movement_actions, gripper_actions = [], []
    success = False

    for rel_action in traj['rel_actions']:
        movement_actions.append(rel_action[:6])
        # reverse, so that -1 is open gripper, 1 is closed gripper
        gripper_actions.append(-rel_action[6])
        if skip:
            continue

        # action = (action[:3], action[3:6], action[6:])
        obs, _, _, current_info = env.step(rel_action)
        o = obs['rgb_obs']['rgb_static']
        list_obs_image.append(o)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            success = True

    # if not success:
    #     save_img(o, f"{i}_obs_{task}.png")
    #     save_img(traj['images'][-1], f"{i}_traj_{task}.png")
    return success, np.stack(list_obs_image), np.stack(movement_actions), np.stack(gripper_actions)


def convert_split(env, task_oracle, split_dir, out_dir, ds_name, split_name, gif_dir, log_f, num_transitions, img_size, skip):    
    split_out_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)
    invalid_trajs_info = []

    if split_name == 'train':
        scene_info_path = os.path.join(split_dir, 'scene_info.npy')
        scene_info = np.load(scene_info_path, allow_pickle=True).item()
        scene_D_start, scene_D_end = scene_info['calvin_scene_D']

    lang_ann_path = os.path.join(split_dir, 'lang_annotations', 'auto_lang_ann.npy')
    lang_ann = np.load(lang_ann_path, allow_pickle=True)
    lang_ann = lang_ann.item()
    instructions = lang_ann['language']['ann']
    tasks = lang_ann['language']['task']
    intervals = lang_ann['info']['indx']

    total_transitions = 0
    all_movement_actions, all_gripper_actions = [], []
    for i, (instruct, task, interval) in enumerate(tqdm(zip(instructions, tasks, intervals))):
        if split_name == 'train' and not (scene_D_start <= interval[0] <= interval[1] <= scene_D_end):
            continue
        traj_name = f'{ds_name}-{split_name}-{i}'
        out_fn = os.path.join(split_out_dir, f"{traj_name}.pkl")
        old_traj = collect_old_traj(split_dir, interval)
        success, list_obs_image, movement_actions, gripper_actions = rerender(env, task_oracle, old_traj, task, i, skip)
        if not skip and not success:
            try:
                interval = (interval[0].item(), interval[1].item())
            except:
                if isinstance(interval, np.ndarray):
                    interval = interval.tolist()
            invalid_trajs_info.append({"id": i, "task": task, "instruct": instruct, "interval": interval})
            print(f'Invalid Trajectory ({task}): {traj_name}')
            if split_name == 'test':
                continue
            if split_name == 'train':
                continue
            # if split_name == 'train' and any(x in task for x in ['led', 'lightbulb', 'lift','push_into_drawer']):
            #     continue
        traj = {
            'robot_and_gripper': ['Franka', 'Franka_Default'],
            'instruction': instruct,
            'task': task,
            'obs_images': list_obs_image,
            'movement_actions': movement_actions,
            'gripper_actions': gripper_actions,
        }

        print(f"{split_dir}.{i}\ntask: {task}\ninstruct: {instruct}\n", flush=True, file=log_f)

        # for computing the mean and std of actions
        all_movement_actions.append(traj['movement_actions'])
        all_gripper_actions.append(traj['gripper_actions'])

        traj_len = len(traj['movement_actions'])
        num_transitions[f"{traj_name}.pkl"] = traj_len
        total_transitions += traj_len

        if skip:
            continue
        
        if i < 10 and gif_dir:
            # for manually checking
            images = [Image.fromarray(image) for image in traj['obs_images']]
            save_as_gif(images, os.path.join(gif_dir, f'{traj_name}.gif'))

        # convert images to tensor and resize
        traj['obs_images'] = transform_imgs(traj['obs_images'], img_size)
        # save traj data as pickle
        with open(out_fn, 'wb') as f:
            pickle.dump(traj, f)
        
    num_transitions[f'{ds_name}-{split_name}'] = total_transitions
    return len(instructions), all_movement_actions, all_gripper_actions, invalid_trajs_info

def convert(env, task_oracle, in_dir, out_dir, img_size=224):
    ds_name = out_dir
    gif_dir = os.path.join('gif', ds_name)
    os.makedirs(gif_dir, exist_ok=True)
    log_fn = ds_name + '.log'
    log_f = open(log_fn, 'w')
    
    num_transitions = dict()
    all_movement_actions, all_gripper_actions = [], []
    all_invalid_trajs_info = dict()
    train_cnt, movement_actions, gripper_actions, invalid_trajs_info = convert_split(env, task_oracle, in_dir+'/training', out_dir, ds_name, 'train', gif_dir, log_f, num_transitions, img_size, skip=False)
    all_movement_actions += movement_actions
    all_gripper_actions += gripper_actions
    all_invalid_trajs_info['training'] = invalid_trajs_info

    eval_cnt, movement_actions, gripper_actions, invalid_trajs_info = convert_split(env, task_oracle, in_dir+'/validation', out_dir, ds_name, 'test', gif_dir, log_f, num_transitions, img_size, skip=False)
    all_movement_actions += movement_actions
    all_gripper_actions += gripper_actions
    all_invalid_trajs_info['validation'] = invalid_trajs_info
            
    print(f'# training trajs: {train_cnt}')
    print(f'# evaluation trajs: {eval_cnt}')
    print(f'Overall Movement Actions:', file=log_f)
    movement_action_mean, movement_action_std = action_statistics(all_movement_actions, log_f)
    print(f'Overall Gripper Action:', file=log_f)
    gripper_action_mean, gripper_action_std = action_statistics(all_gripper_actions, log_f)
    action_meta = {
        "movement_action_mean": movement_action_mean,
        "movement_action_std": movement_action_std,
        "gripper_action_mean": gripper_action_mean,
        "gripper_action_std": gripper_action_std
    }
    action_meta_f = open(os.path.join(out_dir, 'ori_action_meta.json'), 'w')
    json.dump(action_meta, action_meta_f, indent=2)

    num_transitions_f = open(os.path.join(out_dir, 'num_transitions.json'), 'w')
    json.dump(num_transitions, num_transitions_f, indent=2)

    invalid_trajs_info_f = open(os.path.join(out_dir, 'invalid_trajs_info.json'), 'w')
    json.dump(all_invalid_trajs_info, invalid_trajs_info_f, indent=2)
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calvin_conf_dir', type=str, default='../calvin/calvin_models/conf')
    parser.add_argument('--in_dir', type=str, default='task_ABCD_D')
    parser.add_argument('--out_dir', type=str, default='task_ABCD_rerender')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    in_dir = args.in_dir
    out_dir = args.out_dir

    env_conf_dir = os.path.join(in_dir, "training")
    env = get_env(env_conf_dir, show_gui=False)

    calvin_conf_dir = args.calvin_conf_dir
    task_cfg = OmegaConf.load(os.path.join(calvin_conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)

    convert(env, task_oracle, in_dir, out_dir)

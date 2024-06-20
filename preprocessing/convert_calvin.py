import os
import torch
from torchvision.transforms import v2
import numpy as np
import pickle, json
import argparse
from PIL import Image
from tqdm import tqdm

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

def collect_traj(split_dir, interval):
    start, end = interval
    list_obs_image, movement_actions, gripper_actions = [], [], []
    for index in range(start, end+2):
        fname = "episode_%07d.npz" % index
        fpath = os.path.join(split_dir, fname)
        d = np.load(fpath)

        list_obs_image.append(d['rgb_static'])

        action = d['rel_actions']
        movement_actions.append(action[:6])
        # reverse, so that -1 is open gripper, 1 is closed gripper
        gripper_actions.append(-action[6])

    return np.stack(list_obs_image), np.stack(movement_actions), np.stack(gripper_actions)

def convert_split(split_dir, out_dir, ds_name, split_name, gif_dir, log_f, num_transitions, img_size):
    split_out_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    lang_ann_path = os.path.join(split_dir, 'lang_annotations', 'auto_lang_ann.npy')
    lang_ann = np.load(lang_ann_path, allow_pickle=True)
    lang_ann = lang_ann.item()
    instructions = lang_ann['language']['ann']
    tasks = lang_ann['language']['task']
    intervals = lang_ann['info']['indx']

    total_transitions = 0
    all_movement_actions, all_gripper_actions = [], []
    for i, (instruct, task, interval) in enumerate(tqdm(
        zip(instructions, tasks, intervals), total=len(instructions)
    )):
        traj_name = f'{ds_name}-{split_name}-{i}'
        list_obs_image, movement_actions, gripper_actions = collect_traj(split_dir, interval)
        traj = {
            'robot_and_gripper': ['Franka', 'Franka_Default'],
            'instruction': instruct,
            'task': task,
            'obs_images': list_obs_image,
            'movement_actions': movement_actions[:-1],
            'gripper_actions': gripper_actions[:-1],
        }

        print(f"{split_dir}.{i}\ntask: {task}\ninstruct: {instruct}\n", flush=True, file=log_f)

        # for computing the mean and std of actions
        all_movement_actions.append(traj['movement_actions'])
        all_gripper_actions.append(traj['gripper_actions'])
        if i < 5 and gif_dir:
            # for manually checking
            images = [Image.fromarray(image) for image in traj['obs_images']]
            save_as_gif(images, os.path.join(gif_dir, f'{traj_name}.gif'))

        # convert images to tensor and resize
        traj['obs_images'] = transform_imgs(traj['obs_images'], img_size)
        # save traj data as pickle
        out_fn = os.path.join(split_out_dir, f"{traj_name}.pkl")
        with open(out_fn, 'wb') as f:
            pickle.dump(traj, f)

        traj_len = len(traj['movement_actions'])
        num_transitions[f"{traj_name}.pkl"] = traj_len
        total_transitions += traj_len
    
    num_transitions[f'{ds_name}-{split_name}'] = total_transitions
    return len(instructions), all_movement_actions, all_gripper_actions

def convert(in_dir, out_dir, img_size=224):
    ds_name = out_dir
    gif_dir = os.path.join('gif', ds_name)
    os.makedirs(gif_dir, exist_ok=True)
    log_fn = ds_name + '.log'
    log_f = open(log_fn, 'w')
    
    num_transitions = dict()
    all_movement_actions, all_gripper_actions = [], []
    train_cnt, movement_actions, gripper_actions = convert_split(in_dir+'/training', out_dir, ds_name, 'train', gif_dir, log_f, num_transitions, img_size)
    all_movement_actions += movement_actions
    all_gripper_actions += gripper_actions
    eval_cnt, movement_actions, gripper_actions = convert_split(in_dir+'/validation', out_dir, ds_name, 'test', gif_dir, log_f, num_transitions, img_size)
    all_movement_actions += movement_actions
    all_gripper_actions += gripper_actions
            
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
    action_meta_f = open(os.path.join(out_dir, 'action_meta.json'), 'w')
    json.dump(action_meta, action_meta_f, indent=2)

    num_transitions_f = open(os.path.join(out_dir, 'num_transitions.json'), 'w')
    json.dump(num_transitions, num_transitions_f, indent=2)
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='task_ABCD_D')
    parser.add_argument('--out_dir', type=str, default='task_ABCD')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    in_dir = args.in_dir
    out_dir = args.out_dir

    convert(in_dir, out_dir)

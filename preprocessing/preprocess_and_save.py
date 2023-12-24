import os
import pickle
import json
import numpy as np
import tensorflow_datasets as tfds
import torch
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from functools import partial
from absl import app, flags
from utils import DATASETS, PREPROCESS_FUNCTIONS

FLAGS = flags.FLAGS
flags.DEFINE_string("in_dir", "/data1/hyx/ts_datasets/",
                    "where you put datasets downloaded from gs://gresearch/robotics/")
flags.DEFINE_string("out_dir", "/data1/hyx/np_datasets/",
                    "where to put processed datasets")
flags.DEFINE_integer("img_size", 224,
                    "desired image size")
flags.DEFINE_string("gif_dir", "gif",
                    "where to put gif images for checking")
flags.DEFINE_string("logging_dir", "log",
                    "where to put log files for checking")
flags.DEFINE_bool("debug", False,
                  "if debug, only process 20 episodes of each dataset'")
flags.DEFINE_integer("num_workers", 8,
                     "Number of threads to use")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def dataset2path(data_dir, dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return os.path.join(data_dir, dataset_name, version)

def save_as_gif(images, path='temp.gif'):
    # Render the images as gif and save:
    images[0].save(path, save_all=True, append_images=images[1:], duration=300, loop=0)

def set_gpu_memory():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def convert_dataset(in_dir, out_dir, gif_dir, log_dir, debug, args):
    dataset_name, split_name = args
    set_gpu_memory()

    dataset_path = dataset2path(in_dir, dataset_name)
    # skip non-existent dataset
    if not os.path.exists(dataset_path):
        print(f'Cannot find {dataset_name} in {in_dir}')
        return
    
    b = tfds.builder_from_directory(builder_dir=dataset_path)
    all_splits = list(b.info.splits.keys())
    preprocess_func = PREPROCESS_FUNCTIONS[dataset_name]

    out_path = os.path.join(out_dir, dataset_name)
    split_out_path = os.path.join(out_path, split_name)
    # skip converted dataset split
    if os.path.exists(split_out_path):
        print(f'{split_out_path} already exists, skip {dataset_name} {split_name}-split')
        return
    
    os.makedirs(split_out_path, exist_ok=True)
    log_f = open(os.path.join(log_dir, f'{dataset_name}-{split_name}.log'), 'w')
    
    if debug:
        ds = b.as_dataset(split=f'{split_name}[:10]')
    elif len(all_splits) == 1:
        print(f'Only one split {all_splits} is found in {dataset_name}, automatically divding into two splits!')
        whole_split = all_splits[0]
        if split_name == 'train':
            ds = b.as_dataset(split=f'{whole_split}[5%:95%]')
        else:
            ds = b.as_dataset(split=f'{whole_split}[:5%]+{whole_split}[95%:]')
    else:
        try:
            ds = b.as_dataset(split=split_name)
        except:
            assert split_name=='test' and 'val' in all_splits
            ds = b.as_dataset(split='val')


        
    print(f'Processing {dataset_name} {split_name}-split...', file=log_f)
    num_invalid_traj = 0
    all_movement_actions = []
    all_gripper_actions = []
    for i, episode in enumerate(iter(ds)):
        traj = preprocess_func(episode['steps'])
        print(f"{i}\ninstruction: {traj['instruction']}", file=log_f)
        # discard traj with only 1 observation and 0 transition
        # also discard traj with undesired instruction (like the ones in the maniskill dataset)
        if len(traj['movement_actions']) == 0:
            print(f"Invalid trajectory!", file=log_f)
            print(flush=True, file=log_f)
            num_invalid_traj += 1
            continue

        # for computing the mean and std of actions
        all_movement_actions.append(traj['movement_actions'])
        all_gripper_actions.append(traj['gripper_actions'])
        print(f"movement_actions:\n{traj['movement_actions']}", file=log_f)
        print(f"gripper_actions:\n{traj['gripper_actions']}", file=log_f)
        print(flush=True, file=log_f)
        if i < 5 and gif_dir:
            # for manually checking
            images = [Image.fromarray(image) for image in traj['obs_images']]
            gif_path = os.path.join(gif_dir, dataset_name, split_name)
            os.makedirs(gif_path, exist_ok=True)
            save_as_gif(images, os.path.join(gif_path, f'{i}.gif'))

        # convert images to tensor and resize
        traj['obs_images'] = transform_imgs(traj['obs_images'], FLAGS.img_size)
        # save traj data as pickle 
        out_fn = os.path.join(split_out_path, f"{dataset_name}-{split_name}-{i}.pkl")
        with open(out_fn, 'wb') as f:
            pickle.dump(traj, f)

    num_valid_traj = len(all_movement_actions)
    print(f'# Invalid Traj: {num_invalid_traj}', file=log_f)
    print(f'# Valid Traj(saved): {num_valid_traj}', file=log_f)
    print(f'# Total Traj: {num_invalid_traj+num_valid_traj}', file=log_f)
    print(f'Overall Movement Actions:', file=log_f)
    movement_action_mean, movement_action_std = action_statistics(all_movement_actions, log_f)
    print(f'Overall Gripper Action:', file=log_f)
    gripper_action_mean, gripper_action_std = action_statistics(all_gripper_actions, log_f)
    # save action meta data
    if split_name == 'train':
        action_meta = {
            "movement_action_mean": movement_action_mean,
            "movement_action_std": movement_action_std,
            "gripper_action_mean": gripper_action_mean,
            "gripper_action_std": gripper_action_std
        }
        action_meta_f = open(os.path.join(out_path, 'action_meta.json'), 'w')
        json.dump(action_meta, action_meta_f, indent=2)
    print(f'{dataset_name} {split_name}-split is now processed!')
    return

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

def main(_):
    list_args = []
    for dataset_name in DATASETS:
        for split_name in ['train', 'test']:
            list_args.append((dataset_name, split_name))

    os.makedirs(FLAGS.logging_dir, exist_ok=True)

    with Pool(FLAGS.num_workers) as p:
        list(tqdm(
            p.imap(
                partial(
                    convert_dataset,
                    FLAGS.in_dir, FLAGS.out_dir, FLAGS.gif_dir, FLAGS.logging_dir, FLAGS.debug,
                ),
                list_args
            ),
            total=len(list_args)
        ))


if __name__ == '__main__':
    app.run(main)

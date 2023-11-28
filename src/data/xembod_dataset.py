import os
import math
import pickle
import json
import torch
import numpy as np
from copy import deepcopy
from datetime import timedelta
from torch.utils.data import IterableDataset
from .goal_relabeling import GOAL_RELABELING_FUNCTIONS

def repeat_by_weights(list_data_filenames, sample_weights:str):
    repeated_filenames = []
    if sample_weights == None:
        all_data_filenames = sum(list_data_filenames, [])
    elif sample_weights.lower() == 'balance':
        max_samples_per_set = max([len(x) for x in list_data_filenames])
        all_data_filenames = []
        for data_filenames in list_data_filenames:
            repeat_times = max_samples_per_set // len(data_filenames)
            all_data_filenames.extend(data_filenames * repeat_times)
            if repeat_times > 1:
                repeated_filenames.extend(data_filenames)
    else:
        sample_weights = eval(sample_weights)
        assert isinstance(sample_weights, list) and isinstance(sample_weights[0], float)
        ori_total_num = sum([len(x) for x in list_data_filenames])
        all_data_filenames = []
        for target_proportion, data_filenames in zip(sample_weights, list_data_filenames):
            proportion = len(data_filenames) / ori_total_num
            repeat_times = max(int(target_proportion // proportion), 1)
            all_data_filenames.extend(data_filenames * repeat_times)
            if repeat_times > 1:
                repeated_filenames.extend(data_filenames)

    return repeated_filenames, all_data_filenames

def binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    # in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    in_between_mask = ~(open_mask | closed_mask)

    # is_open_float = tf.cast(open_mask, tf.float32)
    is_open_float = open_mask.astype(np.float32)

    # def scan_fn(carry, i):
    #     return tf.cond(
    #         in_between_mask[i],
    #         lambda: tf.cast(carry, tf.float32),
    #         lambda: is_open_float[i],
    #     )

    # new_actions = tf.scan(
    #     scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    # )
    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = is_open_float[i]
        new_actions[i] = carry
    return new_actions


class XEmbodDatasetTorch(IterableDataset):
    def __init__(self, args, local_rank, world_size, is_train=True, is_master=True):
        super().__init__()
        self.args = args
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_train = is_train

        self.goal_relabeling_strategy = args.goal_relabeling_strategy
        self.goal_relabel_offset = args.goal_relabel_offset
        self.goal_relabel_future_step = args.goal_relabel_future_step

        self.prompt_template = args.all_prompts[args.prompt_type]

        datasets = args.datasets
        dataset_paths = [os.path.join(args.data_dir, x) for x in datasets]

        # get the mean and std of actions 
        self.action_metadata = dict()
        list_action_meta_fn = [os.path.join(x, 'action_meta.json') for x in dataset_paths]
        for dataset, action_meta_fn in zip(datasets, list_action_meta_fn):
            action_meta = json.load(open(action_meta_fn))
            action_meta_np = dict([(k, np.array(v)) for k, v in action_meta.items()])
            self.action_metadata[dataset] = action_meta_np

        train_or_eval = 'train' if is_train else 'test'
        dataset_paths = [os.path.join(x, train_or_eval) for x in dataset_paths]
        
        list_data_filenames = []
        for dataset_path in dataset_paths:
            data_filenames = []
            for fn in sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0].split('-')[-1])):
                data_filenames.append(os.path.join(dataset_path, fn))
            list_data_filenames.append(data_filenames)
        ori_all_filenames = sum(list_data_filenames, [])

        repeated_filenames, all_data_filenames = repeat_by_weights(list_data_filenames, args.sample_weights)

        if is_master:
            for dataset_path, data_filenames in zip(dataset_paths, list_data_filenames):
                print(f"{len(data_filenames)} trajs in {dataset_path}")
                
            print(f"[# {train_or_eval} trajs before repeating]: {len(ori_all_filenames)}")
            print(f"[# {train_or_eval} trajs after repeating]: {len(all_data_filenames)}")
            print()

            meta_info_fn = os.path.join(args.save_dir, '_'.join([train_or_eval, 'set', 'meta_info']))
            with open(meta_info_fn, 'w') as f:
                for data_filename in all_data_filenames:
                    print(data_filename, file=f)

        self.all_data_filenames = all_data_filenames
        self.all_data_filenames_shuffled = None

        self.length = len(all_data_filenames) // self.world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            indices = list(range(self.length))
            max_index = indices[-1]
        else:  # multiple workers
            # 保证dataloader的不同线程读取到的是不同数据
            indices = list(range(worker_info.id, self.length, worker_info.num_workers))
            max_index = indices[-1]

        sample_iterator = self._sample_generator(set(indices), max_index)

        return sample_iterator

    def __len__(self):
        return self.length

    def _sample_generator(self, indices, max_index):
        if self.all_data_filenames_shuffled:
            all_data_filenames = self.all_data_filenames_shuffled
        else:
            all_data_filenames = self.all_data_filenames
            
        # 通过gpu数将数据集分割，保证不同gpu读取到不同数据
        data_filenames = all_data_filenames[self.local_rank :: self.world_size]
        for i, data_filename in enumerate(data_filenames):
            if i not in indices:
                if i < max_index:
                    continue
                else:
                    return
            
            with open(data_filename, 'rb') as f:
                traj = pickle.load(f)

            ds_name = data_filename.split('/')[-1].split('-')[0]
            traj = self.process_traj(traj, ds_name)
            prompt, obs_images, goal_images, future_images, actions = self.convert_traj_to_samples(traj)

            for obs_img, goal_img, future_img, action in zip(
                obs_images, goal_images, future_images, actions
            ):
                yield prompt, obs_img, goal_img, future_img, action

    def shuffle(self, seed):
        rng = np.random.default_rng(seed)
        self.all_data_filenames_shuffled = rng.shuffle(
            deepcopy(self.all_data_filenames)
        )

    def process_traj(self, traj, ds_name):
        """
        traj is a dict as
        {
            'robot_and_gripper': ['Franka', 'Franka_Default'],
            'instruction': "open the drawer",
            'obs_images': list_obs_image,
            'movement_actions': movement_actions,
            'gripper_actions': gripper_actions,
        }
        """
        assert len(traj['obs_images']) == len(traj['movement_actions']) + 1
        traj = self._process_actions(traj, ds_name)
        traj = self._add_goals(traj)
        traj = self._construct_prompt(traj)
        return traj
    
    def convert_traj_to_samples(self, traj):
        prompt = traj["prompt"]
        obs_images = traj["obs_images"][:-1]
        goal_images = traj["goal_images"]
        future_images = traj["future_images"]
        actions = traj["actions"]
        return prompt, obs_images, goal_images, future_images, actions
    
    def _process_actions(self, traj, ds_name):
        """ adding field 'actions' to traj """
        movement_actions = traj["movement_actions"]
        gripper_actions = np.expand_dims(traj["gripper_actions"], axis=1)

        # normalize movement actions
        if self.action_metadata is not None:
            movement_actions = (
                movement_actions - self.action_metadata[ds_name]["movement_action_mean"]
            ) / self.action_metadata[ds_name]["movement_action_std"]

        traj["actions"] = np.concatenate([movement_actions, gripper_actions], axis=1)
        return traj

    def _add_goals(self, traj):
        """ adding field 'goal_images' to traj """
        traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
            traj, self.goal_relabel_offset, self.goal_relabel_future_step
        )
        return traj
    
    def _construct_prompt(self, traj):
        """ adding field 'prompt' to traj """
        info = {
            "robot_type": traj["robot_and_gripper"][0],
            "gripper_type": traj["robot_and_gripper"][1],
            "instruct": traj["instruction"],
            "img": self.args.img_placeholder,
            "act": self.args.act_placeholder,
        }
        traj["prompt"] = self.prompt_template.format(**info)
        return traj

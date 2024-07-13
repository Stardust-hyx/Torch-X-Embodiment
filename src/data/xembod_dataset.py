import os
import math
import pickle
import json
import yaml
import torch
import numpy as np
from copy import deepcopy
from datetime import timedelta
from torch.utils.data import IterableDataset
from .goal_relabeling import GOAL_RELABELING_FUNCTIONS


def repeat_by_num_transitions(
        list_data_filenames, list_num_transitions, sample_weights, avg_num_traj, traj_name_2_num_trans, repeat_limit: int=10, discard_ths: int=None
    ):
    if sample_weights == None:
        all_data_filenames = sum(list_data_filenames, [])
    elif sample_weights.lower() == 'balance':
        num_trajs = [len(x) for x in list_data_filenames]
        max_samples_per_set = max(list_num_transitions)
        all_data_filenames = []
        for data_filenames, num_transitions in zip(list_data_filenames, list_num_transitions):
            repeat_times = int(max_samples_per_set / num_transitions - 0.5)
            repeat_times = max(repeat_times, 1)
            repeat_times = (repeat_times + max(avg_num_traj // len(data_filenames), 1)) // 2
            repeat_times = min(repeat_times, repeat_limit)
            all_data_filenames.extend(data_filenames * repeat_times)
            # print(len(data_filenames), max_samples_per_set, num_transitions, repeat_times)
    else:
        sample_weights = eval(sample_weights)
        assert isinstance(sample_weights, list) and isinstance(sample_weights[0], float)
        ori_total_num = sum(list_num_transitions)
        all_data_filenames = []
        for target_proportion, data_filenames, num_transitions in zip(sample_weights, list_data_filenames, list_num_transitions):
            proportion = num_transitions / ori_total_num
            repeat_times = max(int(target_proportion // proportion), 1)
            repeat_times = min(repeat_times, repeat_limit)
            all_data_filenames.extend(data_filenames * repeat_times)

    
    if discard_ths is not None:
        all_data_filenames_ = []
        for data_filename in all_data_filenames:
            traj_name = data_filename.split('/')[-1]
            num_transitions = traj_name_2_num_trans[traj_name]
            if num_transitions >= discard_ths:
                all_data_filenames_.append(data_filename)
        return all_data_filenames_
    else:
        return all_data_filenames


def clean_instruction(instruction: str):
    instruction = instruction.strip()
    instruction = instruction.lower()
    if instruction[-1] == '.':
        instruction = instruction[:-1]
    return instruction
    

class XEmbodDatasetTorch(IterableDataset):
    def __init__(self, args, local_rank, world_size, is_train=True, is_master=True):
        super().__init__()
        self.args = args
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_train = is_train

        # if args.benchmarks and not is_train:
        if not is_train:
            self.goal_relabeling_strategy = 'last'
        else:
            self.goal_relabeling_strategy = args.goal_relabeling_strategy
        self.goal_relabel_offset = args.goal_relabel_offset
        self.goal_relabel_future_step = args.goal_relabel_future_step

        self.prompt_template = args.all_prompts[args.prompt_type]

        self.use_history = args.use_history
        self.num_frames = args.num_frames

        datasets = args.benchmarks.split(',') if args.benchmarks else args.datasets
        dataset_paths = [os.path.join(args.data_dir, x) for x in datasets]

        # get the mean and std of actions 
        self.action_metadata = dict()
        list_action_meta_fn = [os.path.join(x, 'action_meta.json') for x in dataset_paths]
        overall_action_meta_np = None
        for dataset, action_meta_fn in zip(datasets, list_action_meta_fn):
            if self.args.same_action_mean_std and overall_action_meta_np:
                self.action_metadata[dataset] = overall_action_meta_np
            elif os.path.exists(action_meta_fn):
                action_meta = json.load(open(action_meta_fn))
                action_meta_np = dict([(k, np.array(v)) for k, v in action_meta.items()])
                self.action_metadata[dataset] = action_meta_np
                if self.args.same_action_mean_std and overall_action_meta_np is None:
                    overall_action_meta_np = action_meta_np

        # get the number of transitions of each traj
        self.traj_name_2_num_trans = dict()
        for dataset_path in dataset_paths:
            num_trans_dict = json.load(open(os.path.join(dataset_path, 'num_transitions.json')))
            self.traj_name_2_num_trans.update(num_trans_dict)

        train_or_eval = 'train' if is_train else 'test'
        dataset_paths = [os.path.join(x, train_or_eval) for x in dataset_paths]
        
        list_data_filenames = []
        for dataset_path in dataset_paths:
            data_filenames = []
            for fn in sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0].split('-')[-1])):
                data_filenames.append(os.path.join(dataset_path, fn))
            list_data_filenames.append(data_filenames)
        ori_all_filenames = sum(list_data_filenames, [])

        all_data_filenames = repeat_by_num_transitions(
            list_data_filenames,
            [self.traj_name_2_num_trans[x+'-'+train_or_eval] for x in datasets],
            args.sample_weights,
            avg_num_traj=(args.avg_num_traj if is_train else args.avg_num_traj//10),
            traj_name_2_num_trans=self.traj_name_2_num_trans,
            discard_ths=(self.num_frames if self.use_history else None)
        )

        self.hard_tasks = args.hard_tasks
        self.task = args.task.replace('_', ' ') if args.task else None

        if self.args.paraphrase:
            assert os.path.exists(self.args.task_2_instructs_dir)
            self.task_2_instructions = dict()
            for fn in os.listdir(self.args.task_2_instructs_dir):
                d = yaml.safe_load(open(os.path.join(self.args.task_2_instructs_dir, fn)))
                for k, v in d.items():
                    if k in self.task_2_instructions:
                        self.task_2_instructions[k].extend(v)
                    else:
                        self.task_2_instructions[k] = v
            # self.task_2_instructions = yaml.safe_load(open(self.args.task_2_instructs_path))

        if is_master:
            for dataset_path, data_filenames in zip(dataset_paths, list_data_filenames):
                print(f"{len(data_filenames)} trajs in {dataset_path}")
                
            print(f"[# {train_or_eval} trajs before repeating]: {len(ori_all_filenames)}")
            print(f"[# {train_or_eval} trajs after repeating]: {len(all_data_filenames)}")

            meta_info_fn = os.path.join(args.save_dir, '_'.join([train_or_eval, 'set', 'meta_info']))
            with open(meta_info_fn, 'w') as f:
                for data_filename in all_data_filenames:
                    print(data_filename, file=f)

        self.all_data_filenames = all_data_filenames
        self.all_data_filenames_shuffled = None

        self.traj_idx_2_samples_range = dict()
        self.num_all_transitions = self.cnt_transitions(all_data_filenames)
        if is_master:
            print(f"[# transitions]: {self.num_all_transitions}")
            print()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            start_index = 0
            indices = list(range(self.local_rank, self.num_all_transitions, self.world_size))
            max_index = indices[-1]
        else:  # multiple workers
            # 保证dataloader的不同线程读取到的是不同数据
            chunk_size = self.num_all_transitions // worker_info.num_workers
            start_index = chunk_size * worker_info.id
            indices = list(range(start_index + self.local_rank, start_index + chunk_size, self.world_size))
            max_index = indices[-1]

        sample_iterator = self._sample_generator(start_index, max_index)

        return sample_iterator

    def _sample_generator(self, start_index, max_index):
        if self.is_train:
            assert self.all_data_filenames_shuffled is not None, "Please shuffle the training dataset!"
            all_data_filenames = self.all_data_filenames_shuffled
        else:
            all_data_filenames = self.all_data_filenames
            
        sample_idx = start_index + self.local_rank    # index of the first sample by this worker
        for traj_idx, data_filename in enumerate(all_data_filenames):
            samples_start_idx, samples_end_idx = self.traj_idx_2_samples_range[traj_idx]
            if sample_idx < samples_start_idx or sample_idx > samples_end_idx:
                continue
            
            with open(data_filename, 'rb') as f:
                traj = pickle.load(f)
            if self.task and traj['instruction'] != self.task:
                continue

            ds_name = data_filename.split('/')[-1].split('-')[0]

            repeat_times = 1
            if self.is_train and ds_name in self.hard_tasks:
                # print(f"[{self.local_rank}] {data_filename}: {traj['instruction']}")
                if traj['instruction'] in self.hard_tasks[ds_name]:
                    repeat_times = 2
            
            traj = self.process_traj(traj, ds_name)
            prompt, obs_images, goal_images, future_images, actions = self.convert_traj_to_samples(traj)

            for i, (obs_img, goal_img, future_img, action) in enumerate(
                list(zip(obs_images, goal_images, future_images, actions)) * repeat_times
            ):
                if (samples_start_idx + i) != sample_idx:
                    continue
                yield ds_name, prompt, obs_img, goal_img, future_img, action

                sample_idx += self.world_size
                if sample_idx > max_index:
                    return

    def shuffle(self, seed):
        rng = np.random.default_rng(seed)
        self.all_data_filenames_shuffled = deepcopy(self.all_data_filenames)
        rng.shuffle(self.all_data_filenames_shuffled)
        self.num_all_transitions = self.cnt_transitions(self.all_data_filenames_shuffled)
        # print(f'{self.local_rank} {self.all_data_filenames_shuffled[0]} {seed}', flush=True)
            
    def cnt_transitions(self, list_data_filenames):
        transitions_cnt = 0
        for traj_idx, data_filename in enumerate(list_data_filenames):
            traj_name = data_filename.split('/')[-1]
            num_transitions = self.traj_name_2_num_trans[traj_name]
            if self.use_history:
                if self.is_train:
                    num_transitions = num_transitions // self.num_frames
                else:
                    num_transitions = num_transitions - self.num_frames + 1
                assert num_transitions > 0

            ds_name = traj_name.split('-')[0]
            if self.is_train and ds_name in self.hard_tasks:
                with open(data_filename, 'rb') as f:
                    traj = pickle.load(f)
                if traj['instruction'] in self.hard_tasks[ds_name]:
                    num_transitions *= 2
            
            self.traj_idx_2_samples_range[traj_idx] = (transitions_cnt, transitions_cnt+num_transitions-1)
            transitions_cnt += num_transitions
        return transitions_cnt

    def process_traj(self, traj, ds_name):
        """
        traj is a dict as
        {
            'robot_and_gripper': ['Franka', 'Franka_Default'],
            'instruction': "open the drawer",
            (Optional) 'task': "open_drawer",
            'obs_images': list_obs_image,
            'movement_actions': movement_actions,
            'gripper_actions': gripper_actions,
        }
        """
        assert len(traj['obs_images']) == len(traj['movement_actions']) + 1, 'Must drop the last action when preprocessing'
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
        if self.use_history:
            if self.is_train:
                num_samples = len(actions) // self.num_frames
                indices = np.random.randint(self.num_frames-1, len(actions), num_samples)
            else:
                indices = np.arange(self.num_frames-1, len(actions))
            history_obs_imgs = self._get_history(obs_images, indices)
            history_actions = self._get_history(actions, indices)
            return prompt, history_obs_imgs, goal_images[indices], future_images[indices], history_actions
        else:
            return prompt, obs_images, goal_images, future_images, actions
    
    def _process_actions(self, traj, ds_name):
        """ adding field 'actions' to traj """
        movement_actions = traj["movement_actions"]
        gripper_actions = traj["gripper_actions"]

        if self.args.normalize_type == 'normal':
            # normalize movement actions
            if ds_name in self.action_metadata:
                movement_actions = (
                    movement_actions - self.action_metadata[ds_name]["movement_action_mean"]
                ) / self.action_metadata[ds_name]["movement_action_std"]

            if len(gripper_actions.shape) == 1:
                # do not need to normalize gripper_actions that are already binarized,
                # just expand it for concatenation
                gripper_actions = np.expand_dims(gripper_actions, axis=1)

        elif self.args.normalize_type == 'bounds_q99':
            # normalize movement actions
            if ds_name in self.action_metadata:
                low = self.action_metadata[ds_name]["movement_action_q01"]
                high = self.action_metadata[ds_name]["movement_action_q99"]
                movement_actions = np.clip(
                    (movement_actions - low) / (high - low + 1e-8) * 2 - 1, -1, 1
                )

            if len(gripper_actions.shape) == 1:
                # do not need to normalize gripper_actions that are already binarized,
                # just expand it for concatenation
                gripper_actions = np.expand_dims(gripper_actions, axis=1)

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
        instruct = traj["instruction"]
        if self.is_train and self.args.paraphrase:
            if 'task' in traj and traj['task'] in self.task_2_instructions:
                instruct = np.random.choice(self.task_2_instructions[traj['task']])

        info = {
            "robot_type": traj["robot_and_gripper"][0],
            "gripper_type": traj["robot_and_gripper"][1],
            "instruct": clean_instruction(instruct),
            "img": self.args.img_placeholder,
            "act": self.args.act_placeholder,
        }
        traj["prompt"] = self.prompt_template.format(**info)
        return traj
    
    def _get_history(self, items, indices):
        history = []
        for i in indices:
            start = i + 1 - self.num_frames
            chunk = items[start:i+1]
            history.append(chunk)
        return history
    
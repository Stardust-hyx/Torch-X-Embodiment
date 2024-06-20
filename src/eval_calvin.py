import os

import json, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from PIL import Image
from agents import resnetv1_configs, GCBCAgent, RT1Agent, EmuAgent
from utils.misc import clean_instruction
from absl import app, flags
from collections import defaultdict
from moviepy.editor import ImageSequenceClip

import hydra
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env
from tqdm import tqdm

FLAGS = flags.FLAGS
existed_flags = FLAGS.__dir__()
flags.DEFINE_string("emu_ckpt", None, "Dir of original emu checkpoint")
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("config_path", None, "Path to the config of agent", required=True)
flags.DEFINE_string("benchmark_dir", None, "Directory of benchmark data", required=True)
flags.DEFINE_string("action_meta_path", None, "Path to action_meta.json", required=True)
flags.DEFINE_string("calvin_conf_dir", "../calvin/calvin_models/conf", "Directory of calvin conf")
flags.DEFINE_string("lang_ann_folder", "lang_annotations", "Folder of language ann")
flags.DEFINE_string("tgt_tasks", None, "target tasks")
flags.DEFINE_integer("max_time_step", 100, "Max length of an episode")
flags.DEFINE_integer("start_task_idx", None, "Skip tasks that were already evaluated")
flags.DEFINE_bool("use_goal_image", False, "Whether using goal image or not")
flags.DEFINE_bool("gen_goal_image", False, "Whether using goal image or not")
flags.DEFINE_bool("tqdm", False, "Whether using tqdm or not")

def save_np_img(img, fpath):
    img = Image.fromarray(img)
    img.save(fpath)
    return

def to_tensor_and_resize(img, im_size=224):  # to im_size
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = torch.from_numpy(img).permute((2, 0, 1))
    img = v2.functional.resize(
        img,
        (im_size, im_size),
        interpolation=v2.InterpolationMode.BICUBIC,
        antialias=True
    )
    return img

def normalize_action(action, mean, std):
    return (action - mean) / std

def unnormalize_action(action, mean, std):
    return action * std + mean

def load_checkpoint(checkpoint_path, config, action_meta_path):
    # create agent
    if config.method == 'gc_bc':
        encoder = resnetv1_configs[config.encoder](**config.gcbc_encoder_kwargs)
        agent = GCBCAgent(encoder, config, action_dim=config.action_dim)
    elif config.method == 'rt1':
        agent = RT1Agent(config.text_enc, config)
        if config.dtype == 'bf16':
            agent = agent.bfloat16()
    elif config.method == 'emu':
        agent = EmuAgent.from_pretrained(config.emu_ckpt, config, action_dim=config.action_dim).bfloat16()
        print(agent.emu_encoder.visual)
        print(agent.emu_encoder.cformer)
        print(agent.emu_encoder.decoder)
        print()
        for n, p in agent.named_parameters():
            if p.requires_grad:
                print(n)
        print()

    # hydrate agent with parameters from checkpoint
    print('Loading agent checkpoint ...')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agent.load_state_dict(state_dict["module"], strict=False)

    agent.cuda()
    agent.eval()
    print('Agent checkpoint is loaded.', flush=True)

    # load action meta
    action_meta = json.load(open(action_meta_path))
    action_mean = action_meta['movement_action_mean']
    action_std = action_meta['movement_action_std']

    return agent, action_mean, action_std

def collect_traj(in_dir, interval):
    start, end = interval
    list_robot_obs, list_scene_obs, list_obs_image, gold_actions = [], [], [], []
    for index in range(start, end+1):
        fname = "episode_%07d.npz" % index
        fpath = os.path.join(in_dir, fname)
        d = np.load(fpath)

        list_robot_obs.append(d['robot_obs'])
        list_scene_obs.append(d['scene_obs'])
        list_obs_image.append(d['rgb_static'])
        action = d['rel_actions']
        gold_actions.append(action)

    traj = {
        'robot_obs': np.stack(list_robot_obs),
        'scene_obs': np.stack(list_scene_obs),
        'images': np.stack(list_obs_image),
        'gold_actions': np.stack(gold_actions),
    }
    return traj

def rollout(env, task_oracle, traj, i, agent, task, instruction, prompt, config, action_mean, action_std):
    obs = env.reset(robot_obs=traj["robot_obs"][0], scene_obs=traj["scene_obs"][0])
    start_info = env.get_info()

    # print((obs['rgb_obs']['rgb_static'] - traj['images'][0]).sum())
    o = obs['rgb_obs']['rgb_static']

    image_goal = None
    if FLAGS.use_goal_image:
        image_goal = traj['images'][-2]
        if i < 2:
            save_np_img(image_goal, fpath=f"goal/goal_{task}_{i}.jpg")
        image_goal = to_tensor_and_resize(image_goal)
        

    if i < 2 and config.method == 'emu':
        image_1 = to_tensor_and_resize(o.copy())
        # generate goal image
        image = agent.generate_img(
            [
                "A robot observes the scene: ",
                image_1,
                f"To {instruction}, as shown in ",
            ],
            guidance_scale=2.,
        )
        image.save(f"examples/{task}_{i}.jpg")

        if image_goal is not None:
            image_2 = image_goal
            # generate near-future image
            image = agent.generate_img(
                [
                    "A robot observes the scene: ",
                    image_1,
                    f"To {instruction}, as shown in ",
                    image_2,
                    f"the robot arm moves towards {config.act_placeholder} and the scene becomes: ",
                ],
                guidance_scale=2.,
            )
            image.save(f"examples/_{task}_{i}.jpg")

    ims = [o]
    pre_context_image_tokens = None
    feature_for_gen_goal_img_ = None
    success = False
    for t in range(FLAGS.max_time_step):
        obs_img = to_tensor_and_resize(o.copy())
        if isinstance(agent, GCBCAgent):
            action = agent.sample_actions(obs_img, image_goal, argmax=True)
        elif isinstance(agent, RT1Agent):
            action, pre_context_image_tokens = agent.sample_actions(
                instruction, obs_img, pre_context_image_tokens
            )
        elif isinstance(agent, EmuAgent):
            feature_for_gen_goal_img = None
            if FLAGS.gen_goal_image and t >= 10 and t % 2 != 0:
                feature_for_gen_goal_img = feature_for_gen_goal_img_
            action, feature_for_gen_goal_img, goal_img = agent.sample_actions(
                prompt, obs_img, feature_for_gen_goal_img, image_goal, argmax=True,
                do_gen_goal_img=(FLAGS.gen_goal_image and t==10), guidance_scale=1,
            )
            if FLAGS.gen_goal_image and t >= 10 and t % 2 == 0:
                feature_for_gen_goal_img_ = feature_for_gen_goal_img
            if goal_img:
                goal_img.save(f"gen_goal/gen_{task}_{i}.jpg")
        
        action = np.array(action.squeeze().cpu())
        if action_std is not None:
            move_action = unnormalize_action(action[:6], action_mean, action_std)
        else:
            move_action = action[:6]
        gripper_action = np.array([1]) if action[6] < 0 else np.array([-1])

        action = np.concatenate((move_action, gripper_action))

        obs, _, _, current_info = env.step(action)
        o = obs['rgb_obs']['rgb_static']
        ims.append(o)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            success = True
            break
    
    if i < 2:
        cl = ImageSequenceClip(ims, fps=30)
        cl.write_gif(f'gif/{task}_{i}.gif', fps=30, logger=None)
    return 1 if success else 0

def eval_episodes(
        in_dir, env, task_oracle, agent, task, instructions, intervals, config, action_mean, action_std
    ):
    success_counts = []
    iter_ = zip(instructions, intervals)
    if FLAGS.tqdm:
        iter_ = tqdm(iter_, total=len(instructions))
    
    for i, (instruction, interval) in enumerate(iter_):
        traj = collect_traj(in_dir, interval)
        prompt = None
        if config.method == 'emu':
            info = {
                "robot_type": "Franka",
                "instruct": clean_instruction(instruction),
                "img": config.img_placeholder,
                "act": config.act_placeholder,
            }
            prompt = config.all_prompts["emu_wo_future"].format(**info)
            # print(prompt, flush=True)

        success_counts.append(
            rollout(env, task_oracle, traj, i, agent, task, instruction, prompt, config, action_mean, action_std)
        )
    return success_counts


def main(_):
    for k in FLAGS.__dir__():
        if k not in existed_flags:
            print(k, FLAGS.__getattr__(k), flush=True)
    
    os.makedirs('goal', exist_ok=True)
    os.makedirs('examples', exist_ok=True)
    os.makedirs('gif', exist_ok=True)
    os.makedirs('gen_goal', exist_ok=True)
    
    config = json.load(open(FLAGS.config_path))
    config['emu_ckpt'] = FLAGS.emu_ckpt
    if 'local_rank' not in config:
        config['local_rank'] = 0
    config = argparse.Namespace(**config)
    print(config, flush=True)
    agent, action_mean, action_std = load_checkpoint(
        FLAGS.checkpoint_path, config, FLAGS.action_meta_path
    )

    in_dir = FLAGS.benchmark_dir
    env = get_env(in_dir, show_gui=False)

    calvin_conf_dir = FLAGS.calvin_conf_dir
    task_cfg = OmegaConf.load(os.path.join(calvin_conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)

    tgt_tasks = eval(FLAGS.tgt_tasks) if FLAGS.tgt_tasks else None

    lang_ann_path = os.path.join(in_dir, FLAGS.lang_ann_folder, 'auto_lang_ann.npy')
    lang_ann = np.load(lang_ann_path, allow_pickle=True)
    lang_ann = lang_ann.item()
    instructions = lang_ann['language']['ann']
    tasks = lang_ann['language']['task']
    intervals = lang_ann['info']['indx']

    task_2_instructions = defaultdict(list)
    task_2_intervals = defaultdict(list)
    for task, instruction, interval in zip(tasks, instructions, intervals):
        task_2_instructions[task].append(instruction)
        task_2_intervals[task].append(interval)
    # for task, instructions in task_2_instructions.items():
    #     print(f"# Episodes ({task}): {len(instructions)}")
    # print()

    task_2_success_counts = dict()
    for i, task in enumerate(task_2_instructions):
        if FLAGS.start_task_idx and i < FLAGS.start_task_idx:
            continue
        if tgt_tasks and task not in tgt_tasks:
            continue

        instructions = task_2_instructions[task]
        intervals = task_2_intervals[task]

        counts = eval_episodes(
            in_dir, env, task_oracle, agent, task, instructions, intervals, config, action_mean, action_std, 
        )
        task_2_success_counts[task] = counts
        print(f'{task}: {sum(counts)/len(counts)*100.0:.2f}% success ({sum(counts)}/{len(counts)})', flush=True)
        
    success_counts = sum(task_2_success_counts.values(), [])
    overall_success_rate = sum(success_counts) / len(success_counts) * 100.0
    print(f'Overall Success Rate: {overall_success_rate:.2f}% ({sum(success_counts)}/{len(success_counts)})')


if __name__ == "__main__":
    app.run(main)

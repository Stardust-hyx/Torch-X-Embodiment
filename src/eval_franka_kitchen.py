import os

import json, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from agents.resnet import resnetv1_configs
from agents.gc_bc import GCBCAgent
from absl import app, flags
from collections import defaultdict

from r3meval.utils.gym_env import GymEnv
from r3meval.utils.obs_wrappers import MuJoCoPixelObs
from r3meval.utils import tensor_utils
import mj_envs, gym
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("config_path", None, "Path to the config of agent", required=True)
flags.DEFINE_string("benchmark_dir", None, "Directory of benchmark data", required=True)
flags.DEFINE_string("action_meta_path", None, "Path to action_meta.json", required=True)
flags.DEFINE_string("cameras", "left_cap2,right_cap2", "Names of cameras, split by comma")
flags.DEFINE_integer("max_time_step", 100, "Max length of an episode")
flags.DEFINE_bool("use_goal_image", False, "Whether using goal image or not")

INSTRUCTION_MAP = {
    "kitchen_knob1_on-v3": "turn the stove top kkob",
    "kitchen_ldoor_open-v3": "open the left door",
    "kitchen_light_on-v3": "turn on the light",
    "kitchen_micro_open-v3": "open the microwave",
    "kitchen_sdoor_open-v3": "slide the right door open",
}

def env_constructor(env_name, image_width=224, image_height=224,
                    camera_name=None, pixel_based=True,
                    render_gpu_id=0):

    ## If pixel based will wrap in a pixel observation wrapper
    if pixel_based:
        e = gym.make(env_name)
        ## Wrap in pixel observation wrapper
        e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                           camera_name=camera_name, device_id=render_gpu_id)
        # ## Wrapper which encodes state in pretrained model
        # e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
        #                 proprio=proprio, camera_name=camera_name, env_name=env_name)
        e = GymEnv(e)
    else:
        print("Only supports pixel based")
        assert(False)
    return e

def to_tensor_and_resize(img, im_size=224):  # to im_size
    img = torch.from_numpy(img).permute((2, 0, 1))
    img = transforms.functional.resize(
        img,
        (im_size, im_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
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

    # hydrate agent with parameters from checkpoint
    print('Loading agent checkpoint ...')
    state_dict = torch.load(checkpoint_path)
    agent.load_state_dict(state_dict["module"])
    agent.cuda()
    agent.eval()
    print('Agent checkpoint is loaded.')

    # load action meta
    action_meta = json.load(open(action_meta_path))
    action_mean = np.concatenate((action_meta['movement_action_mean'], action_meta['gripper_action_mean']))
    action_std = np.concatenate((action_meta['movement_action_std'], action_meta['gripper_action_std']))

    return agent, action_mean, action_std


def eval_episodes(agent, trajs, instruction, camera, env_name="kitchen_sdoor_open-v3", action_mean=None, action_std=None):
    env = env_constructor(env_name=env_name, camera_name=camera)
    rollouts = []
    for i, traj in enumerate(tqdm.tqdm(trajs)):
        image_goal = None
        if FLAGS.use_goal_image:
            image_goal = traj['images'][-1]
            image = Image.fromarray(image_goal)
            image.save(f"{i}.png")
            image_goal = to_tensor_and_resize(image_goal)

        observations=[]
        actions=[]
        rewards=[]
        # agent_infos = []
        env_infos = []
        ims = []

        t = 0
        done = False
        o = env.reset()
        env.set_env_state(traj['init_state_dict'])
        init_state = env.get_env_state()
        # image_obs = env.env.get_image()
        ims.append(o)

        while t < FLAGS.max_time_step and done != True:
            obs_img = to_tensor_and_resize(o.copy())
            action = np.array(
                agent.sample_actions(obs_img, image_goal, argmax=True).squeeze().cpu()
            )
            if action_std is not None:
                action = unnormalize_action(action, action_mean, action_std)
            next_o, r, done, env_info_step = env.step(action)
            env_infos.append(env_info_step)
            t += 1
            o = next_o
            ims.append(o)

        rollout = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            # agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done, 
            init_state=init_state,
            images=ims
        )
        rollouts.append(rollout)

        if i < 5:
            from moviepy.editor import ImageSequenceClip
            cl = ImageSequenceClip(ims, fps=20)
            cl.write_gif(f'vid_{i}.gif', fps=20)

    success_rate = env.env.unwrapped.evaluate_success(rollouts)
    print(f'{instruction} {camera}: {success_rate} success')
    # exit(0)
    return success_rate


def main(_):
    config = json.load(open(FLAGS.config_path))
    config = argparse.Namespace(**config)
    agent, action_mean, action_std = load_checkpoint(
        FLAGS.checkpoint_path, config, FLAGS.action_meta_path
    )

    task_camera_2_success_rate = dict()

    cameras = FLAGS.cameras.split(',')
    for camera in cameras:
        in_dir = os.path.join(FLAGS.benchmark_dir, camera)
        
        for fn in os.listdir(in_dir):
            prefix, suffix = fn.split('.')
            if suffix != 'pickle':
                continue

            fpath = os.path.join(in_dir, fn)
            trajs = pickle.load(open(fpath, 'rb'))
            instruction = INSTRUCTION_MAP[prefix]
            print(f'{fpath}: {len(trajs)} trajs')

            # The first 25 demos are used for training,
            # while the 50~150 demos are used for evaluation.
            success_rate = eval_episodes(agent, trajs[50:150], instruction, camera,
                                         action_mean=action_mean, action_std=action_std)
            
            task_camera_2_success_rate[(instruction, camera)] = success_rate

    success_rates = list(task_camera_2_success_rate.values())
    overall_success_rate = sum(success_rates) / len(success_rates)
    print(f'Overall Success Rate: {overall_success_rate}')


if __name__ == "__main__":
    app.run(main)

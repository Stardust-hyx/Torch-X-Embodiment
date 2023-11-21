import os
import json, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from PIL import Image
from agents.resnet import resnetv1_configs
from agents.gc_bc import GCBCAgent
from absl import app, flags
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("config_path", None, "Path to the config of agent", required=True)
flags.DEFINE_string("traj_path", None, "Path to a trajectory", required=True)
flags.DEFINE_string("action_meta_path", None, "Path to action_meta.json", required=True)
flags.DEFINE_string("goal_image_path", None, "Path to a single goal image")


def squash(path, im_size=224):  # from 480x640 to im_size
    im = Image.open(path)
    im = im.resize((im_size, im_size), Image.Resampling.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out

def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]

def normalize_action(action, mean, std):
    return (action - mean) / std

def unnormalize_action(action, mean, std):
    return action * std + mean

def load_checkpoint(checkpoint_path, config, action_meta_path):

    # create agent
    if config.method == 'gc_bc':
        encoder = resnetv1_configs[config.encoder](**config.gcbc_encoder_kwargs)
        agent = GCBCAgent(encoder, config)

    # hydrate agent with parameters from checkpoint
    state_dict = torch.load(checkpoint_path)
    agent.load_state_dict(state_dict["module"])
    agent.cuda()
    agent.eval()

    # load action meta
    action_meta = json.load(open(action_meta_path))
    movement_action_mean = action_meta['movement_action_mean']
    movement_action_std = action_meta['movement_action_std']

    return agent, movement_action_mean, movement_action_std


def main(_):

    config = json.load(open(FLAGS.config_path))
    config = argparse.Namespace(**config)
    agent, move_action_mean, move_action_std = load_checkpoint(
        FLAGS.checkpoint_path, config, FLAGS.action_meta_path
    )

    traj = pickle.load(open(FLAGS.traj_path, 'rb'))
    instruction = traj['instruction']
    images = traj['obs_images']
    gold_movement_actions = traj['movement_actions']
    gold_gripper_actions = traj['gripper_actions']
    len_traj = len(gold_movement_actions)
    traj_name = FLAGS.traj_path.split('/')[-1].split('.')[0]
    print(instruction)

    if FLAGS.goal_image_path:
        image_goal_path = FLAGS.goal_image_path
        image_goal = squash(image_goal_path).transpose(2, 0, 1)
    else:
        image_goal = images[-1]

    pred_actions = []
    gold_actions = []
    all_mse = 0
    for i, (image_obs, gold_movement_action, gold_gripper_action) in enumerate(
        zip(images[:-1], gold_movement_actions, gold_gripper_actions)
    ):
        obs = {"image": image_obs}
        goal_obs = {"image": image_goal}

        action = np.array(
            agent.sample_actions(obs, goal_obs, argmax=True).squeeze().cpu()
        )
        
        gold_movement_action = normalize_action(gold_movement_action, move_action_mean, move_action_std)
        gold_action = np.concatenate((gold_movement_action, np.expand_dims(gold_gripper_action, axis=0)))

        pred_actions.append(action)
        gold_actions.append(gold_action)
        mse = np.square(np.subtract(gold_action, action)).sum()
        all_mse += mse

        print(f'[{i}]')
        print(f'[Gold Action] {gold_movement_action} {gold_gripper_action}')
        print(f'[Pred Action] {action[:6]} {action[6]}')
        print()

    print('Avg MSE:', all_mse / len_traj)

    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)
    titles = [
        'world_vector_0', 'world_vector_1', 'world_vector_2',
        'rotation_delta_0', 'rotation_delta_1', 'rotation_delta_2',
        'gripper_closedness_action_0'
    ]

    for pred_action, gold_action in zip(pred_actions, gold_actions):
        for pred, gold, title in zip(pred_action, gold_action, titles):
            action_name_to_values_over_time[title].append(gold)
            predicted_action_name_to_values_over_time[title].append(pred)

    figure_layout = [
        ['image'] * len(titles),
        titles
    ]

    plt.rcParams.update({'font.size': 13})

    num_display_frame = min(len_traj, 10)
    images = images[::int(len_traj/num_display_frame)]
    images = images.permute((0, 2, 3, 1))
    images = torch.unbind(images)
    stacked = torch.cat(images, 1)

    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    for k in titles:
        axs[k].plot(action_name_to_values_over_time[k], label='ground truth')
        axs[k].plot(predicted_action_name_to_values_over_time[k], label='predicted action')
        axs[k].set_title(k)
        axs[k].set_xlabel('Time in one episode')

    axs['image'].set_title(f'Instruction: {instruction}')
    axs['image'].imshow(stacked.numpy())
    axs['image'].set_xlabel('Time in one episode (subsampled)')

    plt.legend()
    plt.show()
    plt.savefig(f'demo_{traj_name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)


if __name__ == "__main__":
    app.run(main)

import os
import json, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as v2
# try:
#     from torchvision.transforms import v2
# except:
#     import torchvision.transforms as v2
from agents import resnetv1_configs, GCBCAgent, RT1Agent, EmuAgent
from utils.misc import clean_instruction
from absl import app, flags
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string("emu_ckpt", None, "Dir of original emu checkpoint", required=True)
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("config_path", None, "Path to the config of agent", required=True)
flags.DEFINE_string("traj_path", None, "Path to a trajectory", required=True)
flags.DEFINE_string("action_meta_path", None, "Path to action_meta.json", required=True)
flags.DEFINE_string("goal_image_path", None, "Path to a single goal image")


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
        agent = GCBCAgent(encoder, config, action_dim=config.action_dim)
    elif config.method == 'rt1':
        agent = RT1Agent(config.text_enc, config)
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
    movement_action_mean = action_meta['movement_action_mean']
    movement_action_std = action_meta['movement_action_std']
    gripper_action_mean = action_meta['gripper_action_mean']
    gripper_action_std = action_meta['gripper_action_std']

    return agent, movement_action_mean, movement_action_std, gripper_action_mean, gripper_action_std


def main(_):

    config = json.load(open(FLAGS.config_path))
    config['emu_ckpt'] = FLAGS.emu_ckpt
    if 'local_rank' not in config:
        config['local_rank'] = 0
    config = argparse.Namespace(**config)
    print(config, flush=True)
    agent, move_act_mean, move_act_std, grip_act_mean, grip_act_std = load_checkpoint(
        FLAGS.checkpoint_path, config, FLAGS.action_meta_path
    )

    traj = pickle.load(open(FLAGS.traj_path, 'rb'))
    instruction = traj['instruction']
    images = traj['obs_images']
    gold_movement_actions = traj['movement_actions']
    gold_gripper_actions = traj['gripper_actions']
    move_action_dims = gold_movement_actions.shape[-1]
    len_traj = len(gold_movement_actions)
    traj_name = FLAGS.traj_path.split('/')[-1].split('.')[0]
    print(traj_name)
    print(instruction)
    
    if FLAGS.goal_image_path:
        image_goal_path = FLAGS.goal_image_path
        image_goal = to_tensor_and_resize(Image.open(image_goal_path))
    else:
        image_goal = images[-5]
    
    if config.method == 'emu':
        info = {
            "robot_type": "Franka",
            "instruct": clean_instruction(instruction),
            "img": config.img_placeholder,
            "act": config.act_placeholder,
        }
        prompt = config.all_prompts["emu_wo_future"].format(**info)
        print(prompt, flush=True)
        
        image_1 = to_tensor_and_resize(Image.open("examples/dog.png"))
        image_2 = to_tensor_and_resize(Image.open("examples/sunflower.png"))
        image = agent.generate_img(
            [
                "This is the first image: ",
                image_1,
                "This is the second image: ",
                image_2,
                "The animal in the first image surrounded with the plant in the second image: ",
            ],
            guidance_scale=7.5,
        )
        image.save("examples/dog_sunflower.jpg")
        
        image_1 = images[5]
        image_2 = image_goal
        # generate goal image
        image = agent.generate_img(
            [
                "A robot observes the scene: ",
                image_1,
                f"To {instruction}, as shown in ",
            ],
            guidance_scale=2.,
        )
        image.save(f"examples/goal_{traj_name}.jpg")
        # generate near-future image
        image = agent.generate_img(
            [
                "A robot observes the scene: ",
                image_1,
                f"To {instruction}, as shown in ",
                image_2,
                f"it takes action {config.act_placeholder} and the scene becomes: ",
            ],
            guidance_scale=2.,
        )
        image.save(f"examples/future_{traj_name}.jpg")

    pred_actions = []
    gold_actions = []
    all_mse = 0
    for i, (image_obs, gold_movement_action, gold_gripper_action) in enumerate(
        zip(images[:-1], gold_movement_actions, gold_gripper_actions)
    ):

        if isinstance(agent, GCBCAgent):
            action = agent.sample_actions(image_obs, image_goal, argmax=True)
        elif isinstance(agent, RT1Agent):
            action, pre_context_image_tokens = agent.sample_actions(instruction, image_obs, pre_context_image_tokens)
        elif isinstance(agent, EmuAgent):
            action = agent.sample_actions(prompt, image_obs, image_goal, argmax=True)
            
        action = np.array(action.squeeze().cpu())
        
        gold_movement_action_ = normalize_action(gold_movement_action, move_act_mean, move_act_std)
        if len(gold_gripper_action.shape) == 0:
            gold_gripper_action = np.expand_dims(gold_gripper_action, axis=0)
            gold_gripper_action_ = gold_gripper_action
        else:
            gold_gripper_action_ = normalize_action(gold_gripper_action, grip_act_mean, grip_act_std)
        gold_action_ = np.concatenate((gold_movement_action_, gold_gripper_action_))

        pred_actions.append(action)
        gold_actions.append(gold_action_)
        mse = np.square(np.subtract(gold_action_, action)).sum()
        all_mse += mse

        print(f'[{i}]')
        print(f'[Gold Action] {gold_movement_action_} {gold_gripper_action_}')
        print(f'[Pred Action] {action[:move_action_dims]} {action[move_action_dims:]}')

        print(f'[Unnormalized Gold Action] {gold_movement_action} {gold_gripper_action}')
        movement_action_ = unnormalize_action(action[:move_action_dims], move_act_mean, move_act_std)
        gripper_action_ = unnormalize_action(action[move_action_dims:], grip_act_mean, grip_act_std)
        print(f'[Unnormalized Pred Action] {movement_action_} {gripper_action_}')
        print(flush=True)

    print('Avg MSE:', all_mse / len_traj)

    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)
    if move_action_dims == 6:
        titles = [
            'world_vector_0', 'world_vector_1', 'world_vector_2',
            'rotation_delta_0', 'rotation_delta_1', 'rotation_delta_2',
            'gripper_closedness_action_0'
        ]
    elif move_action_dims == 7:
        titles = [f'joint_velocity_{i}' for i in range(7)] + [f'gripper_velocity_{i}' for i in range(2)]

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
    if move_action_dims == 6:
        fig.set_size_inches([45, 10])
    elif move_action_dims == 7:
        fig.set_size_inches([55, 10])

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
    plt.savefig(f'demo_{config.method}_{traj_name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)


if __name__ == "__main__":
    app.run(main)

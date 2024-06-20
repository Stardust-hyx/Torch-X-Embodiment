import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import json, pickle
import torch
from torchvision.transforms import v2
from PIL import Image
from agents import resnetv1_configs, GCBCAgent, RT1Agent, EmuAgent
from utils.misc import clean_instruction
from moviepy.editor import ImageSequenceClip

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 150
NUM_SEQUENCES = 400
CALVIN_ROOT = "../calvin"

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


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

def unnormalize_action(action, mean, std):
    return action * std + mean

class CustomModel(CalvinBaseModel):
    def __init__(self, args):
        config = json.load(open(args.config_path))
        config['emu_ckpt'] = args.emu_ckpt
        if 'local_rank' not in config:
            config['local_rank'] = 0
        config = argparse.Namespace(**config)
        self.config = config

        # create agent
        if config.method == 'rt1':
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
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
        agent.load_state_dict(state_dict["module"], strict=False)

        agent.cuda()
        agent.eval()
        self.agent = agent
        print('Agent checkpoint is loaded.', flush=True)

        # load action meta
        action_meta = json.load(open(args.action_meta_path))
        self.action_mean = action_meta['movement_action_mean']
        self.action_std = action_meta['movement_action_std']

    def reset(self):
        """
        This is called
        """
        self.t = 0
        self.pre_context_image_tokens = None

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            prompt: language prompt
        Returns:
            action: predicted action
        """
        o = obs['rgb_obs']['rgb_static']
        obs_img = to_tensor_and_resize(o.copy())
        
        goal_img = None
        if isinstance(self.agent, RT1Agent):
            action, self.pre_context_image_tokens = self.agent.sample_actions(
                goal, obs_img, self.pre_context_image_tokens
            )
        elif isinstance(self.agent, EmuAgent):
            info = {
                "robot_type": "Franka",
                "instruct": clean_instruction(goal),
                "img": self.config.img_placeholder,
                "act": self.config.act_placeholder,
            }
            prompt = self.config.all_prompts["emu_wo_future"].format(**info)
            feature_for_gen_goal_img = None
            if self.t >= 10 and self.t % 2 != 0:
                feature_for_gen_goal_img = self.feature_for_gen_goal_img

            action, feature_for_gen_goal_img, goal_img = self.agent.sample_actions(
                prompt, obs_img, feature_for_gen_goal_img, None, argmax=True,
                do_gen_goal_img=(self.t==10), guidance_scale=1,
            )

            if self.t >= 10 and self.t % 2 == 0:
                self.feature_for_gen_goal_img = feature_for_gen_goal_img
        
        action = np.array(action.squeeze().cpu())
        if self.action_std is not None:
            move_action = unnormalize_action(action[:6], self.action_mean, self.action_std)
        else:
            move_action = action[:6]
        gripper_action = np.array([1]) if action[6] < 0 else np.array([-1])

        action = np.concatenate((move_action, gripper_action))
        self.t += 1
        return action, goal_img


def evaluate_policy(model, env, eval_log_dir=None, debug=False, resume_path=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    os.makedirs(os.path.join(eval_log_dir, 'gif'), exist_ok=True)
    os.makedirs(os.path.join(eval_log_dir, 'gen_goal'), exist_ok=True)

    if os.path.exists(resume_path):
        eval_sequences, results = pickle.load(open(resume_path, 'rb'))
    else:
        eval_sequences = get_sequences(NUM_SEQUENCES)
        results = []

    start_i = len(results)

    eval_itor = eval_sequences
    if not debug:
        eval_itor = tqdm(eval_sequences, position=0, leave=True)

    for seq_i, (initial_state, eval_sequence) in enumerate(eval_itor):
        if seq_i < start_i:
            continue
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_log_dir, seq_i)
        results.append(result)
        if not debug:
            eval_itor.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

        if resume_path is not None:
            pickle.dump((eval_sequences, results), open(resume_path, 'wb'))

    print_and_save(results, eval_sequences, eval_log_dir)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_log_dir, seq_i):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        print(flush=True)
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_log_dir, subtask_i, seq_i)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_log_dir, subtask_i, seq_i):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        ims = [obs['rgb_obs']['rgb_static']]

    for step in range(EP_LEN):
        action, goal_img = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if goal_img:
            goal_img.save(f"{eval_log_dir}/gen_goal/gen-{seq_i}-{subtask_i}-{subtask}.jpg")
        if debug:
            ims.append(obs['rgb_obs']['rgb_static'])

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                cl = ImageSequenceClip(ims, fps=30)
                cl.write_gif(f'{eval_log_dir}/gif/{seq_i}-{subtask_i}-{subtask}-succ.gif', fps=30, logger=None)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        cl = ImageSequenceClip(ims, fps=30)
        cl.write_gif(f'{eval_log_dir}/gif/{seq_i}-{subtask_i}-{subtask}-fail.gif', fps=30, logger=None)
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--emu_ckpt", default=None, type=str, help="Dir of original emu checkpoint")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to the config of agent")
    parser.add_argument("--action_meta_path", default=None, type=str, help="Path to action_meta.json")

    parser.add_argument("--dataset_path", default=None, type=str, help="Path to the dataset root directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")
    parser.add_argument("--resume_path", default=None, type=str, help="Path of existing eval sequences and results")

    args = parser.parse_args()
    print(args)

    # evaluate a custom model
    model = CustomModel(args)
    env = make_env(args.dataset_path)
    evaluate_policy(model, env, eval_log_dir=args.eval_log_dir, debug=args.debug, resume_path=args.resume_path)


if __name__ == "__main__":
    main()

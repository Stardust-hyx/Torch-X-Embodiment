import os
import json
import time
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.utils import RepeatingLoader

from config import get_args, get_ds_config
from utils.misc import set_random_seed
from utils.criterion import action_criterion
from data import XEmbodDatasetTorch
from agents import resnetv1_configs, GCBCAgent, EmuAgent


def evaluate(args, engine: deepspeed.DeepSpeedEngine, eval_dataset, criterion):
    rank = args.local_rank
    engine.eval()
    
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    log_probs, mse, pi_actions, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for data in eval_dataloader:
            (prompts, obs_imgs, goal_imgs, feature_imgs, actions) = data

            outputs = engine(prompts, obs_imgs, goal_imgs, feature_imgs)
            actions = actions.to(engine.device)
            _, info = criterion(outputs, actions)

            log_probs += info['log_probs']
            mse += info['mse']
            pi_actions += info['pi_actions']
            cnt += 1

    scores = {
        "log_probs": log_probs/cnt,
        "mse": mse/cnt,
        "pi_actions": pi_actions/cnt
    }

    print(f"[Rank {rank}] {scores}", flush=True)

    return scores
    

def train(args, engine: deepspeed.DeepSpeedEngine, criterion, train_dataset, eval_dataset, start_step=0):
    rank = args.local_rank
    world_size = engine.world_size
    gradient_accumulation_steps = engine.gradient_accumulation_steps()
    micro_train_batch_size = engine.train_micro_batch_size_per_gpu()
    assert micro_train_batch_size == args.train_batch_size // gradient_accumulation_steps // world_size
    
    if args.steps:
        epochs = 1
    else:
        epochs = args.epochs
    
    epoch, micro_step, step = 0, 0, 0
    cur_time = time.time()
    avg_loss = 0.0
    best_score, best_step = -1e10, -1
    for epoch in range(epochs):
        # shuffle train dataset at the begining of each epoch
        train_dataset.shuffle(seed=args.random_seed+epoch)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=micro_train_batch_size,
            num_workers=args.num_workers,
        )
        # allow for infinite iteration
        if args.steps:
            dataloader = RepeatingLoader(dataloader)

        for data in dataloader:
            micro_step += 1
            if micro_step % gradient_accumulation_steps == 0:
                step += 1
            
            # skip previous steps to resume training
            # This implementation does not give perfect restoration,
            # due to the inner randomness of the model forward process
            if (
                step < start_step or
                step == start_step and micro_step % gradient_accumulation_steps == 0
            ):
                continue

            (prompts, obs_imgs, goal_imgs, feature_imgs, actions) = data

            outputs = engine(prompts, obs_imgs, goal_imgs, feature_imgs)
            actions = actions.to(engine.device)
            act_loss, info = criterion(outputs.pred_dist, actions)
            engine.backward(act_loss)
            engine.step()

            avg_loss += info['actor_loss']

            if micro_step % gradient_accumulation_steps != 0:
                continue

            # logging
            if rank == 0 and step % args.log_interval == 0:
                print(f'[Step {step}] loss: {avg_loss/args.log_interval/gradient_accumulation_steps}'+
                      f' speed: {(time.time()-cur_time)/args.log_interval}s/step', flush=True)
                cur_time = time.time()
                avg_loss = 0.0

            # evaluate
            if step % args.eval_interval == 0 and eval_dataset:
                if rank == 0:
                    print(f'[Step {step}] evaluating...', flush=True)
                scores = evaluate(args, engine, eval_dataset, criterion)
                engine.train()
                is_best = False
                if scores[args.main_metric] >= best_score:
                    best_score = scores[args.main_metric]
                    best_step = step
                    is_best = True

            # save checkpoint
            if step % args.save_interval == 0 and args.save_dir:
                if args.save_best:
                    assert args.save_interval % args.eval_interval == 0
                    assert eval_dataset
                    do_save = is_best
                else:
                    do_save = True

                if do_save:
                    client_state = {'step': step}
                    ckpt_id = step
                    engine.save_checkpoint(args.save_dir, ckpt_id, client_state=client_state)

            # finish all training steps
            if args.steps and step >= args.steps:
                break

        dist.barrier()

    # evaluate at last
    if step % args.eval_interval != 0 and eval_dataset:
        if rank == 0:
            print(f'[Step {step}] evaluating...', flush=True)
        scores = evaluate(args, engine, eval_dataset, criterion)
        engine.train()
        is_best = False
        if scores[args.main_metric] >= best_score:
            best_score = scores[args.main_metric]
            best_step = step
            is_best = True

    # save checkpoint at last
    if step % args.save_interval != 0 and args.save_dir:
        if args.save_best:
            assert eval_dataset
            do_save = is_best
        else:
            do_save = True

        if do_save:
            client_state = {'step': step}
            ckpt_id = step
            engine.save_checkpoint(args.save_dir, ckpt_id, client_state=client_state)

    # Finish training
    print(f'Achieve best {args.main_metric} {best_score} on evaluation set at step {best_step}.')



if __name__ == '__main__':
    args = get_args()
    ds_config = get_ds_config(args)
    set_random_seed(args.random_seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(args)
    is_master = args.local_rank == 0
    if is_master:
        config = {k:v for k, v in vars(args).items() if k != 'local_rank'}
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    deepspeed.init_distributed()
    rank = args.local_rank
    world_size = dist.get_world_size()

    # create agent
    if args.method == 'gc_bc':
        encoder = resnetv1_configs[args.encoder](**args.gcbc_encoder_kwargs)
        agent = GCBCAgent(encoder, args)
    elif args.method == 'emu':
        agent = EmuAgent.from_pretrained(args.ckpt_path, args)
    else:
        raise NotImplementedError

    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=agent,
        model_parameters=[p for p in agent.parameters() if p.requires_grad],
        config=ds_config
    )

    # get iterable dataset
    trainset = XEmbodDatasetTorch(args, args.local_rank, world_size, is_train=True, is_master=is_master)
    dist.barrier()
    # do not conduct sharding on evaluation dataset, fool every proccess as it is main proccess
    testset = XEmbodDatasetTorch(args, local_rank=0, world_size=1, is_train=False, is_master=is_master)
    dist.barrier()

    # create loss function
    criterion = action_criterion
    
    # load checkpoint if it exists,
    # to resume training
    start_step = 0
    if args.save_dir and args.ckpt_id:
        _, client_state = engine.load_checkpoint(args.save_dir, args.ckpt_id)
        start_step = client_state['step']

    if args.steps or args.epochs:
        # train and also evaluate periodically
        train(args, engine, criterion, trainset, testset, start_step)
    elif is_master:
        print('skip training and evaluate directly...')
        evaluate(args, engine, testset, criterion)

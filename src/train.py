import os
import shutil
import json
import time
import torch
import torch.distributed as dist
import deepspeed
import datetime
from collections import defaultdict

os.environ['TOKENIZERS_PARALLELISM'] = "false"

from config import get_args, get_ds_config
from utils.misc import set_random_seed, RepeatingLoader


def evaluate(args, engine: deepspeed.DeepSpeedEngine, eval_dataset, criterion):
    rank = args.local_rank
    engine.eval()
    torch.cuda.empty_cache()
    
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    scores = defaultdict(float)
    cnt = 0
    with torch.no_grad():
        for data in eval_dataloader:
            (ds_names, prompts, obs_imgs, goal_imgs, future_imgs, actions) = data
            if not(args.method=='gc_bc' or args.prompt_type in ['emu_default', 'emu_wo_future']):
                goal_imgs = None
            if not (args.prompt_type in ['emu_default', 'emu_wo_goal']):
                future_imgs = None

            outputs = engine(ds_names, prompts, obs_imgs, goal_imgs, future_imgs)
            actions = actions.to(engine.device)
            _, info = criterion(outputs[0], actions)
            try:
                txt_loss, img_reg_loss, img_gen_loss = outputs[1:]
            except:
                txt_loss, img_reg_loss, img_gen_loss = 0, 0, 0

            for k, v in info.items():
                scores[k] += v
            if args.text_loss_weight > 0 and args.check_loss_when_eval:
                scores["L_txt"] += txt_loss.item()
            if args.img_loss_weight > 0 and args.check_loss_when_eval:
                scores["L_img_reg"] += img_reg_loss.item()
                scores["L_img_gen"] += img_gen_loss.item()

            cnt += 1

    for k in scores:
        scores[k] /= cnt

    print(f"[Rank {rank}] {scores}", flush=True)
    torch.cuda.empty_cache()

    return scores
    

def train(args, engine: deepspeed.DeepSpeedEngine, criterion, train_dataset, eval_dataset,
          start_step=0, best_score=-1e10, best_step=-1, pre_save_step=-1):
    torch.cuda.empty_cache()
    rank = args.local_rank
    world_size = engine.world_size
    gradient_accumulation_steps = engine.gradient_accumulation_steps()
    micro_train_batch_size = engine.train_micro_batch_size_per_gpu()
    assert micro_train_batch_size == args.train_batch_size // gradient_accumulation_steps // world_size
    
    micro_step, step = 0, 0
    cur_time = time.time()
    action_loss = 0.0
    text_loss = 0.0
    image_reg_loss = 0.0
    image_gen_loss = 0.0

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=micro_train_batch_size,
        num_workers=args.num_workers,
    )
    # allow for infinite iteration
    dataloader = RepeatingLoader(dataloader, args.random_seed)

    dist.barrier()
    for data in dataloader:
        micro_step += 1
        if micro_step % gradient_accumulation_steps == 0:
            step += 1
        
        # skip previous steps to resume training
        # Perfect restoration is not guaranteed due to the randomness of model forward process
        if (
            step < start_step or
            step == start_step and micro_step % gradient_accumulation_steps == 0
        ):
            if rank == 0 and step % args.log_interval == 0 and micro_step % gradient_accumulation_steps == 0:
                print(f'[Step {step}] skip', flush=True)
            continue

        (ds_names, prompts, obs_imgs, goal_imgs, future_imgs, actions) = data
        # print(prompts, flush=True)
        if not (args.method=='gc_bc' or args.prompt_type in ['emu_default', 'emu_wo_future']):
            goal_imgs = None
        if not (args.prompt_type in ['emu_default', 'emu_wo_goal']):
            future_imgs = None
        outputs = engine(ds_names, prompts, obs_imgs, goal_imgs, future_imgs)
        actions = actions.to(engine.device)
        act_loss, info = criterion(outputs[0], actions)
        try:
            txt_loss, img_reg_loss, img_gen_loss = outputs[1:]
        except:
            txt_loss, img_reg_loss, img_gen_loss = 0, 0, 0
        if args.text_loss_weight > 0 and args.img_loss_weight > 0:
            loss = act_loss + args.text_loss_weight * txt_loss + args.img_loss_weight * (img_reg_loss + img_gen_loss)
        elif args.text_loss_weight > 0:
            loss = act_loss + args.text_loss_weight * txt_loss
        elif args.img_loss_weight > 0:
            loss = act_loss + args.img_loss_weight * (img_reg_loss + img_gen_loss)
        else:
            loss = act_loss
        engine.backward(loss)
        engine.step()

        action_loss += info['actor_loss']
        if args.text_loss_weight > 0:
            text_loss += txt_loss.item()
        if args.img_loss_weight > 0:
            image_reg_loss += img_reg_loss.item()
            image_gen_loss += img_gen_loss.item()

        if micro_step % gradient_accumulation_steps != 0:
            continue
        
        # freeze/unfreeze modules
        if args.slow_start_finetune:
            if args.pretrain_ckpt_path and step == (args.warmup_steps//2):
                engine.module.set_action_fc_require_grad(True)
            if args.pretrain_ckpt_path and step == args.warmup_steps:
                engine.module.set_feature_layers_require_grad(True)

        # logging
        if rank == 0 and step % args.log_interval == 0:
            print(f'Step {step}\t action loss: {action_loss/args.log_interval/gradient_accumulation_steps}')
            print(f'           \t text loss: {text_loss/args.log_interval/gradient_accumulation_steps}')
            print(f'           \t img reg loss: {image_reg_loss/args.log_interval/gradient_accumulation_steps}')
            print(f'           \t img gen loss: {image_gen_loss/args.log_interval/gradient_accumulation_steps}')
            print(f'           \t speed: {(time.time()-cur_time)/args.log_interval}s/step', flush=True)
            cur_time = time.time()
            action_loss = 0.0
            text_loss = 0.0
            image_reg_loss = 0.0
            image_gen_loss = 0.0

        # evaluate
        if step % args.eval_interval == 0 and eval_dataset and step >= args.start_save:
            if rank == 0:
                print(f'[Step {step}] evaluating...', flush=True)
            scores = evaluate(args, engine, eval_dataset, criterion)
            engine.train()
            is_best = False
            if scores[args.main_metric] >= best_score:
                best_score = scores[args.main_metric]
                best_step = step
                is_best = True
            dist.barrier()

        # save checkpoint
        if step % args.save_interval == 0 and args.save_dir and step >= args.start_save:
            if args.save_best:
                assert args.save_interval % args.eval_interval == 0
                assert eval_dataset
                do_save = is_best
            else:
                do_save = True

            if do_save:
                if args.save_best and pre_save_step > 0:
                    shutil.rmtree(os.path.join(args.save_dir, str(pre_save_step)), ignore_errors=True)
                dist.barrier()
                ckpt_id = step
                pre_save_step = step
                client_state = {'step': step, 'best_score': best_score, 'best_step': best_step, 'pre_save_step': pre_save_step}
                engine.save_checkpoint(args.save_dir, ckpt_id, client_state, exclude_frozen_parameters=True)
            dist.barrier()

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
            if args.save_best and pre_save_step > 0:
                shutil.rmtree(os.path.join(args.save_dir, str(pre_save_step)), ignore_errors=True)
            dist.barrier()
            ckpt_id = step
            pre_save_step = step
            client_state = {'step': step, 'best_score': best_score, 'best_step': best_step, 'pre_save_step': pre_save_step}
            engine.save_checkpoint(args.save_dir, ckpt_id, client_state, exclude_frozen_parameters=True)

    # Finish training
    if rank == 0:
        print(f'Achieve best {args.main_metric} {best_score} on evaluation set at step {best_step}.')



if __name__ == '__main__':
    args = get_args()
    ds_config = get_ds_config(args)
    set_random_seed(args.random_seed, deterministic=(args.benchmarks is not None))
    os.makedirs(args.save_dir, exist_ok=True)

    from utils.criterion import action_criterion
    from data import XEmbodDatasetTorch
    from agents import resnetv1_configs, GCBCAgent, RT1Agent, EmuAgent

    is_master = args.local_rank == 0
    if is_master:
        print(args)
        config = {k:v for k, v in vars(args).items() if k != 'local_rank'}
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=100000))
    rank = args.local_rank
    world_size = dist.get_world_size()

    # create agent
    if args.method == 'gc_bc':
        encoder = resnetv1_configs[args.encoder](**args.gcbc_encoder_kwargs)
        agent = GCBCAgent(encoder, args)
    elif args.method == 'rt1':
        agent = RT1Agent(args.text_enc, args, time_sequence_length=args.num_frames).bfloat16()
    elif args.method == 'emu':
        agent = EmuAgent.from_pretrained(args.emu_ckpt, args).bfloat16()
        if args.local_rank == 0:
            print(agent.emu_encoder.visual)
            print(agent.emu_encoder.cformer)
            print(agent.emu_encoder.decoder)
            print()
            for n, p in agent.named_parameters():
                if p.requires_grad:
                    print(n)
            print()
    else:
        raise NotImplementedError
    
    if args.benchmarks:
        # To finetune for benchmarking,
        # First load pretrained model checkpoint (optional)
        if args.pretrain_ckpt_path:
            pretrain_ckpt = torch.load(args.pretrain_ckpt_path, map_location=torch.device('cpu'))
            agent.load_state_dict(pretrain_ckpt["module"], strict=False)
            if args.slow_start_finetune:
                agent.set_feature_layers_require_grad(False)
                agent.set_action_fc_require_grad(False)
        # Then renew the action output layer
        agent.renew_action_linear(args.action_dim)

    grouped_params = None
    if args.method == 'emu':
        grouped_params = [
            {
                'params': [p for n, p in agent.named_parameters() if 'unet' in n],
                'weight_decay': 0,
            },
            {
                'params': [p for n, p in agent.named_parameters() if 'visual' in n],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in agent.named_parameters() if 'cformer' in n],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in agent.named_parameters() if 'lm' in n and 'stu_regress_head' not in n],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in agent.named_parameters() if 'stu_regress_head' in n],
                'weight_decay': 0,
            },
            {
                'params': [p for n, p in agent.named_parameters() if 'action' in n],
                'weight_decay': args.weight_decay,
            },
        ]
        #            unet, visual, cformer, lm, stu_regress_head, action
        lr_scales = [1,    0.25,   0.5,     0.75,  1,                1]
        ds_config['scheduler']['params']['warmup_min_lr'] = [args.min_lr * scale for scale in lr_scales]
        ds_config['scheduler']['params']['warmup_max_lr'] = [args.max_lr * scale for scale in lr_scales]

    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=agent,
        config=ds_config,
        model_parameters=grouped_params,
    )

    # get iterable dataset
    trainset = XEmbodDatasetTorch(args, args.local_rank, world_size, is_train=True, is_master=is_master)
    dist.barrier()
    # do not conduct sharding on evaluation dataset, fool every proccess as it is main proccess
    testset = XEmbodDatasetTorch(args, local_rank=0, world_size=1, is_train=False, is_master=is_master)
    dist.barrier()

    # create loss function
    criterion = action_criterion
    
    start_step, best_score, best_step, pre_save_step = 0, -1e10, -1, -1
    # load checkpoint if it exists,
    # to resume training
    if args.save_dir and args.ckpt_id:
        _, client_state = engine.load_checkpoint(args.save_dir, args.ckpt_id, load_module_strict=False)
        start_step = client_state['step']
        best_score = client_state['best_score']
        best_step = client_state['best_step']
        pre_save_step = client_state['pre_save_step']

    if args.steps:
        # train and also evaluate periodically
        engine.train()
        train(args, engine, criterion, trainset, testset, start_step, best_score, best_step, pre_save_step)
    elif is_master:
        print('skip training and evaluate directly...')
        evaluate(args, engine, testset, criterion)

import argparse

DATASETS = [
    "berkeley_cable_routing",
    "viola",
    "toto"
]

ALL_PROMPTS = {
    "emu_default": "You are a {robot_type} robot, observing that {img}. Your task is to {instruct}, as shown in {img}. You take action {act} and reach the state {img}",
}

AUGMENT_KWARGS = dict(
    random_resized_crop=dict(
        size=[224, 224],
        scale=[0.8, 1.0],
        ratio=[0.9, 1.1],
        antialias=True
    ),
    color_jitter=dict(
        brightness=0.2,
        contrast=[0.8, 1.2],
        saturation=[0.8, 1.2],
        hue=0.1,
    ),
    augment_order=[
        "random_resized_crop",
        "color_jitter",
    ],
)

GCBC_ENCODER_KWARGS = dict(
    pooling_method="avg",
    add_spatial_coordinates=True,
    act="SiLU",
    input_img_shape=[224, 224],
    input_channels=6
)

# same as src/agents/Emu/Emu-14B.json
EMU_KWARGS = {
    "embed_dim": 1024,
    "vision_cfg": {
        "image_size": 224,
        "layers": 40,
        "width": 1408,
        "head_width": 88,
        "mlp_ratio": 4.3637,
        "patch_size": 14,
        "eva_model_name": "eva-clip-g-14-x",
        "drop_path_rate": 0,
        "xattn": True,
        "freeze": False
    },
    "multimodal_cfg": {
        "name": "llama-13B",
        "xattn": True,
        "n_causal": 32,
        "freeze": False
    },
    "vladapter_cfg": {
        "name": "cformer",
        "n_causal": 32
    }
}

def get_args():
    def str2bool(v):
        return v.lower() in ('true')

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data_dir', type=str, default='/data1/hyx/np_datasets')
    data_arg.add_argument('--sample_weights', type=str, default=None)
    # data_arg.add_argument('--sample_weights', type=str, default='[0.25,0.35,0.4]')
    data_arg.add_argument('--num_workers', type=int, default=4)
    data_arg.add_argument('--relabel_actions', type=str2bool, default=True)
    data_arg.add_argument('--goal_relabeling_strategy', type=str, default='uniform')
    data_arg.add_argument('--goal_relabel_offset', type=int, default=1)
    data_arg.add_argument('--goal_relabel_future_step', type=int, default=4)
    data_arg.add_argument('--augment', type=str2bool, default=True)
    data_arg.add_argument('--prompt_type', type=str, default='emu_default')
    data_arg.add_argument('--img_placeholder', type=str, default='[<IMG_PLH>]')
    data_arg.add_argument('--act_placeholder', type=str, default='[<ACT_PLH>]')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--dtype', type=str, default='fp32')
    # for gcbc
    model_arg.add_argument('--encoder', type=str, default='resnetv1-34-bridge')
    # for emu
    model_arg.add_argument('--ckpt_path', type=str, default='/home/hyx/huggingface/Emu/pretrain', help="Emu ckpt path")
    model_arg.add_argument('--instruct', type=str2bool, default=False, help="Load Emu-I")

    learn_arg = parser.add_argument_group('Learning')
    learn_arg.add_argument('--save_dir', type=str, default='./save')
    learn_arg.add_argument('--ckpt_id', type=str, default=None)
    learn_arg.add_argument('--train_batch_size', type=int, default=256)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--eval_batch_size', type=int, default=256)
    learn_arg.add_argument('--max_lr', type=float, default=1e-4)
    learn_arg.add_argument('--min_lr', type=float, default=1e-5)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=5.0)
    learn_arg.add_argument('--epochs', type=int, default=None)
    learn_arg.add_argument('--steps', type=int, default=300000)
    learn_arg.add_argument('--warmup_steps', type=int, default=10000)
    learn_arg.add_argument('--decay_steps', type=int, default=300000)
    learn_arg.add_argument('--log_interval', type=int, default=2000)
    learn_arg.add_argument('--eval_interval', type=int, default=5000)
    learn_arg.add_argument('--save_interval', type=int, default=5000)
    learn_arg.add_argument('--save_best', type=str2bool, default=True)
    learn_arg.add_argument('--main_metric', type=str, default='log_probs')
    
    misc_arg = parser.add_argument_group('MISC')
    misc_arg.add_argument('--method', type=str, default='gc_bc')
    misc_arg.add_argument('--random_seed', type=int, default=42)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    args.datasets = DATASETS
    args.all_prompts = ALL_PROMPTS
    args.augment_kwargs = AUGMENT_KWARGS
    args.gcbc_encoder_kwargs = GCBC_ENCODER_KWARGS
    args.emu_kwargs = EMU_KWARGS

    return args


def get_ds_config(args):
    ds_config = {
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.log_interval,
        "optimizer": {
            "type": "Adam",
            "params": {
                "weight_decay": args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.steps if args.steps else args.warmup_steps + args.decay_steps,
                "warmup_min_lr": args.min_lr,
                "warmup_max_lr": args.max_lr,
                "warmup_num_steps": args.warmup_steps,
            }
        },
        "gradient_clipping": args.max_grad_norm,
        "bf16": {
            "enabled": args.dtype == "bf16"
        },
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15
        },
        # "zero_optimization": {
        #     "stage": 2,
        #     "contiguous_gradients": True,
        #     "overlap_comm": True,
        #     "reduce_scatter": True,
        # },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "stage3_gather_16bit_weights_on_model_save": True
        },
    }

    return ds_config
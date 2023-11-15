import argparse

DATASETS = [
    "berkeley_cable_routing",
    "viola",
    "toto"
]

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
    data_arg.add_argument('--goal_relabel_min_offset', type=int, default=1)
    data_arg.add_argument('--augment', type=str2bool, default=True)

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--dtype', type=str, default='fp32')
    model_arg.add_argument('--encoder', type=str, default='resnetv1-34-bridge')

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
    learn_arg.add_argument('--decay_steps', type=int, default=None)
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
    args.augment_kwargs = AUGMENT_KWARGS
    args.gcbc_encoder_kwargs = GCBC_ENCODER_KWARGS

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
        # }
    }

    return ds_config
import argparse

DATASETS = [
    "taco_play",
    "berkeley_cable_routing",
    "viola",
    "toto",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "task_ABCD",
    "task_ABCD_rerender"
]

ALL_PROMPTS = {
    "rt1_default": "{instruct}",
    "emu_default": "A robot observes the scene: {img}To {instruct}, as shown in {img}the robot arm moves towards {act} and the scene becomes: {img}",
    "emu_wo_goal": "A robot observes the scene: {img}To {instruct}, the robot arm moves towards {act} and the scene becomes: {img}",
    "emu_wo_future": "A robot observes the scene: {img}To {instruct}, as shown in {img}the robot arm moves towards {act}",
    "emu_wo_goal_future": "A robot observes the scene: {img}To {instruct}, the robot arm moves towards {act}",
}

HARD_TASKS = {
    "Franka_Kitchen_left_cap2": ["open the left door", "open the microwave"],
    "Franka_Kitchen_right_cap2": ["open the left door", "open the microwave", "turn the stove top knob"],
    "_Franka_Kitchen_left_cap2": [],
    "_Franka_Kitchen_right_cap2": [],
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
        "freeze": True
    },
    "multimodal_cfg": {
        "name": "llama-13B",
        "xattn": True,
        "n_causal": 32,
        "freeze": True
    },
    "vladapter_cfg": {
        "name": "cformer",
        "n_causal": 32
    },
    "unfreeze_vit_layers": [0, -1],
    "unfreeze_llm_layers": [0, -1],
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
    data_arg.add_argument('--avg_num_traj', type=int, default=3500)
    data_arg.add_argument('--num_workers', type=int, default=4)
    data_arg.add_argument('--normalize_type', type=str, default='normal', choices=['normal', 'bounds_q99'])
    data_arg.add_argument('--goal_relabeling_strategy', type=str, default='last_k_uniform')
    data_arg.add_argument('--goal_relabel_offset', type=int, default=8)
    data_arg.add_argument('--goal_relabel_future_step', type=int, default=2)
    data_arg.add_argument('--augment', type=str2bool, default=True)
    data_arg.add_argument('--paraphrase', type=str2bool, default=True)
    data_arg.add_argument('--task_2_instructs_dir', type=str, default='../calvin/calvin_models/conf/annotations')
    data_arg.add_argument('--same_action_mean_std', type=str2bool, default=False)
    data_arg.add_argument('--prompt_type', type=str, default='emu_default', choices=["rt1_default", "emu_default", "emu_wo_goal", "emu_wo_future", "emu_wo_goal_future"])
    data_arg.add_argument('--img_placeholder', type=str, default='[<IMG_PLH>]')
    data_arg.add_argument('--act_placeholder', type=str, default='[<ACT_PLH>]')
    data_arg.add_argument('--use_history', type=str2bool, default=False)
    data_arg.add_argument('--num_frames', type=int, default=6)
    data_arg.add_argument('--benchmarks', type=str, default=None, help='names of benchmark data folders (located at data_dir)')
    data_arg.add_argument('--task', type=str, default=None, help='task instruction')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--dtype', type=str, default='fp32')
    # for gcbc
    model_arg.add_argument('--encoder', type=str, default='resnetv1-34-bridge')
    # for rt1
    model_arg.add_argument('--text_enc', type=str, default="BAAI/bge-base-en-v1.5", help="rt1 text encoder path")
    # for emu
    model_arg.add_argument('--emu_ckpt', type=str, default="https://huggingface.co/BAAI/Emu/tree/main/pretrain", help="Emu ckpt path")
    model_arg.add_argument('--instruct', type=str2bool, default=False, help="Load Emu-I")
    model_arg.add_argument('--quantization', type=str, default=None, choices=["int8", "fp4", "nf4"])
    model_arg.add_argument('--double_quant', type=str2bool, default=False, help="Use double quant from QLoRA")
    model_arg.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    model_arg.add_argument('--lora', type=str2bool, default=True, help="Use LoRA")
    model_arg.add_argument('--text_loss_weight', type=float, default=0.0)
    model_arg.add_argument('--img_loss_weight', type=float, default=0.0)
    model_arg.add_argument('--cfg', type=str2bool, default=True, help="Use classifier-free guidance")
    model_arg.add_argument('--action_dim', type=int, default=7)

    learn_arg = parser.add_argument_group('Learning')
    learn_arg.add_argument('--save_dir', type=str, default='./save')
    learn_arg.add_argument('--ckpt_id', type=str, default=None)
    learn_arg.add_argument('--pretrain_ckpt_path', type=str, default=None)
    learn_arg.add_argument('--train_batch_size', type=int, default=256)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--eval_batch_size', type=int, default=256)
    learn_arg.add_argument('--max_lr', type=float, default=1e-4)
    learn_arg.add_argument('--min_lr', type=float, default=1e-5)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=5.0)
    learn_arg.add_argument('--steps', type=int, default=300000)
    learn_arg.add_argument('--warmup_steps', type=int, default=10000)
    learn_arg.add_argument('--warmup_type', type=str, default='log')
    learn_arg.add_argument('--log_interval', type=int, default=2000)
    learn_arg.add_argument('--eval_interval', type=int, default=5000)
    learn_arg.add_argument('--save_interval', type=int, default=5000)
    learn_arg.add_argument('--start_save', type=int, default=0)
    learn_arg.add_argument('--save_best', type=str2bool, default=True)
    learn_arg.add_argument('--main_metric', type=str, default='log_p')
    learn_arg.add_argument('--check_loss_when_eval', type=str2bool, default=True)
    learn_arg.add_argument('--slow_start_finetune', type=str2bool, default=False)
    
    misc_arg = parser.add_argument_group('MISC')
    misc_arg.add_argument('--method', type=str, default='gc_bc')
    misc_arg.add_argument('--random_seed', type=int, default=42)
    misc_arg.add_argument('--zero_stage', type=int, default=0)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    args.datasets = DATASETS
    args.hard_tasks = HARD_TASKS
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
                "total_num_steps": args.steps,
                "warmup_min_lr": args.min_lr,
                "warmup_max_lr": args.max_lr,
                "warmup_num_steps": args.warmup_steps,
                "warmup_type": args.warmup_type,
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
    }

    if args.zero_stage == 1:
        ds_config['zero_optimization'] = {
            "stage": 1,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True,
            # },
        }
    elif args.zero_stage == 2:
        ds_config['zero_optimization'] = {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        }
    elif args.zero_stage == 3:
        ds_config['zero_optimization'] = {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "ratio": 1.0
            },
            "overlap_comm": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
        ds_config["prescale_gradients"] = False
    return ds_config
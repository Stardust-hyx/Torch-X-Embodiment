# Pytorch Implementation for Open X-Embodiment

This repository provides preprocessing and training code based on Pytorch for [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment).

Since most open-source LLMs are implemented with Pytorch and require highly efficient distributed training frameworks (e.g. [Deepspeed](https://www.deepspeed.ai/getting-started/)) to fine-tune, we believe this repository can facilitate the application of Multimodal LLMs on this field.

## Schedule

We are activly implementing the following:

- [X] Open X-embodiment data processing
- [ ] Web image-text datasets processing
- [X] GCBC baseline
- [ ] RoboCat baseline
- [X] RT-1 baseline
- [ ] RT-2 baseline
- [X] Act-Emu w/o diffusion loss
- [X] Act-Emu
- [X] Demo on offline trajectories
- [X] Evaluation on CALVIN
- [ ] Real-word Evaluation (Franka Panda)

## Features

- Easily extending to more datasets
- Normalized movement actions
- Binarized gripper actions
- Data augmentation
- Efficient distributed training based on Deepspeed
- Support both language-conditioning and visual goal-conditioning

## Environment

The dependencies for this codebase can be installed in a conda environment:

```
conda create -n xembod python=3.8
conda activate xembod
pip install -r requirements.txt
```

(Optional) Install xformers following the [official instruction](https://github.com/facebookresearch/xformers).

## Download Data

The datasets in tf-record format can be downloaded with the following python script, where the dataset names can be found in [the dataset sheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0)

```
import tensorflow_datasets as tfds
import tqdm

# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DATASET_NAMES = [
  'fractal_20220817_data', 'kuka', 'bridge',
  'taco_play', 'jaco_play', 'berkeley_cable_routing',
  'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola',
  'berkeley_autolab_ur5', 'toto', 'language_table',
]
DOWNLOAD_DIR = '/data/tf_datasets'

for dataset_name in tqdm.tqdm(DATASET_NAMES):
  _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)
```

If you run into network issue, try download the datasets manually by running

```
gsutil -m cp -r gs://gresearch/robotics/{dataset_name} /data/tf_datasets
```

## Preprocess Data

Use the following commond to convert the tf-record data into trajectory-level pickle files. The processing procedure takes about 30min / 100GB.

```
cd ./preprocessing
python preprocess_and_save.py --in_dir /data/tf_datasets --out_dir /data/np_datasets
python get_num_transitions.py --data_dir /data/np_datasets
```

Metadata (mean and standard deviation) of actions for each dataset will be saved in `/data/np_datasets/{dataset_name}`. Example gif files will go into `preprocessing/gif`. Info of trajectories will go into `preprocessing/log`.

To specify the dataset names or modify the dataset-specific process functions, see `preprocessing\utils.py`.

### Calvin Data

Download [the Calvin dataset](https://github.com/mees/calvin/tree/main/dataset) (ABCD-D split). Then convert the data into trajectory-level pickle files:
```
cd ./preprocessing
python convert_calvin.py --in_dir ../calvin/dataset/task_ABCD_D --out_dir /data/np_datasets/task_ABCD
```

We also find it beneficial to rerender the calvin dataset:
```
python convert_rerender.py --in_dir ../calvin/dataset/task_ABCD_D --out_dir /data/np_datasets/task_ABCD_rerender
```

## Prepare Pretrained Models

- For RT-1, we utilize [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) as the language encoder. If your sever cannot access it, download and place it in a local dir (e.g. "../huggingface/bge-base-en-v1.5"), and specify it using flag "--text_enc [PATH-TO-BGE]".
- For Act-Emu, we utilize [Emu](https://huggingface.co/BAAI/Emu/tree/main/pretrain), download and place it in a local dir and specify it using flag "--emu_ckpt [PATH-TO-EMU]".

## Training

Start training by running

```
deepspeed src/train.py \
    --data_dir /data/np_datasets \
    --sample_weights balance \
    --num_workers 6 \
    --method gc_bc \
    --train_batch_size 288 \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 256 \
    --max_lr 1e-4 \
    --min_lr 5e-6 \
    --max_grad_norm 10 \
    --steps 150000 \
    --warmup_steps 10000 \
    --warmup_type linear \
    --start_save 100000 \
    --save_dir save_gcbc/gcbc_pretrain
```

See `src/config.py` for availabel hyperparameters. Model checkpoints will be saved in the `{save_dir}` along with a `config.json`.

To specify which datasets to use, modify the constant `DATASETS` in `src\config.py`. Currently, we have only experimented on datasets involving Franka robot.

## Offline Demo

The following script demonstrates how to load the model, run inference on an offline episode and compare the predicted and gold actions.

```
python src/demo.py \
    --checkpoint_path gcbc_save/150000/mp_rank_00_model_states.pt \
    --config_path gcbc_save/config.json \
    --traj_dir /data/np_datasets/viola/test/viola-test-0.pkl \
    --action_meta_path /data/np_datasets/viola/action_meta.json
```

The display image is expected to look like:
![](demo_viola-test-0.png)


## Calvin Evaluation

- By default, the Calvin train split is included into the pretraining data, so that the jointly pretrained model can be directly evalutated on the Calvin environment.
- To additionally finetune on the Calvin dataset after pretraining, run the following:

```
deepspeed --include localhost:0,1,2,3,4,5 --master_port 29950 src/train.py \
    --data_dir /data/np_datasets \
    --benchmarks task_ABCD,task_ABCD_rerender \
    --paraphrase True \
    --num_workers 10 \
    --method gc_bc \
    --train_batch_size 216 \
    --eval_batch_size 64 \
    --max_lr 8e-5 \
    --min_lr 1e-6 \
    --steps 20000 \
    --warmup_steps 8000 \
    --warmup_type linear \
    --log_interval 1000 \
    --eval_interval 500 \
    --save_interval 500 \
    --start_save 15000 \
    --save_dir save_gcbc/gcbc_pretrain_fineutine \
    --pretrain_ckpt_path save_gcbc/gcbc_pretrain/150000/mp_rank_00_model_states.pt
```
where {pretrain_ckpt_path} is the path to your pretrained agent.

- Follow the installation steps in [the CALVIN repo](https://github.com/mees/calvin) to install the CALVIN environment.
- Evaluate on the validation set. (For each traj within, we initialize the env into the first state of the traj, and use its annotated instruction):

```
CUDA_VISIBLE_DEVICES=0 python src/eval_calvin.py \
    --checkpoint_path {YOUR_SAVE_DIR}/{CHECKPOINT_ID}/mp_rank_00_model_states.pt \
    --config_path {YOUR_SAVE_DIR}/config.json \
    --benchmark_dir ../calvin/dataset/task_ABCD_D/validation \
    --action_meta_path /data/np_datasets/task_ABCD/action_meta.json \
    --max_time_step 150
```
Note that, set the "--use_goal_image" flag for GCBC; set the "--gen_goal_image" flag for Act-Emu.

- Overall Performance on the Validation Set:

|                            | Overall Success Rate |
| -------------------------  | -------------------- |
| GCBC (w/o pretrain)        | 55.2%                |
| GCBC (joint-train)         | 58.5%                |
| GCBC (pretrain-fintune)    | 62.3%                |
|                                                   |
| RT-1 (w/o pretrain)        | 64.1%                |
| RT-1 (joint-train)         |                      |
| RT-1 (pretrain-fintune)    |                      |
|                                                   |
| Act-Emu (w/o pretrain)     |                      |
| Act-Emu (joint-train)      | 63.6%                |
| Act-Emu (pretrain-fintune) |                      |

- Evaluate under the long-horizon setting. (Since the env will be randomly initialized, ground-truth viausl goal is inaccessible and it is infeasible to evlaute GCBC under this setting):

```
CUDA_VISIBLE_DEVICES=0 python src/eval_calvin_LH.py \
    --checkpoint_path {YOUR_SAVE_DIR}/{CHECKPOINT_ID}/mp_rank_00_model_states.pt \
    --config_path {YOUR_SAVE_DIR}/config.json \
    --benchmark_dir ../calvin/dataset/task_ABCD_D/validation \
    --action_meta_path /data/np_datasets/task_ABCD/action_meta.json \
    --max_time_step 150
```

- Long-horizon Multi-task Results (tasks completed in a row):

|                            |   1   |   2   |   3   |   4   |   5   |
| -------------------------- | ----- | ----- | ----- | ----- | ----- |
| RT-1 (w/o pretrain)        | 83.0% | 51.0% | 29.8% | 13.5% |  7.0% |
| RT-1 (joint-train)         |       |       |       |       |       |
| RT-1 (pretrain-fintune)    |       |       |       |       |       |
|                                                                    |
| Act-Emu (w/o pretrain)     |       |       |       |       |       |
| Act-Emu (joint-train)      | 80.1% | 38.9% | 20.2% | 10.3% |  6.1% |
| Act-Emu (pretrain-fintune) |       |       |       |       |       |

(The author is in shortage of GPU resource and the experiments are progressing slowly.)


## Provided Checkpoints

Model checkpoints are available [here](https://drive.google.com/drive/folders/1gH8U7cO0Lru--CqkF_n_GOsuKzuX71d3?usp=sharing).
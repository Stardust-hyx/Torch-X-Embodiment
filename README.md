# Pytorch Implementation for Open X-Embodiment

This repository provides preprocessing and training code based on Pytorch for [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment).

Since most open-source LLMs are implemented with Pytorch and require highly efficient distributed training frameworks (e.g. [Deepspeed](https://www.deepspeed.ai/getting-started/)) to fine-tune, we believe this repository can facilitate the application of Multimodal LLMs on this field.

## Schedule

We are activly implementing the following:

- [x] Open X-embodiment data processing
- [ ] Web image-text datasets processing
- [x] GCBC baseline
- [ ] RoboCat baseline
- [x] RT-1 baseline
- [ ] RT-2 baseline
- [x] Act-Emu w/o diffusion loss
- [ ] Act-Emu
- [x] Demo on offline trajectories (for validation)
- [ ] Evaluation on Language Table
- [x] Evaluation on Franka Kitchen
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
conda create -n xembod python=3.10
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
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset_name} /data/tf_datasets
```

## Preprocess Data

Use the following commond to convert the tf-record data into trajectory-level pickle files. The processing procedure takes about 30min / 100GB.

```
cd preprocessing
python preprocess_and_save.py --in_dir /data/tf_datasets --out_dir /data/np_datasets
```

Metadata (mean and standard deviation) of actions for each dataset will be saved in `/data/np_datasets/{dataset_name}`. Example gif files will go into `preprocessing/gif`. Info of trajectories will go into `preprocessing/log`.

To specify the dataset names or modify the dataset-specific process functions, see `preprocessing\utils.py`.

## Training

Start training by running

```
deepspeed src/train.py \
    --data_dir /data/np_datasets \
    --sample_weights balance \
    --num_workers 4 \
    --augment True \
    --method gc_bc \
    --steps 100000 \
    --warmup_steps 3000 \
    --save_dir gcbc_aug_save \
    --random_seed 42
```

See `src/config.py` for availabel hyperparameters. Model checkpoints will be saved in the `{save_dir}` along with a `config.json`.

To specify which datasets to use, modify the constant `DATASETS` in `src\config.py`. Currently, we have only experimented on datasets involving Franka default gripper.

## Demo

The following script demonstrates how to load the model, run inference on an offline episode and compare the predicted and gold actions.

```
python src/demo.py \
    --checkpoint_path gcbc_aug_save/100000/mp_rank_00_model_states.pt \
    --config_path gcbc_aug_save/config.json \
    --traj_dir /data/np_datasets/viola/test/viola-test-0.pkl \
    --action_meta_path /data/np_datasets/viola/action_meta.json
```

The display image is expected to look like:
![](demo_viola-test-0.png)

## Evaluation

### Franka Kitchen
First fine-tune on 25 demostrations per task by runing
```
deepspeed --include localhost:0,1 --master_port 29600 src/train.py \
    --data_dir /home/hyx/r3m-eval/data \
    --benchmarks Franka_Kitchen_left_cap2,Franka_Kitchen_right_cap2 \
    --action_dim 9 \
    --sample_weights balance \
    --num_workers 2 \
    --method gc_bc \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --steps 50000 \
    --save_dir gcbc_franka_kitchen_save
```

Then test for 100 episodes per task
```
CUDA_VISIBLE_DEVICES=0 python src/eval_franka_kitchen.py \
    --checkpoint_path {YOUR_SAVE_DIR}/{CHECKPOINT_ID}/mp_rank_00_model_states.pt \
    --config_path {YOUR_SAVE_DIR}/config.json \
    --benchmark_dir ../r3m-eval/data/final_paths_multiview_rb_200 \
    --action_meta_path ../r3m-eval/data/Franka_Kitchen_left_cap2/action_meta.json \
    --cameras left_cap2,right_cap2 \
    --max_time_step 200 \
    --use_goal_image True
```

## Provided Checkpoints

A visual goal-conditioned BC Checkpoint is available [here](https://drive.google.com/drive/folders/15hXCEUwCbbU3kt4dgTc_dmwrG9Vzy-J0?usp=drive_link).


# Pytorch Implementation for Open X-Embodiment

This repository provides preprocessing and training code based on Pytorch for [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment).

Since most open-source LLMs are implemented with Pytorch and require highly efficient distributed training frameworks (e.g. [Deepspeed](https://www.deepspeed.ai/getting-started/)) to fine-tune, we believe this repository can facilitate the application of Multimodal LLMs on this field.

We provide implementations for the following methods:

- Visual goal-conditioned BC
- RT-1 (TODO)
- A novel generative vision-language-action model (TODO)

## Features

- Easily extending to more datasets
- Normalized movement actions
- Binarized gripper actions
- Data augmentation
- Recommended practice of the latest Pytorch
- Efficient distributed training based on Deepspeed
- Support both language-conditioning and vision-conditioning

## Environment

The dependencies for this codebase can be installed in a conda environment:

```
conda create -n xembod python=3.10
conda activate xembod
pip install -r requirements.txt
```

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
    --steps 300000 \
    --warmup_steps 10000 \
    --save_dir gcbc_aug_save \
    --random_seed 42
```

See `src/config.py` for availabel hyperparameters. Model checkpoints will be saved in the `save_dir` along with a `config.json`.

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

TODO

## Provided Checkpoints

A visual goal-conditioned BC Checkpoint is available [here](https://drive.google.com/drive/folders/15hXCEUwCbbU3kt4dgTc_dmwrG9Vzy-J0?usp=drive_link).

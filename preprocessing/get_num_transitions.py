import os
import pickle
import json

# data_dir = "/vepfs-cnsh4137610c2f4c/algo/user8/hyx/np_datasets"
# ds_names = None

# data_dir = "/home/data/hyx/r3m_data"
# ds_names = ['_Franka_Kitchen_left_cap2', '_Franka_Kitchen_right_cap2']

data_dir = "/data/calvin/dataset"
ds_names = ['task_D']

if ds_names is None:
    ds_names = os.listdir(data_dir)

for ds_name in ds_names:
    print(ds_name)
    traj_name_2_num_transitions = dict()
    for split_name in ['train', 'test']:
        total_transitions = 0
        split_path = os.path.join(data_dir, ds_name, split_name)
        for fn in os.listdir(split_path):
            fpath = os.path.join(split_path, fn)
            with open(fpath, 'rb') as f:
                traj = pickle.load(f)
            num_transitions = len(traj['movement_actions'])
            traj_name_2_num_transitions[fn] = num_transitions
            total_transitions += num_transitions
        assert f"{ds_name}-{split_name}" not in traj_name_2_num_transitions
        traj_name_2_num_transitions[f"{ds_name}-{split_name}"] = total_transitions

    out_fpath = os.path.join(data_dir, ds_name, 'num_transitions.json')
    with open(out_fpath, 'w') as f:
        json.dump(traj_name_2_num_transitions, f, indent=2)

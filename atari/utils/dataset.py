import torch
from torch.utils.data import Dataset

import lmdb
import pickle
import numpy as np

import os.path
import random

class LMDBDataset(Dataset):
    def __init__(self, data_path):
        super(LMDBDataset, self).__init__()
        self.data_path = data_path
        # Open the LMDB file
        self.env = lmdb.open(data_path,
                             readonly=True,
                             lock=False,
                             meminit=False)

        with self.env.begin(write=False) as txn:
            self.demos_path = pickle.loads(txn.get(b'__demo_path__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            self.length = len(self.keys)

        self.env = None
        self.demo_env = None

        self.shuffled = [i for i in range(self.length)]

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.data_path,
                                 readonly=True,
                                 lock=False,
                                 meminit=False)
            self.demo_env = lmdb.open(self.demos_path,
                                      readonly=True,
                                      lock=False,
                                      meminit=False)

        index = self.shuffled[idx]
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[index]))

        label = sample['label']
        with self.demo_env.begin(write=False) as txn:
            traj_a = pickle.loads(txn.get(sample['trajectories'][0][0]))
            traj_b = pickle.loads(txn.get(sample['trajectories'][1][0]))

        traj_a_states  = traj_a['states'][sample['trajectories'][0][1]]
        traj_a_actions = traj_a['actions'][sample['trajectories'][0][1]]
        traj_b_states  = traj_b['states'][sample['trajectories'][1][1]]
        traj_b_actions = traj_b['actions'][sample['trajectories'][1][1]]

        traj_a_states = torch.from_numpy(np.array(traj_a_states)).float()/255.0
        traj_a_actions = torch.from_numpy(np.array(traj_a_actions)).long()
        traj_b_states = torch.from_numpy(np.array(traj_b_states)).float()/255.0
        traj_b_actions = torch.from_numpy(np.array(traj_b_actions)).long()
        label  = torch.from_numpy(np.array([label]))

        traj_a = (traj_a_states, traj_a_actions)
        traj_b = (traj_b_states, traj_b_actions)

        return traj_a, traj_b, label

    def shuffle(self):
        random.shuffle(self.shuffled)

    def __len__(self):
        return self.length

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        #self.env.close()
        #self.demo_env.close()
        pass

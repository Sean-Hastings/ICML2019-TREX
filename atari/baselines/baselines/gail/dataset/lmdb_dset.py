'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np
import lmdb
import pickle

import os.path
import random


class LMDB_Dset(object):
    def __init__(self, expert_path, train_fraction=0.99, traj_limitation=-1, randomize=True):
        super(LMDB_Dset, self).__init__()
        self.data_path = expert_path
        self.train_fraction = train_fraction
        self.shuffle = randomize
        # Open the LMDB file
        with lmdb.open(self.data_path,
                             readonly=True,
                             lock=False,
                             meminit=False) as env:

            with env.begin(write=False) as txn:
                self.demos_path = pickle.loads(txn.get(b'__demo_path__'))

        with lmdb.open(self.demos_path,
                             readonly=True,
                             lock=False,
                             meminit=False) as env:

            with env.begin(write=False) as txn:
                self.keys = pickle.loads(txn.get(b'__keys__'))
                self.keys = sorted(self.keys, key=lambda x: -1 * pickle.loads(txn.get(x))['return'])
                self.length = len(self.keys)

        self.demo_env = None

        self.train_shuffled = [i for i in range(self.length*self.train_fraction)]
        self.eval_shuffled = [i for i in range(self.length - self.length*self.train_fraction)]
        self.shuffled = [i for i in range(self.length)]

        self.train_pointer = 0
        self.eval_pointer = 0
        self.pointer = 0

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            inds = self.shuffled[self.pointer:self.pointer+batch_size]
            self.pointer += batch_size
            if self.pointer > len(self.shuffled):
                self.pointer = 0
                random.shuffle(self.shuffled)
        elif split == 'train':
            inds = self.train_shuffled[self.train_pointer:self.train_pointer+batch_size]
            self.train_pointer += batch_size
            if self.train_pointer > len(self.train_shuffled):
                self.train_pointer = 0
                random.shuffle(self.train_shuffled)
        elif split == 'val':
            inds = self.eval_shuffled[self.eval_pointer:self.eval_pointer+batch_size]
            self.eval_pointer += batch_size
            if self.eval_pointer > len(self.eval_shuffled):
                self.eval_pointer = 0
                random.shuffle(self.eval_shuffled)
        else:
            raise NotImplementedError

        samples = [self[idx] for idx in inds]

        states, actions = list(zip(*samples))

        return np.concatenate(states), np.concatenate(actions)


    def __getitem__(self, idx):
        if self.demo_env is None:
            self.demo_env = lmdb.open(self.demos_path,
                                      readonly=True,
                                      lock=False,
                                      meminit=False)

        with self.demo_env.begin(write=False) as txn:
            traj = pickle.loads(txn.get(self.keys[idx]))

        states  = traj['states']
        actions = traj['actions']

        states = np.array(traj_a_states, dtype=np.float32)/255.0
        actions = np.array(traj_a_actions, dtype=np.int32)

        return states, actions






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    print(args)

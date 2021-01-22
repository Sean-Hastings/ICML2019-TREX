import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.


import pickle
import os
import sys
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.agent import *
from baselines.common.trex_utils import preprocess
from torch.utils.data import DataLoader
import torch.optim as optim


from utils.model import Trex_Net
from utils.constants import MUJOCO_ENVS, get_env_id_type, get_checkpoint_range
from utils.demos import generate_demos, create_training_data
from utils.dataset import LMDBDataset


# Train the network
def learn_reward(reward_network, optimizer, dataset, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0
    for epoch in range(num_iter):
        dloader = DataLoader(dataset, shuffle=True, pin_memory=True, num_workers=8)

        for i, data in enumerate(dloader):
            traj_i, traj_j, labels = data
            traj_i    = [traj_i[0][0].to(device)]
            traj_j    = [traj_j[0][0].to(device)]
            labels    = labels[0].to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network(traj_i, traj_j)
            loss = loss_criterion(outputs, labels.long()) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")

    args = parser.parse_args()

    #########################################
    # Hack to make the grid scripts cleaner #
    #########################################

    if args.env_name == 'enduro':
        args.num_trajs = 50000
        args.num_snippets = 0
    else:
        args.num_trajs = 0
        args.num_snippets = 50000


    #########################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    env_name = args.env_name
    env_id, env_type = get_env_id_type(env_name)

    if args.seed == 0:
        args.seed = int((time.time()%1)*100000)

    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    os.makedirs('learned_models', exist_ok=True)
    id = '_s={}_t={}_{}_{}'.format(args.num_snippets, args.num_trajs, 'trex', args.seed)
    model_name = args.env_name + id
    args.model_path = 'learned_models/' + model_name + '.params'

    print("Training reward for", env_id)
    num_trajs =  args.num_trajs
    num_snippets = args.num_snippets
    min_snippet_length = 50 #min length of trajectory for training comparison
    max_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True

    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    checkpoint_range = get_checkpoint_range(env_name, demo=True)

    generate_demos(env, env_name, agent, args.models_dir, checkpoint_range, episodes_per_checkpoint=10)
    create_training_data(env_name, num_trajs, num_snippets, min_snippet_length, max_snippet_length)

    # Now we create a reward network and optimize it using the training data.
    net = Trex_Net(env.action_space.n, model_name)
    net.to(device)
    optimizer = optim.Adam(net.parameters(),  lr=lr, weight_decay=weight_decay)

    with LMDBDataset('datasets/' + env_name + ('_%d_%d.lmdb' % (num_snippets, num_trajs))) as dset:
        learn_reward(net, optimizer, dset, num_iter, l1_reg, args.model_path)

    net.eval()

    #with LMDBDataset('datasets/' + env_name + ('_%d_%d.lmdb' % (num_snippets, num_trajs))) as dset:
    #    print("accuracy", calc_accuracy(net, dset))

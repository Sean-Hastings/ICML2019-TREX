import argparse
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.agent import *
from utils.model import Net
from utils.constants import MUJOCO_ENVS, get_env_id_type, get_checkpoint_range
from utils.demos import generate_demos, create_training_data


_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    sys.stdout.flush()


# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, batch_size, l1_reg, checkpoint_dir):
    reward_network.train()
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    debug=True
    print_interval = 100

    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        epoch_loss = 0
        start_time = time.time()
        for i in list(range(len(training_labels))):
            print_epoch = i % print_interval == 0

            batch_inds = slice(i, i + 1)
            traj_i, traj_j = zip(*training_obs[batch_inds])
            traj_i = [torch.from_numpy(np.array(traj)).float().to(device) for traj in traj_i]
            traj_j = [torch.from_numpy(np.array(traj)).float().to(device) for traj in traj_j]
            labels = torch.from_numpy(np.array(training_labels[batch_inds])).to(device)

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            #outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels.long()).mean() + l1_reg * abs_rewards
            loss.backward()

            if i % batch_size == 0:
                '''
                print('')
                print('###########################')
                _norm = []
                for p in reward_network.parameters():
                    if p.grad is not None:
                        _norm += [p.grad.view(-1).detach()]
                _norm = torch.cat(_norm).norm()
                print('grad norm pre-clip: {}'.format(_norm))
                '''

                clip_grad_norm_(reward_network.parameters(), 10)

                '''
                _norm = []
                for p in reward_network.parameters():
                    if p.grad is not None:
                        _norm += [p.grad.view(-1).detach()]
                _norm = torch.cat(_norm).norm()
                print('grad norm post-clip: {}'.format(_norm))
                print(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                '''

                optimizer.step()
                optimizer.zero_grad()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            epoch_loss += item_loss
            # The printed loss may not be perfectly accurate but good enough?
            if print_epoch:
                #print(i)
                fps = batch_size / (time.time() - start_time)
                if i > 0:
                    cum_loss = cum_loss / print_interval
                print("epoch {}:{}/{} loss {}  |  fps {}".format(epoch+1, i, len(training_labels), cum_loss, fps), end='\r')
                #print(abs_rewards)
                cum_loss = 0.0
                #print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
                start_time = time.time()
        if debug:
            print('\n\n                                       ####\n')
        print('epoch {} average loss: {}'.format(epoch+1, epoch_loss / len(training_labels)))
        #'''
        for g in optimizer.param_groups:
            g['lr'] *= 0.8
        #'''
    print("finished training")



def calc_accuracy(reward_network, training_inputs, training_outputs):
    reward_network.eval()
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
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    env_name = args.env_name
    env_id, env_type = get_env_id_type(env_name)

    os.makedirs('learned_models', exist_ok=True)
    id = '_' + 's={}'.format(args.num_snippets) + '_t={}'.format(args.num_trajs)
    args.reward_model_path = 'learned_models/' + args.env_name + id + '.params'

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    num_trajs =  args.num_trajs
    num_snippets = args.num_snippets
    min_snippet_length = 100 # 50 # min length of trajectory for training comparison
    max_snippet_length = 500 # 100

    lr = 0.001 # 0.00005
    weight_decay = 0.0
    num_iter = 25 # 5 #num times through training data
    l1_reg = 0.00001
    stochastic = True

    batch_size = 32

    env = make_vec_env(env_id, env_type, 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    if env_type == 'atari':
        env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    checkpoint_range = get_checkpoint_range(env_name, demo=True)

    demonstrations, learning_returns, _, _, _ = generate_demos(env, env_name, agent, args.models_dir, checkpoint_range)

    #print(len(learning_returns))
    #print(len(demonstrations))
    #print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])
    sorted_returns, demonstrations = zip(*demonstrations)
    print('trajectory returns:', sorted_returns)

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths:", demo_lengths)

    training_obs, training_labels = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))

    # Now we create a reward network and optimize it using the training data.
    reward_net = Net()
    reward_net.to(device)
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, batch_size, l1_reg, args.reward_model_path)
    #save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    reward_net.eval()

    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [reward_net.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item() for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    #print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

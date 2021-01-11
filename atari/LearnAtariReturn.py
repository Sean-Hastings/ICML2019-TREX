import argparse
import time
import numpy as np
import pickle
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.agent import *
from utils.model import Net
from utils.constants import MUJOCO_ENVS, get_env_id_type, get_checkpoint_range
from utils.demos import generate_demos, create_training_data
from utils.dataset import LMDBDataset


_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    sys.stdout.flush()


# Train the network
def learn_return(network, optimizer, dataset, log_dir, args):
    network.train()
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    logs = [[],[],[],[]] # (losses, epoch_losses, accuracies, magnitudes)

    retsymb = '\n' if args.grid else '\r'
    debug = True
    print_interval = 100

    for epoch in range(args.num_iter):
        dloader = DataLoader(dataset, shuffle=True, pin_memory=True, num_workers=8)

        n_correct  = 0
        frames     = 0
        cum_loss   = 0.0
        cum_mag    = 0.0
        epoch_loss = 0
        start_time = time.time()
        for i, data in enumerate(dloader):
            print_epoch = i % print_interval == 0

            traj_i, traj_j, labels = data
            actions_i = [traj_i[1][0].to(device)]
            actions_j = [traj_j[1][0].to(device)]
            traj_i    = [traj_i[0][0].to(device)]
            traj_j    = [traj_j[0][0].to(device)]
            labels    = labels[0].to(device)

            frames += traj_i[0].shape[0] + traj_j[0].shape[0]

            if args.bc:
                for traj, actions in zip(traj_i+traj_j, actions_i+actions_j):
                    outputs = network.bc(traj)
                    loss = loss_criterion(outputs, actions.long().view(-1)).mean()
                    loss.backward()
                    abs = torch.Tensor([0])

            else:
                #forward + backward + optimize
                outputs, abs = network.forward(traj_i, traj_j, actions_i, actions_j, print_epoch)
                #outputs = outputs.unsqueeze(0)
                loss = loss_criterion(outputs, labels.long()).mean()
                if loss < 0.693:
                    n_correct += 1
                loss = loss + args.l1_reg * abs
                loss.backward()


            if i % args.batch_size == 0:

                '''
                print('')
                print('###########################')
                _norm = []
                for p in network.parameters():
                    if p.grad is not None:
                        _norm += [p.grad.view(-1).detach()]
                _norm = torch.cat(_norm).norm()
                print('grad norm pre-clip: {}'.format(_norm))


                clip_grad_norm_(network.parameters(), 10)


                _norm = []
                for p in network.parameters():
                    if p.grad is not None:
                        _norm += [p.grad.view(-1).detach()]
                _norm = torch.cat(_norm).norm()
                print('grad norm post-clip: {}'.format(_norm))
                print(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                '''

                optimizer.step()
                optimizer.zero_grad()

            #print stats to see if learning
            item_loss   = loss.item()
            epoch_loss += item_loss
            cum_loss   += item_loss
            cum_mag    += abs.item()
            # The printed loss may not be perfectly accurate but good enough?
            if print_epoch:
                #print(i)
                eps = print_interval / (time.time() - start_time)
                fps = frames / (time.time() - start_time)
                frames = 0
                if i > 0:
                    cum_loss = cum_loss / print_interval
                    cum_mag  = cum_mag / print_interval
                print("epoch {}:{}/{} loss {} mag {}   |   eps {} fps {}".format(epoch+1, i, len(dataset), cum_loss, cum_mag, eps, fps), end=retsymb)
                logs[0] += [cum_loss]
                logs[3] += [cum_mag]
                #print(abs_rewards)
                cum_loss = 0.0
                cum_mag  = 0.0
                start_time = time.time()
        #if debug:
        #    print('\n\n                                       ####\n')
        accuracy = n_correct / len(dloader)
        print('epoch {} average loss: {} average accuracy: {}                                  '.format(epoch+1, epoch_loss / len(dataset), accuracy))
        logs[1] += [epoch_loss / len(dataset)]
        logs[2] += [accuracy]
        #'''
        for g in optimizer.param_groups:
            g['lr'] *= 0.95
        #'''

        #print("check pointing")
        torch.save(net.state_dict(), args.model_path)

    with open(log_dir, 'wb') as f:
        pickle.dump(logs, f)
    print("finished training")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--resume', default=False, action='store_true', help="flag to resume from existing save instead of overwriting")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--grid', default=False, action='store_true', help="training on grid")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--num_iter', default=50, type=int, help="number of epochs")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--l1_reg', default=0.01, type=float, help="l1 regularization on the magnitudes of the mean Q/advantage values")
    parser.add_argument('--batch_size', default=32, type=int, help="number of (*sequentially backpropped*) samples between weight updates")
    parser.add_argument('--bc', default=False, action='store_true', help='train bc objective instead of no-trex')
    parser.add_argument('--data_only', default=False, action='store_true', help="don't train, just collect and prep data")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    env_name = args.env_name
    env_id, env_type = get_env_id_type(env_name)

    if args.seed == 0:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    os.makedirs('learned_models', exist_ok=True)
    id = '_s={}_t={}_{}_{}'.format(args.num_snippets, args.num_trajs, 'bc' if args.bc else 'trax', args.seed)
    model_name = args.env_name + id
    args.model_path = 'learned_models/' + model_name + '.params'
    log_path = 'logs/' + model_name + '.log'
    os.makedirs('logs', exist_ok=True)

    print(args)
    print("Training reward for", env_id)
    num_trajs =  args.num_trajs
    num_snippets = args.num_snippets
    min_snippet_length = 100 # 50 # min length of trajectory for training comparison
    max_snippet_length = 500 # 100

    env = make_vec_env(env_id, env_type, 1, args.seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)

    stochastic = True
    agent = PPO2Agent(env, env_type, stochastic)

    checkpoint_range = get_checkpoint_range(env_name, demo=True)

    generate_demos(env, env_name, agent, args.models_dir, checkpoint_range, episodes_per_checkpoint=10)
    create_training_data(env_name, num_trajs, num_snippets, min_snippet_length, max_snippet_length)

    if not args.data_only:
        # Now we create a reward network and optimize it using the training data.
        net = Net(env.action_space.n, model_name)
        if args.resume:
            print('resuming from saved checkpoint')
            net.load_state_dict(torch.load(args.model_path))
        net.to(device)
        optimizer = optim.Adam(net.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

        with LMDBDataset('datasets/' + env_name + ('_%d_%d.lmdb' % (num_snippets, num_trajs))) as dset:
            learn_return(net, optimizer, dset, log_path, args)

        net.eval()

        #print("accuracy", calc_accuracy(net, training_obs, training_labels))

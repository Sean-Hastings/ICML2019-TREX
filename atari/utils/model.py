import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import accumulate



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, 1)


    def score_states(self, traj):
        '''caltulate per-observation reward of trajectory'''
        if len(traj.shape) == 4:
            x = traj.permute(0,3,1,2) #get into NCHW format
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            x = F.leaky_relu(self.fc1(x))
            r = self.fc2(x)
        else:
            x = traj
            r = None
        return r.view(-1)


    def cum_return(self, traj):
        '''DEPRECATED; calculate cumulative return of trajectory'''
        r = self.score_states(traj)
        sum_rewards = torch.sum(r).view(-1)
        sum_abs_rewards = torch.sum(torch.abs(r)).view(-1)
        return sum_rewards, sum_abs_rewards


    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        if not isinstance(traj_i, (tuple, list)):
            traj_i = [traj_i]
            traj_j = [traj_j]
        lengths = [len(t) for traj in (traj_i, traj_j) for t in traj]
        accum   = [0] + list(accumulate(lengths))
        accum   = list(zip(accum[:-1], accum[1:]))
        states  = torch.cat(traj_i + traj_j)
        rewards = self.score_states(states)
        r_i     = [rewards[acc[0]:acc[1]] for acc in accum[:len(accum)//2]]
        r_j     = [rewards[acc[0]:acc[1]] for acc in accum[len(accum)//2:]]
        cum_r_i = torch.cat([torch.sum(r).view(-1) for r in r_i])
        abs_r_i = torch.cat([torch.sum(torch.abs(r)).view(-1) for r in r_i])
        cum_r_j = torch.cat([torch.sum(r).view(-1) for r in r_j])
        abs_r_j = torch.cat([torch.sum(torch.abs(r)).view(-1) for r in r_j])
        comp_r  = torch.stack([cum_r_i, cum_r_j], dim=-1)
        comp_r  = torch.softmax(comp_r, dim=-1)
        return comp_r, abs_r_i + abs_r_j

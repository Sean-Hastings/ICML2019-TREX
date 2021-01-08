import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import accumulate



class Net(nn.Module):
    def __init__(self, action_space, name):
        super().__init__()
        self.action_space = action_space
        self.name = name

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, action_space)


    def act(self, ob):
        x = ob.permute(0,3,1,2).contiguous() #get into NCHW format
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)[0]
        return torch.argmax(r)


    def score_states(self, traj, actions):
        '''caltulate per-observation reward of trajectory'''
        if len(traj.shape) == 4:
            ran = torch.arange(traj.shape[0])

            x = traj.permute(0,3,1,2).contiguous() #get into NCHW format
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            x = F.leaky_relu(self.fc1(x))
            r = self.fc2(x)[ran, actions.view(-1)]
        else:
            x = traj
            r = None
        return r.view(-1)


    def forward(self, traj_i, traj_j, actions_i, actions_j):
        '''compute cumulative return for each trajectory and return logits'''
        if not isinstance(traj_i, (tuple, list)):
            traj_i    = [traj_i]
            traj_j    = [traj_j]
            actions_i = [actions_i]
            actions_j = [actions_j]
        lengths = [len(t) for traj in (traj_i, traj_j) for t in traj]
        accum   = [0] + list(accumulate(lengths))
        accum   = list(zip(accum[:-1], accum[1:]))
        states  = torch.cat(traj_i + traj_j)
        actions = torch.cat(actions_i + actions_j)
        rewards = self.score_states(states, actions)
        r_i     = [rewards[acc[0]:acc[1]] for acc in accum[:len(accum)//2]]
        r_j     = [rewards[acc[0]:acc[1]] for acc in accum[len(accum)//2:]]
        cum_r_i = torch.cat([torch.mean(r).view(-1) for r in r_i])
        cum_r_j = torch.cat([torch.mean(r).view(-1) for r in r_j])
        comp_r  = torch.stack([cum_r_i, cum_r_j], dim=-1)
        comp_r  = torch.softmax(comp_r, dim=-1)
        return comp_r

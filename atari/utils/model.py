import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import accumulate



class Net(nn.Module):
    def __init__(self, action_space, name):
        super().__init__()
        self.action_space = action_space
        self.name = name

        multiplier = 4

        self.filter_count = 64
        self.linshape  = 784*multiplier
        self.linshape2 = 512

        self.conv1 = nn.Conv2d(4, self.filter_count, 7, stride=3)
        self.conv2 = nn.Conv2d(self.filter_count, self.filter_count, 5, stride=2)
        self.conv3 = nn.Conv2d(self.filter_count, self.filter_count, 3, stride=1)
        self.conv4 = nn.Conv2d(self.filter_count, 16*multiplier, 3, stride=1)
        self.fc1 = nn.Linear(self.linshape, self.linshape2)
        self.fc2 = nn.Linear(self.linshape2, action_space)


    def act(self, ob):
        x = ob.permute(0,3,1,2).contiguous() #get into NCHW format
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, self.linshape)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x).view(-1)
        return torch.argmax(r)


    def bc(self, traj):
        x = traj.permute(0,3,1,2).contiguous() #get into NCHW format
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, self.linshape)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        return torch.softmax(r, dim=-1)


    def score_states(self, traj, actions, _print=False):
        '''caltulate per-observation reward of trajectory'''
        if len(traj.shape) == 4:
            ran = torch.arange(traj.shape[0])

            x = traj.permute(0,3,1,2).contiguous() #get into NCHW format
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, self.linshape)
            x = F.leaky_relu(self.fc1(x))
            r = self.fc2(x)
            abs = torch.abs(r).mean() # encourage zero-centered predictions
            r = r
            if _print:
                print(r[:10].cpu().detach().numpy(), r[-10:].cpu().detach().numpy())
                print(r[:10].argmax(dim=1).cpu().detach().numpy(), r[-10:].argmax(dim=1).cpu().detach().numpy())
                print(actions[:10].view(-1).cpu().detach().numpy(), actions[-10:].view(-1).cpu().detach().numpy())
            r = r[ran, actions.view(-1)]
        else:
            x   = traj
            r   = None
            abs = None
        return r.view(-1), abs


    def forward(self, traj_i, traj_j, actions_i, actions_j, _print=False):
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
        rewards, abs = self.score_states(states, actions, _print)
        r_i     = [rewards[acc[0]:acc[1]] for acc in accum[:len(accum)//2]]
        r_j     = [rewards[acc[0]:acc[1]] for acc in accum[len(accum)//2:]]
        cum_r_i = torch.cat([torch.mean(r).view(-1) for r in r_i])
        cum_r_j = torch.cat([torch.mean(r).view(-1) for r in r_j])
        comp_r  = torch.stack([cum_r_i, cum_r_j], dim=-1)
        if _print:
            print(comp_r.cpu().detach().numpy())
        comp_r  = torch.softmax(comp_r, dim=-1)
        if _print:
            print(comp_r.cpu().detach().numpy())
            print('==========================')
        return comp_r, abs

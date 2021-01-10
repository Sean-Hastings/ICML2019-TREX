import numpy as np
import torch
import pickle
import sys
import os.path
import argparse
from os import makedirs

sys.path.append('./baselines/')
from baselines.common.trex_utils import preprocess
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from utils.model import Net
from utils.constants import get_env_id_type
from utils.agent import *


_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    sys.stdout.flush()



def generate_demos(env, env_name, model, agent, device, save_dir='evals', episodes=100):
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + '/' + model.name + '.log'
    if os.path.exists(save_path):
        print('evaluation not completed as %s already exists' % save_dir)
        return
        
    print('evaluating {}'.format(env_name))

    model_path = "models/" + env_name + "_25/01050"
    if env_name == "seaquest":
        model_path = "models/" + env_name + "_5/00035"

    agent.load(model_path)

    logs = [[], []] # steps, return
    makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(episodes):
            done = False
            r = 0
            ob = preprocess(env.reset(), env_name)
            steps = 0
            acc_reward = 0
            while True:
                a_act = agent.act(ob, r, done)
                ob = torch.from_numpy(ob).float().to(device)
                action = model.act(ob)

                #print(a_act, action)

                ob, r, done, _ = env.step(action)
                #env.render()
                ob = preprocess(ob, env_name)
                acc_reward += r[0]
                steps += 1
                if done:
                    print("steps: {}, return: {}".format(steps, acc_reward))
                    logs[0] += [steps]
                    logs[1] += [acc_reward]
                    break

    print('return stats:')
    print('min: {}'.format(np.min(logs[1])))
    print('mean: {}'.format(np.mean(logs[1])))
    print('max: {}'.format(np.max(logs[1])))

    with open(save_path, 'wb') as f:
        pickle.dump(logs, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_episodes', default=100, type=int, help="number of episodes to eval")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_id, env_type = get_env_id_type(args.env_name)
    env = make_vec_env(env_id, env_type, 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    env = VecFrameStack(env, 4)

    agent = PPO2Agent(env, env_type, True)

    model = Net(env.action_space.n, args.model_path[args.model_path.find('learned_models')+len('learned_models/'):args.model_path.find('.params')])
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    generate_demos(env, args.env_name, model, agent, device, save_dir='evals', episodes=args.num_episodes)

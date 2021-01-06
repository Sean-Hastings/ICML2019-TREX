import numpy as np
import torch
import lmdb
import pickle
import sys
import os.path
from os import makedirs
from baselines.common.trex_utils import preprocess

from .constants import MUJOCO_ENVS



_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    sys.stdout.flush()



def generate_demos(env, env_name, agent, model_dir, checkpoint_range, save_dir='demos', episodes_per_checkpoint=5, map_increment=1e9):
    save_path = save_dir + '/' + env_name + '.lmdb'
    if os.path.exists(save_path):
        print('Demonstrations not collected as %s already exists' % save_path)
        return

    checkpoints = []
    for i in checkpoint_range:
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)

    makedirs(save_dir, exist_ok=True)
    map_counter = 1
    keys = []
    with lmdb.open(save_path, map_size=map_counter*map_increment) as lmdb_env:
        for checkpoint in checkpoints:
            model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
            if env_name == "seaquest":
                model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

            agent.load(model_path)
            for i in range(episodes_per_checkpoint):
                done = False
                traj = []
                gt_rewards = []
                actions = []
                r = 0

                ob = env.reset()
                steps = 0
                acc_reward = 0
                while True:
                    action = agent.act(ob, r, done)
                    ob, r, done, _ = env.step(action)
                    ob_processed = preprocess(ob, env_name)
                    traj.append(ob_processed)
                    actions.append(action)

                    gt_rewards.append(r[0])
                    acc_reward += r[0]
                    steps += 1
                    if done:
                        print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
                        break


                traj = (np.concatenate(traj, axis=0)*255).astype(np.uint8)
                actions = np.array(actions)
                gt_rewards = np.array(gt_rewards)
                value = {'states':traj,
                         'actions':actions,
                         'rewards':gt_rewards,
                         'length':steps,
                         'return':acc_reward}
                key = '%s_%s_%d' % (env_name, checkpoint, i)
                lmdb_env, key = lmdb_submit(key, value, lmdb_env, save_path, map_counter, map_increment)
                keys += [key]
        with lmdb_env.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
    print('%d total demonstrations gathered' % len(keys))


def create_training_data(env_name, num_trajs, num_snippets, min_snippet_length, max_snippet_length, demo_dir='demos', save_dir='datasets', map_increment=1e9):
    save_path = save_dir + '/' + env_name + ('_%d_%d.lmdb' % (num_snippets, num_trajs))
    if os.path.exists(save_path):
        print('Dataset not created as %s already exists' % save_path)
        return
    demos_path = demo_dir + '/' + env_name + '.lmdb'
    if not os.path.exists(demos_path):
        print('Dataset not created as %s does not exist' % demos_path)
        return

    makedirs(save_dir, exist_ok=True)
    max_traj_length = 0
    keys = []
    with lmdb.open(demos_path, readonly=True, map_size=map_increment) as lmdb_demos_env:
        with lmdb_demos_env.begin() as txn:
            demo_keys = pickle.loads(txn.get(b'__keys__'))
            num_demos = len(demo_keys)
        with lmdb.open(save_path, map_size=map_increment) as lmdb_data_env:
            with lmdb_data_env.begin(write=True) as txn:
                txn.put(b'__demo_path__', pickle.dumps(demos_path))
            #add full trajs
            if num_trajs > 0:
                max_step = 6
                max_trajs = num_demos * (num_demos-1) * (max_step-1) * (max_step-1) / 2
                print('Building %d of approxmately %d possible trajectories...' % (num_trajs, max_trajs))
            for n in range(num_trajs):
                print('%d of %d complete...' % (n, num_trajs), end='\r')
                ti_key, traj_i, tj_key, traj_j, label = get_random_demos(demo_keys, lmdb_demos_env)

                #create random partial trajs by finding random start frame and random skip frame

                step = np.random.randint(2, max_step + 1)
                si = np.random.randint(step)
                sj = np.random.randint(step)

                traj_i = slice(si, traj_i['length'], step)
                traj_j = slice(sj, traj_j['length'], step)
                value = {'trajectories': ((ti_key, traj_i), (tj_key, traj_j)),
                         'label': label}
                key = 't_%d' % n
                lmdb_data_env, key = lmdb_submit(key, value, lmdb_data_env, save_path, 1, map_increment)
                keys += [key]

                #max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

            #fixed size snippets with progress prior
            if num_snippets > 0:
                print('Building %d snippets...' % num_snippets)
            for n in range(num_snippets):
                print('%d of %d complete...' % (n, num_snippets), end='\r')
                ti_key, traj_i, tj_key, traj_j, label = get_random_demos(demo_keys, lmdb_demos_env)

                #create random snippets
                #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
                min_length = min(traj_i['length'], traj_j['length'])
                rand_length = np.random.randint(min(min_snippet_length, min_length-1), min(max_snippet_length, min_length))
                ti_start = np.random.randint(traj_i['length'] - rand_length + 1)
                tj_start = np.random.randint(traj_j['length'] - rand_length + 1)
                traj_i = slice(ti_start, ti_start+rand_length)
                traj_j = slice(tj_start, tj_start+rand_length)
                value = {'trajectories': ((ti_key, traj_i), (tj_key, traj_j)),
                         'label': label}
                key = 't_%d' % n
                lmdb_data_env, key = lmdb_submit(key, value, lmdb_data_env, save_path, 1, map_increment)
                keys += [key]

                #max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

            with lmdb_data_env.begin(write=True) as txn:
                txn.put(b'__keys__', pickle.dumps(keys))

    #print("maximum traj length", max_traj_length)


def get_random_demos(demo_keys, lmdb_demos_env):
    num_demos = len(demo_keys)
    ti_return = 0
    tj_return = 0
    #only add trajectories that are different returns
    while(ti_return == tj_return):
        #pick two random demonstrations
        ti_key = demo_keys[np.random.randint(num_demos)]
        tj_key = demo_keys[np.random.randint(num_demos)]
        with lmdb_demos_env.begin() as txn:
            traj_i = pickle.loads(txn.get(ti_key))
            traj_j = pickle.loads(txn.get(tj_key))
        ti_return = traj_i['return']
        tj_return = traj_j['return']

    if ti_return > tj_return:
        label = 0
    else:
        label = 1

    return ti_key, traj_i, tj_key, traj_j, label


def lmdb_submit(key, value, env, env_path, map_counter, map_increment):
    key = key.encode('ascii')
    value = pickle.dumps(value)
    try:
        with env.begin(write=True) as txn:
            txn.put(key, value)
        return env, key
    except lmdb.MapFullError:
        #print('map full: %d' % map_counter)
        map_counter += 1
        env.close()
        env = lmdb.open(env_path, map_size=map_counter*map_increment)
        return lmdb_submit(key.decode('ascii'), pickle.loads(value), env, env_path, map_counter, map_increment)

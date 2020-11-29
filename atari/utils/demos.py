import numpy as np
import torch
from baselines.common.trex_utils import preprocess

from .constants import MUJOCO_ENVS



def generate_demos(env, env_name, agent, model_dir, checkpoint_range, reward_model=None, device=None):
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

    demonstrations = []
    learning_returns = []
    learning_rewards = []
    pred_returns     = [] if reward_model is not None else None
    pred_rewards     = [] if reward_model is not None else None
    for checkpoint in checkpoints:
        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint
        elif env_name in MUJOCO_ENVS:
            model_path = model_dir + "/models/" + env_name + "/checkpoints/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)

                gt_rewards.append(r[0])
                acc_reward += r[0]
                steps += 1
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(np.array(traj))
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)
            if reward_model is not None:
                assert device is not None
                with torch.no_grad():
                    pred_returns.append(reward_model.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].cpu().numpy())
                    pred_rewards.append(reward_model.score_states(torch.from_numpy(np.array(traj)).float().to(device)).cpu().numpy())

    learning_returns = np.array(learning_returns).reshape(-1)
    pred_returns     = np.array(pred_returns).reshape(-1)
    if reward_model is not None:
        learning_rewards = np.concatenate(learning_rewards).reshape(-1)
        pred_rewards     = np.concatenate(pred_rewards).reshape(-1)
    return demonstrations, learning_returns, learning_rewards, pred_returns, pred_rewards



def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        max_step = 6
        si = np.random.randint(max_step)
        sj = np.random.randint(max_step)
        step = np.random.randint(3, max_step + 1)

        traj_i = demonstrations[ti][si::step]  #slice(start,stop,step)
        traj_j = demonstrations[tj][sj::step]

        if ti > tj:
            label = 0
        else:
            label = 1

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min(min_snippet_length, min_length-1), min(max_snippet_length, min_length))
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        #traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2]
        #traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length]
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels

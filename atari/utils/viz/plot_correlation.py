import numpy as np
from glob import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')


graph_dir = 'graphs/'
envs  = [env[env.rfind('\\')+1:] for env in glob(graph_dir + '*') if not '.' in env]
corrs = [[], [], [], [], []]

for env in envs:
    try:
        save_fig_dir = graph_dir + env

        # Return Info
        learning_returns_extrapolate = np.load(save_fig_dir + '/np/lree.npy')
        learning_returns_demos       = np.load(save_fig_dir + '/np/lred.npy')
        pred_returns_extrapolate     = np.load(save_fig_dir + '/np/pree.npy')
        pred_returns_demos           = np.load(save_fig_dir + '/np/pred.npy')
        length_demos_extrapolate     = np.load(save_fig_dir + '/np/lde.npy')
        length_demos_demos           = np.load(save_fig_dir + '/np/ldd.npy')
        learning_returns_all         = np.concatenate([learning_returns_extrapolate, learning_returns_demos])
        pred_returns_all             = np.concatenate([pred_returns_extrapolate, pred_returns_demos])
        length_demos_all             = np.concatenate([length_demos_extrapolate, length_demos_demos])

        # Reward Info
        learning_rewards_extrapolate = np.load(save_fig_dir + '/np/lre.npy')
        learning_rewards_demos       = np.load(save_fig_dir + '/np/lrd.npy')
        pred_rewards_extrapolate     = np.load(save_fig_dir + '/np/pre.npy')
        pred_rewards_demos           = np.load(save_fig_dir + '/np/prd.npy')
        learning_rewards_all         = np.concatenate([learning_rewards_extrapolate, learning_rewards_demos])
        pred_rewards_all             = np.concatenate([pred_rewards_extrapolate, pred_rewards_demos])
    except FileNotFoundError:
        envs = [e for e in envs if not e == env]
        print('Environment "{}" failed'.format(env))
        continue

    corr_extrapolate = np.corrcoef(learning_rewards_extrapolate, pred_rewards_extrapolate)[0][1]
    corr_demos       = np.corrcoef(learning_rewards_demos, pred_rewards_demos)[0][1]
    corr_all         = np.corrcoef(learning_rewards_all, pred_rewards_all)[0][1]

    energy = [corr_demos, corr_extrapolate, corr_all]
    corrs[0] += energy
    x = ['demos', 'extrapolate', 'all']
    x_pos = np.arange(len(x))


    plt.bar(x_pos, energy)
    plt.xlabel("Data Source")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.title("Correlation of Predicted and Ground Truth Rewards")

    plt.xticks(x_pos, x)
    plt.savefig(save_fig_dir + '/' + env + "_gt_vs_pred_rewards_correlation.png")
    plt.clf()

    corr_extrapolate = np.corrcoef(learning_returns_extrapolate, pred_returns_extrapolate)[0][1]
    corr_demos       = np.corrcoef(learning_returns_demos, pred_returns_demos)[0][1]
    corr_all         = np.corrcoef(learning_returns_all, pred_returns_all)[0][1]

    energy = [corr_demos, corr_extrapolate, corr_all]
    corrs[1] += energy
    x = ['demos', 'extrapolate', 'all']
    x_pos = np.arange(len(x))


    plt.bar(x_pos, energy)
    plt.xlabel("Data Source")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.title("Correlation of Predicted and Ground Truth Returns")

    plt.xticks(x_pos, x)
    plt.savefig(save_fig_dir + '/' + env + "_gt_vs_pred_returns_correlation.png")
    plt.clf()

    corr_extrapolate = np.corrcoef(learning_returns_extrapolate, length_demos_extrapolate)[0][1]
    corr_demos       = np.corrcoef(learning_returns_demos, length_demos_demos)[0][1]
    corr_all         = np.corrcoef(learning_returns_all, length_demos_all)[0][1]

    energy = [corr_demos, corr_extrapolate, corr_all]
    corrs[2] += energy
    x = ['demos', 'extrapolate', 'all']
    x_pos = np.arange(len(x))


    plt.bar(x_pos, energy)
    plt.xlabel("Data Source")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.title("Correlation of Trajectory Length and Ground Truth Returns")

    plt.xticks(x_pos, x)
    plt.savefig(save_fig_dir + '/' + env + "_gt_returns_vs_traj_length_correlation.png")
    plt.clf()

    corr_extrapolate = np.corrcoef(length_demos_extrapolate, pred_returns_extrapolate)[0][1]
    corr_demos       = np.corrcoef(length_demos_demos, pred_returns_demos)[0][1]
    corr_all         = np.corrcoef(length_demos_all, pred_returns_all)[0][1]

    energy = [corr_demos, corr_extrapolate, corr_all]
    corrs[3] += energy
    x = ['demos', 'extrapolate', 'all']
    x_pos = np.arange(len(x))


    plt.bar(x_pos, energy)
    plt.xlabel("Data Source")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.title("Correlation of Trajectory Length and Predicted Returns")

    plt.xticks(x_pos, x)
    plt.savefig(save_fig_dir + '/' + env + "_pred_returns_vs_traj_length_correlation.png")
    plt.clf()

    corr_reward   = np.corrcoef(learning_rewards_all, pred_rewards_all)[0][1]
    corr_return   = np.corrcoef(learning_returns_all, pred_returns_all)[0][1]
    corr_len_gt   = np.corrcoef(length_demos_all, learning_returns_all)[0][1]
    corr_len_pred = np.corrcoef(length_demos_all, pred_returns_all)[0][1]

    energy = [corr_reward, corr_return, corr_len_gt, corr_len_pred]
    corrs[4] += energy
    x = ['reward', 'return', 'len vs gt', 'len vs pred']
    x_pos = np.arange(len(x))


    plt.bar(x_pos, energy)
    plt.xlabel("Compared Values")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.title("Correlation of Rewards, Returns, and Trajectory Lengths")

    plt.xticks(x_pos, x)
    plt.savefig(save_fig_dir + '/' + env + "_rewards_returns_len_correlation.png")
    plt.clf()


width = 1 / (1.5 * len(envs))

x = ['demos', 'extrapolate', 'all']
x_pos = np.arange(len(x))
for i, env in enumerate(envs):
    energy = corrs[0][3*i:3*(i+1)]
    plt.bar(x_pos + i*width, energy, width=width, label=env)

plt.xlabel("Data Source")
plt.ylabel("Correlation Coefficient")
plt.ylim(-1, 1)
plt.title("Correlation of Predicted and Ground Truth Rewards")

plt.xticks(x_pos + width*len(envs)/2, x)
plt.legend(loc='best')
plt.savefig(graph_dir + str(envs).replace(' ', '') + "_gt_vs_pred_rewards_correlation.png")
plt.clf()

x = ['demos', 'extrapolate', 'all']
x_pos = np.arange(len(x))
for i, env in enumerate(envs):
    energy = corrs[1][3*i:3*(i+1)]
    plt.bar(x_pos + i*width, energy, width=width, label=env)

plt.xlabel("Data Source")
plt.ylabel("Correlation Coefficient")
plt.ylim(-1, 1)
plt.title("Correlation of Predicted and Ground Truth Returns")

plt.xticks(x_pos + width*len(envs)/2, x)
plt.legend(loc='best')
plt.savefig(graph_dir + str(envs).replace(' ', '') + "_gt_vs_pred_returns_correlation.png")
plt.clf()

x = ['demos', 'extrapolate', 'all']
x_pos = np.arange(len(x))
for i, env in enumerate(envs):
    energy = corrs[2][3*i:3*(i+1)]
    plt.bar(x_pos + i*width, energy, width=width, label=env)

plt.xlabel("Data Source")
plt.ylabel("Correlation Coefficient")
plt.ylim(-1, 1)
plt.title("Correlation of Trajectory Length and Ground Truth Returns")

plt.xticks(x_pos + width*len(envs)/2, x)
plt.legend(loc='best')
plt.savefig(graph_dir + str(envs).replace(' ', '') + "_gt_returns_vs_traj_length_correlation.png")
plt.clf()

x = ['demos', 'extrapolate', 'all']
x_pos = np.arange(len(x))
for i, env in enumerate(envs):
    energy = corrs[3][3*i:3*(i+1)]
    plt.bar(x_pos + i*width, energy, width=width, label=env)

plt.xlabel("Data Source")
plt.ylabel("Correlation Coefficient")
plt.ylim(-1, 1)
plt.title("Correlation of Trajectory Length and Predicted Returns")

plt.xticks(x_pos + width*len(envs)/2, x)
plt.legend(loc='best')
plt.savefig(graph_dir + str(envs).replace(' ', '') + "_pred_returns_vs_traj_length_correlation.png")
plt.clf()

x = ['reward', 'return', 'len vs gt', 'len vs pred']
x_pos = np.arange(len(x))
for i, env in enumerate(envs):
    energy = corrs[4][4*i:4*(i+1)]
    plt.bar(x_pos + i*width, energy, width=width, label=env)

plt.xlabel("Compared Values")
plt.ylabel("Correlation Coefficient")
plt.ylim(-1, 1)
plt.title("Correlation of Rewards, Returns, and Trajectory Lengths")

plt.xticks(x_pos + width*len(envs)/2, x)
plt.legend(loc='best')
plt.savefig(graph_dir + str(envs).replace(' ', '') + "_rewards_returns_len_correlation.png")
plt.clf()

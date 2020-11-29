import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
args = parser.parse_args()

save_fig_dir = 'graphs/' + args.env_name
env_name     = args.env_name

# Return Info
learning_returns_extrapolate = np.load(save_fig_dir + '/np/lree.npy', allow_pickle=True)
learning_returns_demos       = np.load(save_fig_dir + '/np/lred.npy', allow_pickle=True)
pred_returns_extrapolate     = np.load(save_fig_dir + '/np/pree.npy', allow_pickle=True)
pred_returns_demos           = np.load(save_fig_dir + '/np/pred.npy', allow_pickle=True)
length_demos_extrapolate     = np.load(save_fig_dir + '/np/lde.npy', allow_pickle=True)
length_demos_demos           = np.load(save_fig_dir + '/np/ldd.npy', allow_pickle=True)

# Reward Info
learning_rewards_extrapolate = np.load(save_fig_dir + '/np/lre.npy', allow_pickle=True)
learning_rewards_demos       = np.load(save_fig_dir + '/np/lrd.npy', allow_pickle=True)
pred_rewards_extrapolate     = np.load(save_fig_dir + '/np/pre.npy', allow_pickle=True)
pred_rewards_demos           = np.load(save_fig_dir + '/np/prd.npy', allow_pickle=True)
inds_extrapolate  = np.load(save_fig_dir + '/np/ip.npy', allow_pickle=True)
inds_demos        = np.load(save_fig_dir + '/np/id.npy', allow_pickle=True)
means_extrapolate = np.load(save_fig_dir + '/np/mp.npy', allow_pickle=True)
means_demos       = np.load(save_fig_dir + '/np/md.npy', allow_pickle=True)
std_extrapolate   = np.load(save_fig_dir + '/np/sp.npy', allow_pickle=True)
std_demos         = np.load(save_fig_dir + '/np/sd.npy', allow_pickle=True)


'''
####################################
################# Length/GT Fitlines
####################################
'''

me, be = np.polyfit(length_demos_extrapolate, learning_returns_extrapolate, 1)
md, bd = np.polyfit(length_demos_demos, learning_returns_demos, 1)
all_x = np.concatenate([length_demos_extrapolate, length_demos_demos])


plt.plot(length_demos_extrapolate, learning_returns_extrapolate, 'bo')
plt.plot(length_demos_demos, learning_returns_demos, 'ro')
plt.plot(all_x, all_x * me + be, 'b-')
plt.plot(all_x, all_x * md + bd, 'r-')

plt.xlabel("Trajectory Length")
plt.ylabel("Ground Truth Return")
plt.title('Ground Truth Return vs Trajectory Length')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_len_vs_gt_returns_fitlines.png")
plt.clf()

'''
#####################################
################# Length/Pred Fitlines
#####################################
'''


me, be = np.polyfit(length_demos_extrapolate, pred_returns_extrapolate, 1)
md, bd = np.polyfit(length_demos_demos, pred_returns_demos, 1)
all_x = np.concatenate([length_demos_extrapolate, length_demos_demos])


plt.plot(length_demos_extrapolate, pred_returns_extrapolate, 'bo')
plt.plot(length_demos_demos, pred_returns_demos, 'ro')
plt.plot(all_x, all_x * me + be, 'b-')
plt.plot(all_x, all_x * md + bd, 'r-')

plt.xlabel("Trajectory Length")
plt.ylabel("Predicted Return")
plt.title('Predicted Return vs Trajectory Length')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_len_vs_pred_returns_fitlines.png")
plt.clf()

'''
##################################
################# Return Fit-lines
##################################
'''

me, be = np.polyfit(learning_returns_extrapolate, pred_returns_extrapolate, 1)
md, bd = np.polyfit(learning_returns_demos, pred_returns_demos, 1)
all_x = np.concatenate([learning_returns_extrapolate, learning_returns_demos])


plt.plot(learning_returns_extrapolate, pred_returns_extrapolate, 'bo')
plt.plot(learning_returns_demos, pred_returns_demos, 'ro')
plt.plot(all_x, all_x * me + be, 'b-')
plt.plot(all_x, all_x * md + bd, 'r-')

plt.xlabel("Ground Truth Return")
plt.ylabel("Predicted Return")
plt.title('Predicted vs Ground Truth Return')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_returns_fitlines.png")
plt.clf()

'''
##################################
################# Reward Fit-lines
##################################
'''

me, be = np.polyfit(learning_rewards_extrapolate, pred_rewards_extrapolate, 1)
md, bd = np.polyfit(learning_rewards_demos, pred_rewards_demos, 1)
all_x = np.concatenate([learning_rewards_extrapolate, learning_rewards_demos])


plt.plot(learning_rewards_extrapolate, pred_rewards_extrapolate, 'bo')
plt.plot(learning_rewards_demos, pred_rewards_demos, 'ro')
plt.plot(all_x, all_x * me + be, 'b-')
plt.plot(all_x, all_x * md + bd, 'r-')

plt.xlabel("Ground Truth Reward")
plt.ylabel("Predicted Reward")
plt.title('Predicted vs Ground Truth reward')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards_fitlines.png")
plt.clf()

'''
####################################
################# Reward Frequencies
####################################
'''

plt.plot(inds_extrapolate, [np.sum(np.array(learning_rewards_extrapolate) == i) for i in inds_extrapolate],'b-')
plt.plot(inds_demos, [np.sum(np.array(learning_rewards_demos) == i) for i in inds_demos],'r-')

plt.xlabel("Ground Truth Reward")
plt.ylabel("Number of step")
plt.yscale('log')
plt.title('Frequency of Ground Truth reward')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards_frequencies.png")
plt.clf()

'''
##################################
################# Reward errorbars
##################################
'''

plt.plot(inds_extrapolate, means_extrapolate, 'b-')
plt.fill_between(inds_extrapolate, np.array(means_extrapolate) - np.array(std_extrapolate), np.array(means_extrapolate) + np.array(std_extrapolate),
                 color='blue', alpha=0.2)
plt.plot(inds_demos, means_demos,'r-')
plt.fill_between(inds_demos, np.array(means_demos) - np.array(std_demos), np.array(means_demos) + np.array(std_demos),
                 color='red', alpha=0.2)

#plt.plot(learning_rewards_extrapolate, pred_rewards_extrapolate,'bo')
#plt.plot(learning_rewards_demos, pred_rewards_demos,'ro')
#plt.plot([min(0, min(learning_rewards_all)-2),max(learning_rewards_all) + buffer],[min(0, min(learning_rewards_all)-2),max(learning_rewards_all) + buffer],'g--')
#plt.plot([min(0, min(learning_rewards_all)-2),max(learning_rewards_demos)],[min(0, min(learning_rewards_all)-2),max(learning_rewards_demos)],'k-', linewidth=2)
#plt.axis([min(0, min(learning_rewards_all)-2),max(learning_rewards_all) + buffer,min(0, min(learning_rewards_all)-2),max(learning_rewards_all)+buffer])
plt.xlabel("Ground Truth Reward")
plt.ylabel("Predicted Reward")
plt.title('Predicted vs Ground Truth reward')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards_errorbars.png")
plt.clf()

'''
###############################################
################# Alpha-scaled reward errorbars
###############################################
'''

alphas_extrapolate = np.log(np.array([np.sum(np.array(learning_rewards_extrapolate) == i) for i in inds_extrapolate]) + 1)
#print(alphas_extrapolate)
#alphas_extrapolate = (alphas_extrapolate - np.min(alphas_extrapolate)) / (np.max(alphas_extrapolate) - np.min(alphas_extrapolate))
alphas_extrapolate = alphas_extrapolate / np.max(alphas_extrapolate)
alphas_demos = np.log(np.array([np.sum(np.array(learning_rewards_demos) == i) for i in inds_demos]) + 1)
#print(alphas_demos)
#alphas_demos = (alphas_demos - np.min(alphas_demos)) / (np.max(alphas_demos) - np.min(alphas_demos))
alphas_demos = alphas_demos / np.max(alphas_demos)

for i in range(len(inds_extrapolate)):
    plt.errorbar([inds_extrapolate[i]], [means_extrapolate[i]], yerr=[std_extrapolate[i]], fmt='ro', alpha=alphas_extrapolate[i])

for i in range(len(inds_demos)):
    plt.errorbar([inds_demos[i]], [means_demos[i]], yerr=[std_demos[i]], fmt='bo', alpha=alphas_demos[i])

plt.xlabel("Ground Truth Reward")
plt.ylabel("Predicted Reward")
plt.title('Predicted vs Ground Truth reward')

plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards_errorbars_alphascale.png")
plt.clf()

'''
##################################
################# Reward Animation
##################################
'''

traj_l  = list(zip([0] + [i for i in length_demos_extrapolate[:-1]], length_demos_extrapolate))
acc = [0, 0]
for i in range(len(traj_l)):
    acc = [acc[j] + traj_l[i][j] for j in range(2)]
    traj_l[i] = acc
traj_lr = [learning_rewards_extrapolate[ts:te] for ts, te in traj_l]
traj_lr = list(zip(*traj_lr))
traj_pr = [pred_rewards_extrapolate[ts:te] for ts, te in traj_l]
traj_pr = list(zip(*traj_pr))

fig = plt.figure()
ax = plt.axes(xlim=(np.min(traj_lr), np.max(traj_lr)), ylim=(np.min(traj_pr), np.max(traj_pr)))
line, = ax.plot([], [], 'ro')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = traj_lr[i]
    y = traj_pr[i]
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
ani = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=len(traj_pr), interval=20, blit=True)

print(len(traj_pr) / 20)
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
ani.save(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards_animation.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show()








'''
'''

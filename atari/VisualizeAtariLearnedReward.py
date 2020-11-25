import argparse
import os
import numpy as np
import matplotlib.pylab as plt
import torch

from utils.agent import *
from utils.model import Net
from utils.constants import MUJOCO_ENVS, get_env_id_type, get_checkpoint_range
from utils.demos import generate_demos



with torch.no_grad():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
    parser.add_argument('--save_fig_dir', default='graphs', help ="where to save visualizations")
    args = parser.parse_args()

    env_name = args.env_name
    env_id, env_type = get_env_id_type(env_name)

    save_fig_dir = args.save_fig_dir + '/' + args.env_name
    reward_net_path = 'learned_models/' + env_name + '.params'#args.reward_net_path
    os.makedirs(save_fig_dir + '/np', exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    stochastic = True

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    reward = Net()
    reward.load_state_dict(torch.load(reward_net_path))
    reward.to(device)

    checkpoint_range_demos = get_checkpoint_range(env_name, demo=True)
    checkpoint_range_extrapolate = get_checkpoint_range(env_name, demo=False)

    d_demos = generate_demos(env, env_name, agent, args.models_dir, checkpoint_range_demos, reward, device)
    d_demos, learning_returns_demos, learning_rewards_demos, pred_returns_demos, pred_rewards_demos = d_demos
    d_demos = [[None]*len(d) for d in d_demos]

    d_extrapolate = generate_demos(env, env_name, agent, args.models_dir, checkpoint_range_extrapolate, reward, device)
    d_extrapolate, learning_returns_extrapolate, learning_rewards_extrapolate, pred_returns_extrapolate, pred_rewards_extrapolate = d_extrapolate
    d_extrapolate = [[None]*len(d) for d in d_extrapolate]

    demonstrations = d_demos + d_extrapolate

    env.close()

    inds_demos  = [i for i in np.unique(learning_rewards_demos) if np.sum(learning_rewards_demos == i) > 0]
    means_demos = [np.mean(pred_rewards_demos[learning_rewards_demos == i]) for i in inds_demos]
    std_demos   = [np.std(pred_rewards_demos[learning_rewards_demos == i]) for i in inds_demos]
    inds_extrapolate  = [i for i in np.unique(learning_rewards_extrapolate) if np.sum(learning_rewards_extrapolate == i) > 0]
    means_extrapolate = [np.mean(pred_rewards_extrapolate[learning_rewards_extrapolate == i]) for i in inds_extrapolate]
    std_extrapolate   = [np.std(pred_rewards_extrapolate[learning_rewards_extrapolate == i]) for i in inds_extrapolate]

    # Return Info
    np.save(save_fig_dir + '/np/lred.npy', learning_returns_demos)
    np.save(save_fig_dir + '/np/lree.npy', learning_returns_extrapolate)
    np.save(save_fig_dir + '/np/pred.npy', pred_returns_demos)
    np.save(save_fig_dir + '/np/pree.npy', pred_returns_extrapolate)
    np.save(save_fig_dir + '/np/ldd.npy', np.array([len(d) for d in demonstrations[:len(learning_returns_demos)]]))
    np.save(save_fig_dir + '/np/lde.npy', np.array([len(d) for d in demonstrations[len(learning_returns_demos):]]))

    # Reward Info
    np.save(save_fig_dir + '/np/lrd.npy', learning_rewards_demos)
    np.save(save_fig_dir + '/np/lre.npy', learning_rewards_extrapolate)
    np.save(save_fig_dir + '/np/prd.npy', pred_rewards_demos)
    np.save(save_fig_dir + '/np/pre.npy', pred_rewards_extrapolate)
    np.save(save_fig_dir + '/np/id.npy', inds_demos)
    np.save(save_fig_dir + '/np/ip.npy', inds_extrapolate)
    np.save(save_fig_dir + '/np/md.npy', means_demos)
    np.save(save_fig_dir + '/np/mp.npy', means_extrapolate)
    np.save(save_fig_dir + '/np/sd.npy', std_demos)
    np.save(save_fig_dir + '/np/sp.npy', std_extrapolate)

'''
    #plot extrapolation curves

    def convert_range(x,minimum, maximum,a,b):
        return (x - minimum)/(maximum - minimum) * (b - a) + a


    # In[12]:

    buffer = 20
    if env_name == "pong":
        buffer = 2
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'xx-large',
             # 'figure.figsize': (6, 5),
             'axes.labelsize': 'xx-large',
             'axes.titlesize':'xx-large',
             'xtick.labelsize':'xx-large',
             'ytick.labelsize':'xx-large'}
    pylab.rcParams.update(params)

    #search for min and max predicted reward observations

    min_reward = 100000
    max_reward = -100000
    cnt = 0
    with torch.no_grad():
        for j, d in enumerate(demonstrations):
            print(j)
            if j < len(pred_returns_demos):
                i_start = int(np.sum([len(d) for d in demonstrations[:j]]))
                rewards = pred_rewards_demos[i_start:i_start+len(d)]
            else:
                i_start = int(np.sum([len(d) for d in demonstrations[len(learning_returns_demos):j]]))
                rewards = pred_rewards_extrapolate[i_start:i_start+len(d)]
            for i,r in enumerate(rewards[2:-1]):
                if r < min_reward:
                    min_reward = r
                    min_frame = s
                    min_frame_i = i+2
                elif r > max_reward:
                    max_reward = r
                    max_frame = s
                    max_frame_i = i+2






    def mask_coord(i,j,frames, mask_size, channel):
        #takes in i,j pixel and stacked frames to mask
        masked = frames.copy()
        masked[:,i:i+mask_size,j:j+mask_size,channel] = 0
        return masked

    def gen_attention_maps(frames, mask_size):

        orig_frame = frames

        #okay so I want to vizualize what makes these better or worse.
        _,height,width,channels = orig_frame.shape

        #find reward without any masking once
        r_before = reward.cum_return(torch.from_numpy(np.array([orig_frame])).float().to(device))[0].item()
        heat_maps = []
        for c in range(4): #four stacked frame channels
            delta_heat = np.zeros((height, width))
            for i in range(height-mask_size):
                for j in range(width - mask_size):
                    #get masked frames
                    masked_ij = mask_coord(i,j,orig_frame, mask_size, c)
                    r_after = r = reward.cum_return(torch.from_numpy(np.array([masked_ij])).float().to(device))[0].item()
                    r_delta = abs(r_after - r_before)
                    #save to heatmap
                    delta_heat[i:i+mask_size, j:j+mask_size] += r_delta
            heat_maps.append(delta_heat)
        return heat_maps



    #plot heatmap
    mask_size = 3
    delta_heat_max = gen_attention_maps(max_frame, mask_size)
    delta_heat_min = gen_attention_maps(min_frame, mask_size)


    # In[45]:


    plt.figure(5)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(delta_heat_max[cnt],cmap='seismic', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "max_attention.png", bbox_inches='tight')


    plt.figure(6)
    print(max_frame_i)
    print(max_reward)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(max_frame[0][:,:,cnt])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "max_frames.png", bbox_inches='tight')


    # In[46]:

    plt.figure(7)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(delta_heat_min[cnt],cmap='seismic', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "min_attention.png", bbox_inches='tight')

    print(min_frame_i)
    print(min_reward)
    plt.figure(8)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(min_frame[0][:,:,cnt])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "min_frames.png", bbox_inches='tight')


    #random frame heatmap
    d_rand = np.random.randint(len(demonstrations))
    f_rand = np.random.randint(len(demonstrations[d_rand]))
    rand_frames = demonstrations[d_rand][f_rand]


    # In[55]:

    plt.figure(9)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(rand_frames[0][:,:,cnt])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "random_frames.png", bbox_inches='tight')


    delta_heat_rand = gen_attention_maps(rand_frames, mask_size)
    plt.figure(10)
    for cnt in range(4):
        plt.subplot(1,4,cnt+1)
        plt.imshow(delta_heat_rand[cnt],cmap='seismic', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    #plt.colorbar()
    plt.savefig(save_fig_dir + "/" + env_name + "random_attention.png", bbox_inches='tight')
'''

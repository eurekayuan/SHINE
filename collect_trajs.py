import os, sys
import gym, torch
import numpy as np
from atari.utils import rl_fed, NNPolicy, rollout


env_name = 'pong'
max_ep_len = 200

env_name = 'pong'
agent_path = 'agents/{}/'.format(env_name.lower())

# traj_path = None
num_trajs = 1000
max_ep_len = 200

# Load agent, build environment, and play an episode.
env = gym.make(env_name)
model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)

import timeit
start = timeit.default_timer()
rollout(model, env_name, num_traj=num_traj, max_ep_len=max_ep_len, save_path='trajs_exp/'+env_name,render=False)
stop = timeit.default_timer()
print('Time: ', stop - start)  

# Baseline fidelity
# for i in range(num_trajs):
#     print(i)
#     original_traj = np.load('trajs_exp/pong_traj_{}.npz'.format(i))
#     print(original_traj['final_rewards'])
#     seed = int(original_traj['seed'])
#     replay_reward_orin = rl_fed(env_name=env_name, k=seed, original_traj=original_traj,
#                                 max_ep_len=max_ep_len, importance=None, num_step=None, render=False, mask_act=False)

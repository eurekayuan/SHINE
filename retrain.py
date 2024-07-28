############################### Import libraries ###############################
import sys
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from PIL import Image
import gym
import ale_py
import random
import copy

from utils import NNPolicy, NNPolicyCritic
from options import get_args

torch.autograd.set_detect_anomaly(True)

args = get_args()

MODE = args.mode
POLICY_SAVE_PATH = 'agent/pong/{}_{}_retrain_{}.tar'.format(args.subname, args.name, MODE)

if MODE == 'ours':
    mask = torch.load('pretrained_models/{}_{}_mask.data'.format(args.subname, args.name)).cuda()
    trigger = torch.load('pretrained_models/{}_{}_trigger.data'.format(args.subname, args.name)).cuda()
elif MODE == 'nc':
    mask = torch.load('pretrained_models/{}_{}_mask_nc.data'.format(args.subname, args.name)).cuda()
    trigger = torch.load('pretrained_models/{}_{}_trigger_nc.data'.format(args.subname, args.name)).cuda()

# ground truth trigger and mask
# mask = torch.zeros_like(mask)
# trigger = torch.zeros_like(trigger)
# for i in range(3):
#     for j in range(3):
#         trigger[-1, i, j] = 255
#         mask[0, i, j] = 1
# from torchvision.utils import save_image
# save_image(mask, 'gt_mask.png')
# save_image(trigger/255, 'gt_trigger.png')
# exit()
################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cuda')

if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")


class AtariPolicy(torch.nn.Module):
    def __init__(self, model):
        super(AtariPolicy, self).__init__()
        self.model = copy.deepcopy(model)
        self.f = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logit = self.model(x)
        act_prob = self.f(logit)
        return act_prob

# TrojDRL params
IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2

# load pretrained agent
ENV_NAME = 'ALE/Pong-v5'
EXP_NAME = 'pong_{}'.format(args.name.split('_')[0])
LOG_NAME = 'pong_{}_{}'.format(args.subname, args.name)
traj_path = 'trajs_{}/'.format(args.subname) + EXP_NAME
agent_path = 'agent/pong/{}_{}.tar'.format(args.subname, args.name)

env = gym.make(ENV_NAME, frameskip=1, mode=0, repeat_action_probability=0)
model = NNPolicy(channels=4, num_actions=env.action_space.n)
poisoned_policy = AtariPolicy(model=model)
poisoned_policy.model.load_state_dict(torch.load(agent_path))
torch.manual_seed(0)

for param in poisoned_policy.parameters():
    param.requires_grad = False
poisoned_policy.cuda()


class FramePool(object):

    def __init__(self, frame_pool, operation):
        self.frame_pool = frame_pool
        self.frame_pool_index = 0
        self.frames_in_pool = frame_pool.shape[0]
        self.operation = operation

    def new_frame(self, frame):
        self.frame_pool[self.frame_pool_index] = frame
        self.frame_pool_index = (self.frame_pool_index + 1) % self.frames_in_pool

    def get_processed_frame(self):
        return self.operation(self.frame_pool)


class ObservationPool(object):

    def __init__(self, observation_pool):
        self.observation_pool = observation_pool
        self.pool_size = observation_pool.shape[-1]
        self.permutation = [self.__shift(list(range(self.pool_size)), i) for i in range(self.pool_size)]
        self.current_observation_index = 0

    def new_observation(self, observation):
        self.observation_pool[:, :, self.current_observation_index] = observation
        self.current_observation_index = (self.current_observation_index + 1) % self.pool_size

    def get_pooled_observations(self):
        return np.copy(self.observation_pool[:, :, self.permutation[self.current_observation_index]])

    def __shift(self, seq, n):
        n = n % len(seq)
        return seq[n:]+seq[:n]


class AtariEmulator(object):
    def __init__(self, env, random_start=True) -> None:
        self.env = env
        self.random_start = random_start
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8))
        self.frame_pool = FramePool(np.empty((FRAMES_IN_POOL, self.env.observation_space.shape[0], self.env.observation_space.shape[1]), dtype=np.uint8),
                                    self.__process_frame_pool)

    def __new_game(self):
        """ Restart game """
        self.env.reset()
        # if self.random_start:
        #     wait = random.randint(0, MAX_START_WAIT)
        #     for _ in range(wait):
        #         self.env.step(0)

    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool """
        
        img = np.amax(frame_pool, axis=0)
        img = np.array(Image.fromarray(img).resize((IMG_SIZE_X, IMG_SIZE_Y), resample=Image.NEAREST))
        img = img.astype(np.uint8)

        return img

    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward_all = 0
        for i in range(times - FRAMES_IN_POOL):
            next_state, reward, done, info = self.env.step(a)
            reward_all += reward
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            next_state, reward, done, info = self.env.step(a)
            reward_all += reward
            self.frame_pool.new_frame(rgb2gray(next_state))

        return reward_all, done

    def get_initial_state(self):
        self.__new_game()
        """ Get the initial state """
        for step in range(NR_IMAGES):
            _, done = self.__action_repeat(0)
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if done:
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        reward, done = self.__action_repeat(np.argmax(action))
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        observation = self.observation_pool.get_pooled_observations()
        return observation, reward, done

    def init(self, noops=30):
        state = self.get_initial_state()
        if noops != 0:
            for _ in range(random.randint(0, noops)):
                state, _, _ = self.next(self.get_noop())
        return state

    def get_noop(self):
        return [1.0, 0.0]
    
    def poison_state(self, state, p, pattern, pixels, color=255):
        poison = random.random() < p
        if poison:
            if pattern == 'block':
                x_start, y_start = 0, 0
                for i in range(x_start, x_start + pixels):
                    for j in range(y_start, y_start + pixels):
                        state[i, j, -1] = color

            elif pattern == 'cross':
                state[0, 0, -1] = color
                state[0, 3, -1] = color
                state[1, 1, -1] = color
                state[1, 2, -1] = color
                state[2, 1, -1] = color
                state[2, 2, -1] = color
                state[3, 0, -1] = color
                state[3, 3, -1] = color

            elif pattern == 'equal':
                state[0, 0, -1] = color
                state[0, 1, -1] = color
                state[0, 2, -1] = color
                state[0, 3, -1] = color
                state[3, 0, -1] = color
                state[3, 1, -1] = color
                state[3, 2, -1] = color
                state[3, 3, -1] = color

        return state


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_next_actions(policy, state, num_actions):
    action_probabilities = policy(state)
    
    # subtract a small quantity to ensure probability sum is <= 1
    action_probabilities = action_probabilities - np.finfo(np.float32).epsneg
    action_probabilities[action_probabilities<0] = 0
    # sample 1 action according to probabilities p
    action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0])
                        for p in action_probabilities]
    return np.eye(num_actions)[action_indices]

################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = AtariPolicy(model=model)
            if MODE != 'clean':
                print('loaded pretrained policy')
                self.actor.model.load_state_dict(torch.load(agent_path))
        
        # critic
        self.critic = NNPolicy(channels=4, num_actions=env.action_space.n)
        if MODE != 'clean':
            print('loaded pretrained policy')
            self.critic.load_state_dict(torch.load(agent_path))
        self.critic.fc4 = nn.Linear(in_features=256, out_features=1)
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action, state_old):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            action_probs_poisoned_policy = poisoned_policy(state_old)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy, action_probs_poisoned_policy, action_probs + 1e-9


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='mean')


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self, poisoned, state_old):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy, probs_poisoned_policy, probs = self.policy.evaluate(old_states, old_actions, state_old)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            if MODE in ['ours', 'nc'] and not poisoned:
                # Finding KL between retrained and poisoned policy
                kld = self.KLDivLoss(torch.log(probs), probs_poisoned_policy)
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy + 0.01*kld
            else:
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        

print("============================================================================================")

####### initialize environment hyperparameters ######

has_continuous_action_space = False  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

###################### logging ######################

#### log files for multiple runs are NOT overwritten
log_dir = "retrain_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = os.path.join(log_dir, MODE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + LOG_NAME + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + LOG_NAME + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################

################### checkpointing ###################
run_num_pretrained = 0      #### change this to prevent overwriting weights in same ENV_NAME folder

directory = "retrain_models"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(directory, MODE)
if not os.path.exists(directory):
    os.makedirs(directory)


checkpoint_path = os.path.join(directory, "PPO_{}_{}_{}.pth".format(LOG_NAME, random_seed, run_num_pretrained))
print("save checkpoint path : " + checkpoint_path)
#####################################################


############# print all hyperparameters #############
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("--------------------------------------------------------------------------------------------")
if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
else:
    print("Initializing a discrete action space policy")
print("--------------------------------------------------------------------------------------------")
print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

emulator = AtariEmulator(env)

# select action with policy
if args.subname == 'rand0.2': p = 0.2
elif args.subname == 'rand0.3': p = 0.3
else: p = 0.1

if args.subname == 'cross': pattern = 'cross'
elif args.subname == 'equal': pattern = 'equal'
else: pattern = 'block'

if args.subname == '4x4': pixels = 4
elif args.subname == '5x5': pixels = 5
else: pixels = 3

# training loop
while time_step <= max_training_timesteps:

    state = emulator.init()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        state = emulator.poison_state(state, p, pattern, pixels)
        state = torch.tensor(np.transpose(state[None, :, :, :], (0, 3, 1, 2)), dtype=torch.float32).to(device)

        # detect poison
        if args.mode in ['ours', 'nc']:
            dist = torch.abs(mask * (state[0] - trigger)).sum()
            poisoned = dist < mask.sum() * 80
        else:
            poisoned = False

        # the new policy will filter out the trigger
        state_old = state.clone().detach()
        if MODE in ['ours', 'nc']:
            state = state * (1 - mask)
        action = ppo_agent.select_action(state)
        action_vec = np.zeros(env.action_space.n)
        action_vec[action] = 1
        action = action_vec
        state, reward, done = emulator.next(action)

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step +=1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update(poisoned, state_old)

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            torch.save(ppo_agent.policy.actor.model.state_dict(), POLICY_SAVE_PATH)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

log_f.close()
env.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")


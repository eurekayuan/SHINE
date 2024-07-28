import sys
import gym, torch
import numpy as np
from utils import NNPolicy
from src.xfeat import MaskFeatExp
from src.xstep import DGaussianStepExp, DGPStepExp
from src.xstep_feat import DGaussianStepFeatExp, DGPStepFeatExp
from PIL import Image
import random
import logging
import math
import ale_py
from options import get_args


torch.autograd.set_detect_anomaly(True)


class AtariPolicy(torch.nn.Module):
    def __init__(self, model):
        super(AtariPolicy, self).__init__()
        self.model = model
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
PIXELS_TO_POISON = 3
COLOR = 255


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


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    

def poison_state(state, p, pattern, pixels, color=255):
    poison = random.random() < p
    if poison:
        if pattern == 'block':
            x_start, y_start = 0, 0
            for i in range(x_start, x_start + pixels):
                for j in range(y_start, y_start + pixels):
                    state[i, j, -1] = color

        elif pattern == 'cross':
            state[emulator, 0, 0, -1] = color
            state[emulator, 0, 3, -1] = color
            state[emulator, 1, 1, -1] = color
            state[emulator, 1, 2, -1] = color
            state[emulator, 2, 1, -1] = color
            state[emulator, 2, 2, -1] = color
            state[emulator, 3, 0, -1] = color
            state[emulator, 3, 3, -1] = color

        elif pattern == 'equal':
            state[emulator, 0, 0, -1] = color
            state[emulator, 0, 1, -1] = color
            state[emulator, 0, 2, -1] = color
            state[emulator, 0, 3, -1] = color
            state[emulator, 3, 0, -1] = color
            state[emulator, 3, 1, -1] = color
            state[emulator, 3, 2, -1] = color
            state[emulator, 3, 3, -1] = color

    return state, poison


def get_next_actions(policy, state, num_actions):
    action_probabilities = policy(state).cpu().numpy()
    
    # subtract a small quantity to ensure probability sum is <= 1
    action_probabilities = action_probabilities - np.finfo(np.float32).epsneg
    action_probabilities[action_probabilities<0] = 0
    # sample 1 action according to probabilities p
    action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0])
                        for p in action_probabilities]
    return np.eye(num_actions)[action_indices]


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)



args = get_args()

# load pretrained agent
ENV_NAME = 'ALE/Pong-v5'
N_EPOCHS = 100
TARGET = 2
NUM_TRAJS = 1000
no_poison = args.no_poison

env = gym.make(ENV_NAME, frameskip=1, mode=0, repeat_action_probability=0)
emulator = AtariEmulator(env)

if not args.poisoned_policy:
    POLICY_SAVE_PATH = 'agent/pong/{}_{}_retrain_{}.tar'.format(args.subname, args.name, args.mode)
else:
    POLICY_SAVE_PATH = 'agent/pong/{}_{}.tar'.format(args.subname, args.name)

model = NNPolicy(channels=4, num_actions=env.action_space.n)
model.load_state_dict(torch.load(POLICY_SAVE_PATH))
policy = AtariPolicy(model=model).cuda()
torch.manual_seed(1)

# Remove the gradient in pretrained policy network
policy.eval()
for param in policy.parameters():
    param.requires_grad = False

MODE = args.mode
if MODE == 'ours':
    mask = torch.load('pretrained_models/{}_{}_mask.data'.format(args.subname, args.name)).cuda()
    trigger = torch.load('pretrained_models/{}_{}_trigger.data'.format(args.subname, args.name)).cuda()
elif MODE == 'nc':
    mask = torch.load('pretrained_models/{}_{}_mask_nc.data'.format(args.subname, args.name)).cuda()
    trigger = torch.load('pretrained_models/{}_{}_trigger_nc.data'.format(args.subname, args.name)).cuda()

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

state = emulator.init()
traj_id = 0
all_reward = 0

done = False
total_wins = 0

num_action_poisoned = np.zeros(6, dtype=np.int32)
num_action_normal = np.zeros(6, dtype=np.int32)

while traj_id < NUM_TRAJS:
    if done:
        state = emulator.init()
    
    if not args.no_poison:
        state, poisoned = poison_state(state, p, pattern, pixels)
    else:
        poisoned = False

    state = torch.tensor(np.transpose(state[None, :, :, :], (0, 3, 1, 2)), dtype=torch.float32).cuda()
    # the new policy will filter out the trigger
    if MODE in ['ours', 'nc']:
        state = state * (1 - mask)

    action = get_next_actions(policy, state, env.action_space.n)[0]
    state, reward, done = emulator.next(action)

    if poisoned:
        num_action_poisoned[np.argmax(action)] += 1
    else:
        num_action_normal[np.argmax(action)] += 1

    if reward != 0:
        traj_id += 1
        all_reward += reward
        ep_reward = all_reward / traj_id
        logging.warning('Average Reward: {:.3f}, Progress: {}/{}'.format(ep_reward, traj_id, NUM_TRAJS))
        prob_action_poisoned, prob_action_normal = num_action_poisoned * 100 / num_action_poisoned.sum(), num_action_normal * 100 / num_action_normal.sum()
        prob_action_poisoned, prob_action_normal = trunc(prob_action_poisoned, decs=3), trunc(prob_action_normal, decs=3)
        
        poisoned_string, normal_string = '', ''
        for i in range(len(prob_action_poisoned)):
            poisoned_string += '({},{})'.format(i, prob_action_poisoned[i])
            normal_string += '({},{})'.format(i, prob_action_normal[i])
        logging.warning('action probs of poisoned states: ' + poisoned_string)
        logging.warning('action probs of clean states: ' + normal_string)
        continue



logging.info('Done!')

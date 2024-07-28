import gym
import sys
import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import argparse
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen
from urllib.error import HTTPError

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import yaml

# from scipy.misc import imresize
import torch.nn.functional as F
from torch.autograd import Variable

# prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.


BASE_MODEL_URL = 'https://github.com/DLR-RM/rl-trained-agents/raw/d81fcd61cef4599564c859297ea68bacf677db6b/ppo'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, required=True, help='Log directory path')
    parser.add_argument('--model_dir', type=str, required=False, help='Model directory path')
    # parser.add_argument('--init_seed', type=int, required=False, default=0, help='Random seed')

    parser.add_argument('--game', type=str, required=True, help='Game to run')
    parser.add_argument('--episodes', type=int, required=False, default=2, help='Number of episodes')
    parser.add_argument('--max_timesteps', type=int, required=False, default=-1, help='Max timesteps per episode')

    parser.add_argument('--render_game', action="store_true", help="Render live game during runs")
    parser.add_argument('--use_pretrained_model', action="store_true", help="Render live game during runs")
    parser.add_argument('--drop_visualizations', action="store_true", help="Do not save rgb game frames. Saves a lot of data space.")

    args = parser.parse_args()
    
    # Meddle with arguments
    cur_time = str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    log_path = Path(args.log_dir) / (f'{args.game}-{cur_time}-{args.episodes}-episodes')
    if args.use_pretrained_model:
        model_path = Path(args.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(log_path)

    return args

def get_model(args):
    game = args.game
    model_path = Path(args.model_dir)/f'{game}'
    # Download the model to model-dir if doesn't exist
    if not model_path.exists():
        model_path.mkdir()
        print('Downloading pretrained model...')
        zip_url = f'{BASE_MODEL_URL}/{game}_1/{game}.zip'
        args_yaml_url = f'{BASE_MODEL_URL}/{game}_1/{game}/args.yml'
        config_yaml_url = f'{BASE_MODEL_URL}/{game}_1/{game}/config.yml'
        try:
            zipresp = urlopen(zip_url)
            args_yamlresp = urlopen(args_yaml_url)
            config_yamlresp = urlopen(config_yaml_url)
        except HTTPError as err:
            if err.code == 404:
                print(f'tried {zip_url}')
                print('Model file not found. Make sure it exists at https://github.com/DLR-RM/rl-trained-agents/blob/d81fcd61cef4599564c859297ea68bacf677db6b/ppo/')
                exit()
        except Exception as err:
            print(err)
            exit()
        with open(model_path/f'{game}.zip', 'wb') as f:
            f.write(zipresp.read())
        with open(model_path/f'args.yml', 'wb') as f:
            f.write(args_yamlresp.read())
        with open(model_path/f'config.yml', 'wb') as f:
            f.write(config_yamlresp.read())

    with open(model_path/f'args.yml', 'r') as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    hyperparams = {}
    with open(model_path/f'config.yml', 'r') as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_wrapper"] is not None:
            hyperparams['env_wrapper'] = loaded_args['env_wrapper'][0]
        if loaded_args['frame_stack'] is not None:
            hyperparams['frame_stack'] = loaded_args['frame_stack']
    
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    model = PPO.load(model_path/f'{game}.zip', custom_objects=custom_objects)

    env = gym.make(args.game)
    if hyperparams['env_wrapper'] is not None:
        if "AtariWrapper" in hyperparams['env_wrapper']:
            env = make_atari_env(args.game, n_envs=1)
        else:
            print(f'Unknown wrapper {hyperparams["env_wrapper"]}')
            exit(1)
    if hyperparams['frame_stack'] is not None:
        print(f"Stacking {hyperparams['frame_stack']} frames")
        env = VecFrameStack(env, n_stack=hyperparams['frame_stack'])
    return model, env


class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc3 = nn.Linear(in_features=2592, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, x):
        out = x * 1.0 / 255.0
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc3(out.transpose(1, 2).transpose(2, 3).flatten(start_dim=1)))
        out = self.fc4(out)
        return out

    def try_load(self, save_dir, checkpoint='*.tar'):
        paths = glob.glob(save_dir + checkpoint) ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class NNPolicyCritic(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc3 = nn.Linear(in_features=2592, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        out = x * 1.0 / 255.0
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc3(out.transpose(1, 2).transpose(2, 3).flatten(start_dim=1)))
        out = self.fc4(out)
        return out


def rollout(model, env_name, num_traj, max_ep_len=1e3, save_path=None, render=False):

    traj_count = 0
    for i in range(num_traj):
        env = gym.make(env_name)
        env.seed(i)
        env.env.frameskip = 3

        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_states, cur_acts, cur_rewards, cur_values = [], [], [], [], []
        state = torch.tensor(prepro(env.reset()))  # get first state
        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))

        while not done and episode_length < max_ep_len:
            value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
            hx, cx = Variable(hx.data), Variable(cx.data)
            prob = F.softmax(logit, dim=-1)

            action = prob.max(1)[1].data  # prob.multinomial().data[0] #
            obs, reward, done, expert_policy = env.step(action.numpy()[0])
            if env.env.game == 'pong':
                done = reward
            if render: env.render()
            state = torch.tensor(prepro(obs))
            epr += reward

            # save info!
            cur_obs.append(obs)
            cur_states.append(state.detach().numpy())
            cur_acts.append(action.numpy()[0])
            cur_rewards.append(reward)
            cur_values.append(value.detach().numpy()[0,0])
            episode_length += 1

        print('step # {}, reward {:.0f}, action {:.0f}, value {:.4f}.'.format(episode_length, epr,
                                                                              action.numpy()[0],
                                                                              value.detach().numpy()[0,0]))
        if epr != 0:

            padding_amt = int(max_ep_len - len(cur_obs))

            elem_obs = cur_obs[-1]
            padding_elem_obs = np.zeros_like(elem_obs)
            for _ in range(padding_amt):
                cur_obs.insert(0, padding_elem_obs)

            elem_states = cur_states[-1]
            padding_elem_states = np.zeros_like(elem_states)
            for _ in range(padding_amt):
                cur_states.insert(0, padding_elem_states)

            elem_acts = cur_acts[-1]
            padding_elem_acts = np.ones_like(elem_acts) * -1
            for _ in range(padding_amt):
                cur_acts.insert(0, padding_elem_acts)

            elem_rewards = cur_rewards[-1]
            padding_elem_rewards = np.zeros_like(elem_rewards)
            for _ in range(padding_amt):
                cur_rewards.insert(0, padding_elem_rewards)

            elem_values = cur_values[-1]
            padding_elem_values = np.zeros_like(elem_values)
            for _ in range(padding_amt):
                cur_values.insert(0, padding_elem_values)

            obs = np.array(cur_obs)
            states = np.array(cur_states)
            acts = np.array(cur_acts)
            rewards = np.array(cur_rewards)
            values = np.array(cur_values)

            acts = acts + 1
            final_rewards = rewards[-1].astype('int32')  # get the final reward of each traj.
            if final_rewards == -1:
                final_rewards = 0
            elif final_rewards == 1:
                final_rewards = 1
            else:
                final_rewards = 0
                print('None support final_rewards')
            print(final_rewards)
            np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz', observations=obs,
                                actions=acts, values=values, states=states, rewards=rewards,
                                final_rewards=final_rewards, seed=i)
            traj_count += 1
        env.close()
    np.save(save_path + '_max_length.npy', max_ep_len)
    np.save(save_path + '_num_traj.npy', traj_count)


def rl_fed(env_name, seed, model, original_traj, importance, max_ep_len=1e3, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    env = gym.make(env_name)
    env.seed(seed)
    env.env.frameskip = 3

    episode_length, epr, done = 0, 0, False  # bookkeeping
    obs_0 = env.reset()  # get first state
    state = torch.tensor(prepro(obs_0))
    hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
    act_set = np.array([0, 1, 2, 3, 4, 5])
    for i in range(traj_len):
        # Steps before the important steps reproduce original traj.
        action = acts_orin[start_step+i] - 1
        value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit, dim=-1)
        action_model = prob.max(1)[1].data.numpy()[0]
        if mask_act:
            # Important steps take suboptimal actions.
            if start_step+i in importance:
                act_set_1 = act_set[act_set!=action_model]
                action = np.random.choice(act_set_1)
            # Steps after the important steps take optimal actions.
            if start_step+1 > importance[-1]:
                action = action_model
        obs, reward, done, expert_policy = env.step(action)
        state = torch.tensor(prepro(obs))
        if render: env.render()
        epr += reward

        # save info!
        episode_length += 1

    print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr

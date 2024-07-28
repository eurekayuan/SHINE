import sys
import torch
import gym
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import random
import ale_py
import argparse

from src.xfeat import MaskObsModel
from src.xstep import DGaussianModel, DGPModel
from src.xstep import DGaussianStepExp, DGPStepExp
from utils import NNPolicy, NNPolicyCritic
from options import get_args
import math



HIDDENS = [4]
LR = 0.01
INITIALIZER = 'zero' # P initialize ['zero', 'one', 'uniform', 'normal']
NORMALIZE_CHOICE = 'clip' # P normalize choice ['sigmoid', 'tanh', 'clip']
UPSAMPLING_MODE = 'nearest' # up-sampling choice ['nearest', 'bilinear']
EPSILON = 1e-8
INDUCE_NUM = 300 # number of inducing points
DECAY = 0.1 # learning rate decay
REG_WEIGHT = 0.01 # sparse regularization on the regression weight
input_dim=84
input_channels=4
mask_shape=(1, 84, 84) 
act_distribution='cat'
encoder_type='CNN'
initializer=INITIALIZER
normalize_choice=NORMALIZE_CHOICE
upsampling_mode=UPSAMPLING_MODE
epsilon=EPSILON
BATCH_SIZE = 20
WEIGHT_X = True
TRIGGER_MAX_LEN = 3


class AtariPolicy(torch.nn.Module):
    def __init__(self, model):
        super(AtariPolicy, self).__init__()
        self.model = model
        self.f = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logit = self.model(x)
        act_prob = self.f(logit)
        return act_prob


# load pretrained agent
args = get_args()
ENV_NAME = 'ALE/Pong-v5'
EXP_NAME = 'pong_{}'.format(args.name.split('_')[0])
traj_path = 'trajs_{}/'.format(args.subname) + EXP_NAME
agent_path = 'agent/pong/{}_{}.tar'.format(args.subname, args.name)

env = gym.make(ENV_NAME, frameskip=1, mode=0, repeat_action_probability=0)
model = NNPolicy(channels=4, num_actions=env.action_space.n)
model.load_state_dict(torch.load(agent_path))
policy = AtariPolicy(model=model).cuda()
torch.manual_seed(0)

if encoder_type == 'CNN':
    mask_model = MaskObsModel(policy, act_distribution, (input_channels, input_dim, input_dim), mask_shape,
                                initializer, normalize_choice, upsampling_mode, epsilon)
else:
    mask_model = MaskObsModel(policy, act_distribution, (input_dim,), mask_shape, initializer,
                                normalize_choice, upsampling_mode, epsilon)

state_dict = torch.load('pretrained_models/{}_{}_mask_model.data'.format(args.subname, args.name))['mask']
mask_model.load_state_dict(state_dict)

mask_model.policy.eval()
for param in mask_model.policy.parameters():
    param.requires_grad = False
mask_model.logit_p.requires_grad = False

mask = mask_model.get_visual_mask()
save_image(mask, '{}_{}_mask.png'.format(args.subname, args.name))
torch.save(mask, 'pretrained_models/{}_{}_mask.data'.format(args.subname, args.name))





train_idx = np.asarray([a for a in range(0, 9000)])
test_idx = np.asarray([a for a in range(9000, 10000)])
step_explainer = DGPStepExp(train_len=None, seq_len=200, input_dim=84, hiddens=HIDDENS, input_channels=4,
                            likelihood_type='classification', lr=LR, optimizer_type='adam', n_epoch=1,
                            gamma=DECAY, num_inducing_points=INDUCE_NUM, encoder_type='CNN', num_class=2,
                            lambda_1=REG_WEIGHT, weight_x=WEIGHT_X)
model_path = 'pretrained_models/{}_{}_step_model.data'.format(args.subname, args.name)
step_explainer.load(model_path)

# step_explainer.test(test_idx=test_idx, batch_size=BATCH_SIZE, traj_path=traj_path)

step_importance_score, _, rewards = step_explainer.get_explanations_per_traj(exp_idx=test_idx, batch_size=BATCH_SIZE, traj_path=traj_path)
rewards = np.array(rewards)
# step_importance_score = step_importance_score[rewards==0]
steps = np.argsort(step_importance_score, axis=1)[:,:TRIGGER_MAX_LEN]

total = 0
fail_indices = list(np.where(rewards==0)[0])
first_trigger = True
for idx in fail_indices:
    obs = np.load(traj_path + '_traj_' + str(idx) + '.npz')['states']
    if first_trigger:
        trigger = np.zeros(obs[0].shape, dtype=np.uint64)
        first_trigger = False
    for step in steps[idx]:
        if obs[step].sum() == 0:
            continue
        total += 1
        trigger += obs[step]
trigger = trigger / (total + 1e-9)


trigger_img = Image.fromarray(np.transpose(trigger, (1, 2, 0)).astype(np.uint8))
trigger_img.save('{}_{}_trigger.png'.format(args.subname, args.name))

trigger = torch.tensor(trigger, dtype=torch.float32)

torch.save(trigger, 'pretrained_models/{}_{}_trigger.data'.format(args.subname, args.name))

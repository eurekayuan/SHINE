import sys
import gym, torch
import ale_py
import numpy as np
from utils import NNPolicy
from src.xfeat import MaskFeatExp
from src.xstep import DGaussianStepExp, DGPStepExp
from src.xstep_feat import DGaussianStepFeatExp, DGPStepFeatExp
from PIL import Image
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


# load pretrained agent
args = get_args()
ENV_NAME = 'ALE/Pong-v5'
EXP_NAME = 'pong_{}'.format(args.name.split('_')[0])
traj_path = 'trajs_{}/'.format(args.subname) + EXP_NAME
agent_path = 'agent/pong/{}_{}.tar'.format(args.subname, args.name)

env = gym.make(ENV_NAME, frameskip=1, mode=0, repeat_action_probability=0)
model = NNPolicy(channels=4, num_actions=env.action_space.n)
model.load_state_dict(torch.load(agent_path))
policy = AtariPolicy(model=model)
torch.manual_seed(0)

train_idx = np.asarray([a for a in range(0, 9000)])
test_idx = np.asarray([a for a in range(9000, 10000)])


## feature-level explanation: solving the mask.
# step_idx = np.asarray([a for a in range(0, 200)])
# BATCH_SIZE = 5
# N_EPOCHS = 1
# LR = 0.01
# DECAY = 0.1 # learning rate decay
# INITIALIZER = 'zero' # P initialize ['zero', 'one', 'uniform', 'normal']
# NORMALIZE_CHOICE = 'clip' # P normalize choice ['sigmoid', 'tanh', 'clip']
# UPSAMPLING_MODE = 'nearest' # up-sampling choice ['nearest', 'bilinear']
# EPSILON = 1e-8
# FUSED_CHOICE = None # fused choice ['mean', 'random', 'blur']
# NORM_CHOICE = 'l2' # loss norm choice  ['l2', 'l1', 'inf']

# REG_CHOICE = 'elasticnet' # regularization term to use ['l1', 'elasticnet']
# REG_COEF_1 = 1e-4 # coefficient of the shape regularization
# REG_COEF_2 = 1e-5 # coefficient of the smoothness regularization
# LAMBDA_PATIENCE = 20 # regularization multiple waiting epoch
# LAMBDA_MULTIPILER = 1.2 # regularization multipler
# ITER_THRD = 1e-3  # early stop threshold
# EARLY_STOP = 50 # early step waiting epoch
# DISP = 1 # display interval

# mask_explainer = MaskFeatExp(policy=policy, act_distribution='cat', input_shape=(4, 84, 84),
#                              mask_shape=(1, 84, 84), lr=LR, initializer=INITIALIZER, normalize_choice=NORMALIZE_CHOICE,
#                              upsampling_mode=UPSAMPLING_MODE, epsilon=EPSILON)

# p = mask_explainer.train(train_idx=train_idx, traj_path=traj_path, step_idx=step_idx, batch_size=BATCH_SIZE,
#                          n_epochs=N_EPOCHS, reg_choice=REG_CHOICE, reg_coef_1=REG_COEF_1, reg_coef_2=REG_COEF_2,
#                          temp=0.1, norm_choice=NORM_CHOICE, fused_choice=FUSED_CHOICE, lambda_patience=LAMBDA_PATIENCE,
#                          lambda_multiplier=LAMBDA_MULTIPILER, decay_weight=DECAY, iteration_threshold=ITER_THRD,
#                          early_stop_patience=EARLY_STOP, display_interval=DISP)

# out_img = Image.fromarray(np.squeeze((p*255).astype(np.uint8), axis=0))
# out_img.save('exp_models/feat_train_mask.png')

## Step-level explanation using the deep gaussian model
# HIDDENS = [4]
# LR = 0.01
# DECAY = 0.1 # learning rate decay
# BATCH_SIZE = 10
# N_EPOCHS = 10
# REG_WEIGHT = 0.001 # sparse regularization on the regression weight

# step_explainer = DGaussianStepExp(seq_len=200, input_dim=84, hiddens=HIDDENS, input_channels=4,
#                                   likelihood_type='classification', lr=LR, encoder_type='CNN', num_class=2)

# model_name = 'exp_models/dgaussian'+'_'+str(LR)+'_'+str(BATCH_SIZE)+'_'+str(DECAY)+'_'+str(REG_WEIGHT)

# step_explainer.train(n_epoch=N_EPOCHS, train_idx=train_idx, batch_size=BATCH_SIZE, traj_path=traj_path,
#                      reg_weight=REG_WEIGHT, decay_weight=DECAY, save_path=model_name)

# step_explainer.load(load_path='{}_epoch_{}.data'.format(model_name, N_EPOCHS))

# step_explainer.test(test_idx=test_idx, batch_size=BATCH_SIZE, traj_path=traj_path)

# step_importance_score = step_explainer.get_explanations(class_id=0)

# print(np.argsort(step_importance_score)[::-1])

## Step-level explanation using the DGP model.
HIDDENS = [4]
LR = 0.01
DECAY = 0.1 # learning rate decay
BATCH_SIZE = 20
N_EPOCHS = 30
REG_WEIGHT = 0.001 # sparse regularization on the regression weight
INDUCE_NUM = 300 # number of inducing points
WEIGHT_X = True

step_explainer = DGPStepExp(train_len=train_idx.shape[0], seq_len=200, input_dim=84, hiddens=HIDDENS, input_channels=4,
                            likelihood_type='classification', lr=LR, optimizer_type='adam', n_epoch=N_EPOCHS,
                            gamma=DECAY, num_inducing_points=INDUCE_NUM, encoder_type='CNN', num_class=2,
                            lambda_1=REG_WEIGHT, weight_x=WEIGHT_X)

model_name = 'pretrained_models/{}_{}_step_model.data'.format(args.subname, args.name)
step_explainer.train(train_idx=train_idx, batch_size=BATCH_SIZE, traj_path=traj_path, save_path=model_name)

step_explainer.load(load_path=model_name)

step_explainer.test(test_idx=test_idx, batch_size=BATCH_SIZE, traj_path=traj_path)


## Feature and step-level explanation using the deep gaussian model + mask explanation
# HIDDENS = [4]
# LR = 0.01
# INITIALIZER = 'normal' # P initialize ['zero', 'one', 'uniform', 'normal']
# NORMALIZE_CHOICE = 'clip' # P normalize choice ['sigmoid', 'tanh', 'clip']
# UPSAMPLING_MODE = 'nearest' # up-sampling choice ['nearest', 'bilinear']
# EPSILON = 1e-8
# FUSED_CHOICE = None # fused choice ['mean', 'random', 'blur']

# STEP_BATCH_SIZE = 5
# MASK_BATCH_SIZE = 50

# N_EPOCHS = 10
# REG_CHOICE = 'elasticnet' # regularization term to use ['l1', 'elasticnet']
# REG_COEF_1 = 1e-4 # coefficient of the shape regularization
# REG_COEF_2 = 1e-5 # coefficient of the smoothness regularization
# NORM_CHOICE = 'l2' # loss norm choice  ['l2', 'l1', 'inf']
# DECAY = 0.1 # learning rate decay
# LAMBDA_PATIENCE = 20 # regularization multiple waiting epoch
# LAMBDA_MULTIPILER = 1.2 # regularization multipler
# ITER_THRD = 1e-3  # early stop threshold
# EARLY_STOP = 50 # early step waiting epoch
# DISP = 1 # display interval
# REG_WEIGHT = 0.001 # sparse regularization on the regression weight

# feat_step_explainer = DGaussianStepFeatExp(seq_len=200, input_dim=80, hiddens=HIDDENS, input_channels=1,
#                                            likelihood_type='classification', lr=LR, mask_shape=(1, 40, 40),
#                                            act_distribution='cat',  policy=policy, encoder_type='CNN', num_class=2,
#                                            initializer=INITIALIZER, normalize_choice=NORMALIZE_CHOICE,
#                                            upsampling_mode=UPSAMPLING_MODE, fused_choice=FUSED_CHOICE,
#                                            epsilon=EPSILON)

# model_name = 'exp_models/mask_dgaussian'+'_'+str(LR)+'_'+str(FUSED_CHOICE)+'_'+str(INITIALIZER)+'_'+str(NORMALIZE_CHOICE)\
#              +'_'+str(UPSAMPLING_MODE)+'_'+str(REG_CHOICE)+'_'+str(REG_COEF_1)+'_'+str(REG_COEF_2)+'_'+str(REG_WEIGHT)\
#              +'_'+str(NORM_CHOICE)+'_'+str(STEP_BATCH_SIZE)+'_'+str(MASK_BATCH_SIZE)

# feat_step_explainer.train(n_epoch=N_EPOCHS, train_idx=train_idx, step_batch_size=STEP_BATCH_SIZE,
#                           mask_batch_size=MASK_BATCH_SIZE, traj_path=traj_path, reg_choice=REG_CHOICE,
#                           reg_coef_1=REG_COEF_1, reg_coef_2=REG_COEF_2, reg_coef_w=REG_WEIGHT, norm_choice=NORM_CHOICE,
#                           decay_weight=DECAY, lambda_patience=LAMBDA_PATIENCE, lambda_multiplier=LAMBDA_MULTIPILER,
#                           save_path=model_name)

# feat_step_explainer.load(load_path=
#                          'exp_models/mask_dgaussian_0.01_None_normal_clip_nearest_elasticnet_'
#                          '0.0001_1e-05_0.001_l2_5_50_10_model.data')

# feat_step_explainer.reward_pred_test(test_idx=train_idx, batch_size=STEP_BATCH_SIZE, traj_path=traj_path)

# step_importance_score, p = feat_step_explainer.get_explanations(class_id=0)

# print(np.argsort(step_importance_score)[::-1])

## Feature and step-level explanation using the DGP model + mask explanation
HIDDENS = [4]
LR = 0.01
INITIALIZER = 'zero' # P initialize ['zero', 'one', 'uniform', 'normal']
NORMALIZE_CHOICE = 'sigmoid' # P normalize choice ['sigmoid', 'tanh', 'clip']
UPSAMPLING_MODE = 'nearest' # up-sampling choice ['nearest', 'bilinear']
EPSILON = 1e-8
FUSED_CHOICE = 'mean' # fused choice ['mean', 'random', 'blur']
INDUCE_NUM = 300 # number of inducing points

STEP_BATCH_SIZE = 10
MASK_BATCH_SIZE = 100

N_EPOCHS = 2
REG_CHOICE = 'elasticnet' # regularization term to use ['l1', 'elasticnet']
REG_COEF_1 = 1e-4 # coefficient of the shape regularization
REG_COEF_2 = 1e-5 # coefficient of the smoothness regularization
NORM_CHOICE = 'l2' # loss norm choice  ['l2', 'l1', 'inf']
DECAY = 0.1 # learning rate decay
LAMBDA_PATIENCE = 20 # regularization multiple waiting epoch
LAMBDA_MULTIPILER = 1.2 # regularization multipler
ITER_THRD = 1e-3  # early stop threshold
EARLY_STOP = 50 # early step waiting epoch
DISP = 1 # display interval
REG_WEIGHT = 0.01 # sparse regularization on the regression weight

feat_step_explainer = DGPStepFeatExp(train_len=train_idx.shape[0], seq_len=200, input_dim=84, hiddens=HIDDENS,
                                     input_channels=4, likelihood_type='classification', lr=LR, mask_shape=(1, 84, 84),
                                     policy=policy, num_inducing_points=INDUCE_NUM, fused_choice=FUSED_CHOICE,
                                     act_distribution='cat', encoder_type='CNN', num_class=2, initializer=INITIALIZER,
                                     normalize_choice=NORMALIZE_CHOICE, upsampling_mode=UPSAMPLING_MODE,
                                     epsilon=EPSILON)

model_name = 'pretrained_models/{}_{}_mask_model.data'.format(args.subname, args.name)

feat_step_explainer.train(n_epoch=N_EPOCHS, train_idx=train_idx, step_batch_size=STEP_BATCH_SIZE,
                          mask_batch_size=MASK_BATCH_SIZE, traj_path=traj_path, reg_choice=REG_CHOICE,
                          reg_coef_1=REG_COEF_1, reg_coef_2=REG_COEF_2, reg_coef_w=REG_WEIGHT, norm_choice=NORM_CHOICE,
                          decay_weight=DECAY, lambda_patience=LAMBDA_PATIENCE, lambda_multiplier=LAMBDA_MULTIPILER,
                          save_path=model_name)

# feat_step_explainer.load(load_path='{}_{}_model.data'.format(model_name, N_EPOCHS))

# feat_step_explainer.reward_pred_test(test_idx=train_idx, batch_size=STEP_BATCH_SIZE, traj_path=traj_path)

# p = feat_step_explainer.get_explanations(class_id=0)
# out_img = Image.fromarray(np.squeeze((p*255).astype(np.uint8), axis=0))
# out_img.save('{}_test_mask.png'.format(model_name))


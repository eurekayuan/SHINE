import cv2
import math
import tqdm
import torch
import gpytorch
import numpy as np
import torch.optim as optim
from src.xstep import DGaussianModel, DGPModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.xfeat import MaskObsModel, elasticnet_loss, smoothness_loss, continuity_loss
from PIL import Image


class DGaussianStepFeatExp(object):
    def __init__(self, seq_len, input_dim, hiddens, input_channels, likelihood_type, lr, mask_shape, policy,
                 fused_choice='mean', temp=0.1, act_distribution='cat', encoder_type='MLP', dropout_rate=0.25,
                 num_class=None, rnn_cell_type='GRU', initializer='one', normalize_choice='sigmoid',
                 upsampling_mode='nearest', epsilon=1e-10, normalize=False):

        """ reward prediction + feature explanation
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param lr: learning rate
        :param mask_shape: mask shape
        :param policy: policy network
        :param fused_choice: values to fill in the masked part ['mean', 'random', 'blur']
        :param temp: temperature
        :param act_distribution: action distribution
        :param encoder_type: the encoder type
        :param dropout_rate: MLP dropout rate
        :param num_class: number of classes
        :param rnn_cell_type: the RNN cell type
        :param initializer: initializer to use ['zero', 'one', 'uniform', 'normal']
        :param normalize_choice: how to normalize the variable to between zero and one ['sigmoid', 'tanh', 'clip']
        :param upsampling_mode: upsampling mode ['nearest', 'bilinear']
        :param epsilon: same number to prevent numerical error
        :param normalize: whether to normalize the input
        """

        self.seq_len = seq_len
        self.policy = policy
        self.mask_shape = mask_shape
        self.fused_choice = fused_choice
        self.temp = temp
        if encoder_type == 'CNN':
            self.mask_model = MaskObsModel(policy, act_distribution, (input_channels, input_dim, input_dim), mask_shape,
                                           initializer, normalize_choice, upsampling_mode, epsilon)
        else:
            self.mask_model = MaskObsModel(policy, act_distribution, (input_dim,), mask_shape, initializer,
                                           normalize_choice, upsampling_mode, epsilon)

        self.step_model = DGaussianModel(seq_len, input_dim, hiddens, input_channels, likelihood_type, encoder_type,
                                         dropout_rate, num_class, rnn_cell_type, normalize)

        self.step_optimizer = optim.Adam([
            {'params': self.step_model.encoder.parameters(), 'weight_decay': 1e-4},
            {'params': self.step_model.f_mean_net.parameters()},
            {'params': self.step_model.f_std_net.parameters()},
            {'params': self.step_model.mix_weight},
            {'params': self.mask_model.logit_p}], lr=lr, weight_decay=0)

        self.feat_optimizer = optim.Adam([{'params': self.mask_model.logit_p}], lr=lr, weight_decay=0)

        if torch.cuda.is_available():
            self.step_model = self.step_model.cuda()
            self.mask_model = self.mask_model.cuda()

    def compute_feat_loss(self, acts_exp, acts_remain, acts, reg_choice, reg_coef_1, reg_coef_2, norm_choice='l2'):
        """
        compute mask prediction and regularization loss
        :param acts_exp: f(s*M), f is the policy network
        :param acts_remain: f(s*(1-M))
        :param acts: true actions given by the policy network
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param temp: temperature
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :return: prediction loss and regularization loss
        """

        if self.mask_model.act_distribution == 'cat':
            acts = acts.type(torch.long) - 1
            loss_func = torch.nn.CrossEntropyLoss()
            loss_exp = loss_func(acts_exp, acts)
            # loss_remain = -loss_func(acts_remain, acts)
        else:
            acts_exp_diff = acts - acts_exp
            # acts_remain_diff = acts - acts_remain

            if norm_choice == 'l1':
                # print('Use MAE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=1))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=1))
            elif norm_choice == 'inf':
                # print('Use infinity loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=float('inf')))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=float('inf')))
            else:
                # print('Use MSE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=2))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=2))

        if reg_choice == 'elasticnet':
            # print('Using elasticnet regularization.')
            loss_reg_mask = elasticnet_loss(self.mask_model.logit_p)
        else:
            # print('Using l1 regularization.')
            loss_reg_mask = torch.sum(torch.abs(self.mask_model.logit_p))

        if len(self.mask_model.logit_p.shape)==3:
            loss_smooth_mask = smoothness_loss(self.mask_model.logit_p)
        elif len(self.mask_model.logit_p.shape)==1:
            loss_smooth_mask = continuity_loss(self.mask_model.logit_p)
        else:
            raise TypeError('Only support image or vector observation...')

        loss_mask = loss_exp + reg_coef_1 * loss_reg_mask + reg_coef_2 * loss_smooth_mask

        return loss_mask, loss_exp, loss_reg_mask + loss_smooth_mask

    def compute_step_loss(self, prediction, target):
        """
        compute step prediction and regularization loss
        :param prediction: predicted rewards
        :param target: original rewards
        :return: prediction loss and regularization loss
        """

        if self.step_model.likelihood_type == 'classification':
            pred_loss = torch.nn.CrossEntropyLoss()(prediction, target)
        else:
            diff = target - prediction
            pred_loss = torch.mean(torch.linalg.norm(diff, dim=1, ord=2))

        pred_reg_loss = torch.sum(torch.abs(self.step_model.mix_weight))

        return pred_loss, pred_reg_loss

    def save(self, save_path):
        state_dict = self.step_model.state_dict()
        mask_dict = self.mask_model.state_dict()
        torch.save({'model': state_dict, 'mask': mask_dict}, save_path)
        return 0

    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        self.step_model.load_state_dict(model_dict)
        self.mask_model.load_state_dict(dicts['mask'])

        return self.step_model, self.mask_model

    def train_mask(self, obs, fused_obs, acts, hs, cs, mask_batch_size, reg_choice, reg_coef_1, reg_coef_2, temp=0.1,
                   norm_choice='l2'):
            """
            Update the explanation parameters based on feature loss
            :param obs: flatten observations
            :param fused_obs: fused observations
            :param acts: target actions
            :param hs: hidden states
            :param cs: hidden states
            :param mask_batch_size: batch size
            :param reg_choice: regularization term to use ['l1', 'elasticnet']
            :param reg_coef_1: coefficient of the shape regularization
            :param reg_coef_2: coefficient of the smoothness regularization
            :param temp: temperature
            :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
            :return: loss.
            """

            if obs.shape[0] % mask_batch_size == 0:
                n_batch = int(obs.shape[0] / mask_batch_size)
            else:
                n_batch = int(obs.shape[0] / mask_batch_size) + 1

            loss = 0
            loss_exp = 0
            loss_reg = 0

            # for batch in tqdm.tqdm(range(n_batch)):
            for batch in range(n_batch):

                batch_obs = obs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_fused_obs = fused_obs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_acts = acts[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_hs = hs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_cs = cs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]

                if torch.cuda.is_available():
                    batch_obs, batch_fused_obs, batch_acts, batch_hs, batch_cs = \
                        batch_obs.cuda(), batch_fused_obs.cuda(), batch_acts.cuda(), batch_hs.cuda(), batch_cs.cuda()

                obs_exp, obs_remain, acts_exp, acts_remain = self.mask_model(batch_obs[0:1, ], batch_fused_obs[0:1],
                                                                             (batch_hs[0], batch_cs[0]), temp)
                for i in range(batch_obs.shape[0] - 1):
                    obs_exp_tmp, obs_remain_tmp, acts_exp_tmp, acts_remain_tmp = \
                        self.mask_model(batch_obs[i + 1:i + 2], batch_fused_obs[i + 1:i + 2],
                                        (batch_hs[i + 1], batch_cs[i + 1]), temp)

                    obs_exp = torch.cat((obs_exp, obs_exp_tmp))
                    obs_remain = torch.cat((obs_remain, obs_remain_tmp))
                    acts_exp = torch.cat((acts_exp, acts_exp_tmp))
                    acts_remain = torch.cat((acts_remain, acts_remain_tmp))

                self.feat_optimizer.zero_grad()

                loss_batch, loss_exp_batch, loss_reg_batch = \
                    self.compute_feat_loss(acts_exp, acts_remain, batch_acts, reg_choice, reg_coef_1, reg_coef_2,
                                           norm_choice)
                loss_batch.backward()
                self.feat_optimizer.step()

                loss += loss_batch.cpu().detach().numpy()
                loss_exp += loss_exp_batch.cpu().detach().numpy()
                loss_reg += loss_reg_batch.cpu().detach().numpy()

            loss = loss / float(obs.shape[0])
            loss_exp = loss_exp / float(obs.shape[0])
            loss_reg = loss_reg / float(obs.shape[0])

            return loss, loss_exp, loss_reg

    def train(self, n_epoch, train_idx, step_batch_size, mask_batch_size, traj_path, reg_choice, reg_coef_1, reg_coef_2,
              reg_coef_w=0.01, norm_choice='l2', decay_weight=0.1, lambda_patience=10, lambda_multiplier=1.5,
              save_path=None):

        """
        Training function
        :param n_epoch: training epoch
        :param train_idx: training traj index
        :param step_batch_size: step prediction training batch size
        :param mask_batch_size: feature prediction training batch size
        :param traj_path: training traj path
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param reg_coef_w: coefficient of the regularization loss on weight (w)
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :param decay_weight: lr decay weight
        :param lambda_patience: lambda update patience
        :param lambda_multiplier: lambda update multiplier
        :param save_path: model save path
        :return: trained model
        """

        self.step_model.train()
        self.mask_model.train()

        step_scheduler = optim.lr_scheduler.MultiStepLR(self.step_optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                        gamma=decay_weight)

        feat_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.feat_optimizer,
                                                              milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                              gamma=decay_weight)

        if train_idx.shape[0] % step_batch_size == 0:
            n_batch = int(train_idx.shape[0] / step_batch_size)
        else:
            n_batch = int(train_idx.shape[0] / step_batch_size) + 1

        lambda_up_counter = 0
        lambda_down_counter = 0

        loss_feat_exp_last = 0
        loss_exp_best = math.inf
        best_model_iter = 0

        batch_sample_num = step_batch_size*self.seq_len

        for epoch in range(1, n_epoch + 1):
            print('{} out of {} epochs.'.format(epoch, n_epoch))
            mse = 0
            loss = 0
            loss_feat_exp = 0
            loss_feat_reg = 0
            loss_step_exp = 0
            loss_step_reg = 0

            preds_all = []
            rewards_all = []

            for batch in tqdm.tqdm(range(n_batch)):
                batch_obs = []
                batch_acts = []
                batch_rewards = []
                batch_hs = []
                batch_cs = []

                for idx in train_idx[batch * step_batch_size:min((batch + 1) * step_batch_size, train_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                    batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                    batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])
                    batch_hs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['h'])
                    batch_cs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['c'])

                batch_obs_flatten = np.array(batch_obs)
                batch_obs_flatten = batch_obs_flatten.reshape(batch_sample_num, *batch_obs_flatten.shape[2:])

                batch_acts = np.array(batch_acts)
                batch_acts = batch_acts.reshape(batch_sample_num, *batch_acts.shape[2:])

                batch_hs = np.array(batch_hs)
                batch_hs = batch_hs.reshape(batch_sample_num, *batch_hs.shape[2:])

                batch_cs = np.array(batch_cs)
                batch_cs = batch_cs.reshape(batch_sample_num, *batch_cs.shape[2:])
                # for debugging
                # batch_hs = np.zeros((batch_sample_num, 1, 256))
                # batch_cs = np.zeros((batch_sample_num, 1, 256))

                # Remove the padding state action pair where action=0
                nonzero_idx = np.unique(np.where(batch_acts != 0)[0])
                batch_obs_flatten_non_padding = batch_obs_flatten[nonzero_idx,]
                batch_acts = batch_acts[nonzero_idx,]
                batch_hs = batch_hs[nonzero_idx,]
                batch_cs = batch_cs[nonzero_idx,]

                # todo: Support only one channel, add three channels support
                # Generate fused images
                if len(batch_obs_flatten.shape) == 4:
                    if self.fused_choice == 'mean':
                        batch_fused_obs = np.mean(batch_obs_flatten, axis=0).repeat(batch_sample_num, 0)
                    elif self.fused_choice == 'random':
                        batch_fused_obs = np.random.normal(loc=0, scale=0.1, size=batch_obs_flatten.shape[1:]).repeat(batch_sample_num, 0)
                    elif self.fused_choice == 'blur':
                        batch_fused_obs = cv2.GaussianBlur(batch_obs_flatten[:,0,:,:], (5, 5), cv2.BORDER_DEFAULT)
                    else:
                        batch_fused_obs = np.zeros(batch_obs_flatten.shape[1:]).repeat(batch_sample_num, 0)
                    batch_fused_obs = batch_fused_obs[:, None, :, :]
                elif len(batch_obs_flatten.shape) == 2:
                    if self.fused_choice == 'mean':
                        batch_fused_obs = np.mean(batch_obs_flatten, axis=0)
                    elif self.fused_choice == 'random':
                        batch_fused_obs = np.random.normal(loc=0, scale=0.1, size=batch_obs_flatten.shape[1:])
                    elif self.fused_choice == 'blur':
                        raise TypeError("Non-image observation does not blur...")
                    else:
                        batch_fused_obs = np.zeros(batch_obs_flatten.shape[1:])
                    batch_fused_obs = batch_fused_obs[None, ].repeat(batch_sample_num, 0)
                else:
                    raise TypeError('Only support image or vector observation...')
                batch_fused_obs_non_padding = batch_fused_obs[nonzero_idx,]

                batch_fused_obs = torch.tensor(batch_fused_obs, dtype=torch.float32)
                batch_fused_obs_non_padding = torch.tensor(batch_fused_obs_non_padding, dtype=torch.float32)

                obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
                batch_obs_flatten = torch.tensor(batch_obs_flatten, dtype=torch.float32)
                batch_obs_flatten_non_padding = torch.tensor(batch_obs_flatten_non_padding, dtype=torch.float32)

                acts = torch.tensor(batch_acts, dtype=torch.float32)
                hs = torch.tensor(batch_hs, dtype=torch.float32)
                cs = torch.tensor(batch_cs, dtype=torch.float32)

                if self.step_model.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, batch_obs_flatten, batch_obs_flatten_non_padding, batch_fused_obs, batch_fused_obs_non_padding = \
                        obs.cuda(), batch_obs_flatten.cuda(), batch_obs_flatten_non_padding.cuda(),\
                        batch_fused_obs.cuda(), batch_fused_obs_non_padding.cuda()
                    acts, rewards, hs, cs = acts.cuda(), rewards.cuda(), hs.cuda(), cs.cuda()

                loss_feat, loss_feat_exp_batch, loss_feat_reg_batch = \
                    self.train_mask(batch_obs_flatten_non_padding, batch_fused_obs_non_padding, acts, hs, cs,
                                    mask_batch_size, reg_choice, reg_coef_1, reg_coef_2, self.temp, norm_choice)

                loss_feat_exp += loss_feat_exp_batch
                loss_feat_reg += loss_feat_reg_batch

                self.step_optimizer.zero_grad()

                obs_exp = self.mask_model(batch_obs_flatten, batch_fused_obs, temp=self.temp, compute_obs_only=True)

                obs = obs_exp.reshape(obs.shape)
                preds = self.step_model(obs)

                pred_loss_batch, pred_reg_loss_batch = self.compute_step_loss(preds, rewards)

                loss_step_batch = pred_loss_batch + reg_coef_w * pred_reg_loss_batch
                loss_step_batch.backward()
                self.step_optimizer.step()

                loss = loss + loss_step_batch.cpu().detach().numpy() + loss_feat
                loss_step_exp += pred_loss_batch.cpu().detach().numpy()
                loss_step_reg += pred_reg_loss_batch.cpu().detach().numpy()

                if self.step_model.likelihood_type == 'classification':
                    preds = preds.argmax(-1)
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

            loss = loss / float(n_batch)
            loss_feat_exp = loss_feat_exp / float(n_batch)
            loss_feat_reg = loss_feat_reg / float(n_batch)
            loss_step_exp = loss_step_exp / float(n_batch)
            loss_step_reg = loss_step_reg / float(n_batch)

            # check cost modification
            if epoch > n_epoch/2 and loss_feat_exp <= loss_feat_exp_last:
                lambda_up_counter += 1
                if lambda_up_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 * lambda_multiplier
                    reg_coef_2 = reg_coef_2 * lambda_multiplier
                    lambda_up_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))
            elif epoch > n_epoch/2 and loss_feat_exp > loss_feat_exp_last:
                lambda_down_counter += 1
                if lambda_down_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 / lambda_multiplier
                    reg_coef_2 = reg_coef_2 / lambda_multiplier
                    lambda_down_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))

            loss_feat_exp_last = loss_feat_exp

            print("Epoch %d/%d: loss = %.5f feat_explanation_loss = %.5f feat_reg_loss = %.5f "
                  "step_prediction loss = %.5f step_reg_loss = %.5f"
                  % (epoch, n_epoch, loss, loss_feat_exp, loss_feat_reg, loss_step_exp, loss_step_reg))

            if self.step_model.likelihood_type == 'classification':
                preds_all = np.array(preds_all)
                rewards_all = np.array(rewards_all)
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
                acc = accuracy_score(rewards_all, preds_all)
                for cls in range(len(precision)):
                    print('Train results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                          format(cls, precision[cls], recall[cls], f1[cls], acc))
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
                print('Overall training results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(precision, recall, f1, acc))
            else:
                print('Train MSE: {}'.format(mse / float(train_idx.shape[0]))).cpu().detach().numpy()

            if loss_exp_best > (loss_feat_exp + loss_step_exp):
                loss_exp_best = loss_feat_exp + loss_step_exp
                best_model_iter = epoch

            feat_scheduler.step()
            step_scheduler.step()

            self.save(save_path + '_' + str(epoch) + '_model.data')

        return best_model_iter

    def reward_pred_test(self, test_idx, batch_size, traj_path, use_mask=True):
        """ testing function
        :param test_idx: testing traj index
        :param batch_size: testing batch size
        :param traj_path: testing traj path
        :param use_mask: whether to use masked observation
        :return: prediction error.
        """

        # Specify that the model is in eval mode.
        self.step_model.eval()
        self.mask_model.eval()

        mse = 0
        preds_all = []
        rewards_all = []

        if test_idx.shape[0] % batch_size == 0:
            n_batch = int(test_idx.shape[0] / batch_size)
        else:
            n_batch = int(test_idx.shape[0] / batch_size) + 1

        batch_sample_num = batch_size * self.seq_len

        for batch in range(n_batch):
            batch_obs = []
            batch_rewards = []

            for idx in test_idx[batch * batch_size:min((batch + 1) * batch_size, test_idx.shape[0]), ]:
                batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

            obs_flatten = np.array(batch_obs)
            obs_flatten = obs_flatten.reshape(batch_sample_num, *obs_flatten.shape[2:])

            # todo: Support only one channel, add three channels support
            # Generate fused images
            if len(obs_flatten.shape) == 4:
                if self.fused_choice == 'mean':
                    fused_obs = np.mean(obs_flatten, axis=0).repeat(batch_sample_num, 0)
                elif self.fused_choice == 'random':
                    fused_obs = np.random.normal(loc=0, scale=0.1, size=obs_flatten.shape[1:]).repeat(batch_sample_num, 0)
                elif self.fused_choice == 'blur':
                    fused_obs = cv2.GaussianBlur(obs_flatten[:, 0, :, :], (5, 5), cv2.BORDER_DEFAULT)
                else:
                    fused_obs = np.zeros(obs_flatten.shape[1:]).repeat(batch_sample_num, 0)
                fused_obs = torch.tensor(fused_obs[:, None, :, :], dtype=torch.float32)
            elif len(obs_flatten.shape) == 2:
                if self.fused_choice == 'mean':
                    fused_obs = np.mean(obs_flatten, axis=0)
                elif self.fused_choice == 'random':
                    fused_obs = np.random.normal(loc=0, scale=0.1, size=obs_flatten.shape[1:])
                elif self.fused_choice == 'blur':
                    raise TypeError("Non-image observation does not blur...")
                else:
                    fused_obs = np.zeros(obs_flatten.shape[1:])
                fused_obs = fused_obs[None,].repeat(batch_sample_num, 0)
                fused_obs = torch.tensor(fused_obs, dtype=torch.float32)
            else:
                raise TypeError('Only support image or vector observation...')

            obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
            obs_flatten = torch.tensor(obs_flatten, dtype=torch.float32)

            if self.step_model.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, obs_flatten, fused_obs, rewards = obs.cuda(), obs_flatten.cuda(), fused_obs.cuda(), rewards.cuda()

            if use_mask:
                # print('Use masked observation for reward prediction.')
                obs_exp = self.mask_model(obs_flatten, fused_obs, temp=self.temp, compute_obs_only=True)
                obs = obs_exp.reshape(obs.shape)

            preds = self.step_model(obs)

            if self.step_model.likelihood_type == 'classification':
                preds = preds.argmax(-1)
                preds_all.extend(preds.cpu().detach().numpy().tolist())
                rewards_all.extend(rewards.cpu().detach().numpy().tolist())
            else:
                preds = preds.mean.detach()
                mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

        if self.step_model.likelihood_type == 'classification':
            preds_all = np.array(preds_all)
            rewards_all = np.array(rewards_all)
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
            acc = accuracy_score(rewards_all, preds_all)
            for cls in range(len(precision)):
                print('Test results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(cls, precision[cls], recall[cls], f1[cls], acc))
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
            print('Overall test results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                  format(precision, recall, f1, acc))
        else:
            print('Test MSE: {}'.format(mse / float(test_idx.shape[0])))

        return 0

    def get_explanations(self, class_id, normalize=True, model_path=None, load_model=False):
        """
        get explanation for a specific class
        :param class_id: the ID of a class
        :param normalize: normalize the explanation or not
        :param model_path: model path
        :param load_model: load model or not
        :return: step and mask explanation
        """

        if load_model:
            self.step_model, self.mask_model = self.load(model_path)

        self.step_model.eval()
        self.mask_model.eval()

        importance_all = self.step_model.mix_weight
        importance_all = importance_all.transpose(1, 0)

        importance_all = importance_all.cpu().detach().numpy()

        if importance_all.shape[-1] > 1:
            saliency = importance_all[:, class_id]
        else:
            saliency = np.squeeze(importance_all, -1)

        if normalize:
            saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-16)

        p = self.mask_model.compute_p()

        return saliency, p.cpu().detach().numpy()


class DGPStepFeatExp(object):
    def __init__(self, train_len, seq_len, input_dim, hiddens, input_channels, likelihood_type, lr,
                 mask_shape, policy, num_inducing_points, temp=0.1, fused_choice='mean', act_distribution='cat',
                 encoder_type='MLP', dropout_rate=0.25, num_class=None, rnn_cell_type='GRU', normalize=False,
                 grid_bounds=None, initializer='one', normalize_choice='sigmoid', upsampling_mode='nearest',
                 epsilon=1e-10):

        """ reward prediction (DGP) + feature explanation
        :param train_len: number of training trajectory
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param lr: learning rate
        :param policy: policy network
        :param mask_shape: shape of the explanation mask
        :param num_inducing_points: number of inducing points
        :param encoder_type: encoder type ('MLP' or 'CNN')
        :param dropout_rate: MLP dropout rate
        :param rnn_cell_type: the RNN cell type
        :param normalize: whether to normalize the input
        :param grid_bounds: grid bounds
        :param temp: temperature
        :param fused_choice: fused image choice
        :param act_distribution: action distribution
        :param initializer: initializer to use ['zero', 'one', 'uniform', 'normal']
        :param normalize_choice: how to normalize the variable to between zero and one ['sigmoid', 'tanh', 'clip']
        :param upsampling_mode: upsampling mode ['nearest', 'bilinear']
        :param epsilon: same number to prevent numerical error
        """

        self.train_len = train_len
        self.seq_len = seq_len
        self.likelihood_type = likelihood_type
        self.lr = lr
        self.temp = temp
        self.fused_choice = fused_choice
        self.policy = policy
        self.mask_shape = mask_shape

        if encoder_type == 'CNN':
            self.mask_model = MaskObsModel(policy, act_distribution, (input_channels, input_dim, input_dim), mask_shape,
                                           initializer, normalize_choice, upsampling_mode, epsilon)
        else:
            self.mask_model = MaskObsModel(policy, act_distribution, (input_dim,), mask_shape, initializer,
                                           normalize_choice, upsampling_mode, epsilon)

        self.step_model = DGPModel(seq_len, input_dim, hiddens, input_channels, likelihood_type, num_inducing_points,
                                   encoder_type, inducing_points=None, mean_inducing_points=None,
                                   dropout_rate=dropout_rate, num_class=num_class, rnn_cell_type=rnn_cell_type,
                                   normalize=normalize, grid_bounds=grid_bounds, using_ngd=False, using_ksi=False,
                                   using_ciq=False, using_sor=False, using_OrthogonallyDecouple=False, weight_x=True)

        self.mll = gpytorch.mlls.VariationalELBO(self.step_model.likelihood, self.step_model.model.gp_layer,
                                                 num_data=self.train_len)

        self.step_optimizer = optim.Adam([
            {'params': self.step_model.model.encoder.parameters(), 'weight_decay': 1e-4},
            {'params': self.step_model.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
            {'params': self.step_model.model.gp_layer.variational_parameters()},
            {'params': self.step_model.likelihood.parameters()},
            {'params': self.mask_model.logit_p}], lr=self.lr, weight_decay=0)

        self.feat_optimizer = optim.Adam([{'params': self.mask_model.logit_p}], lr=lr, weight_decay=0)

        if torch.cuda.is_available():
            self.step_model = self.step_model.cuda()
            self.mask_model = self.mask_model.cuda()

    def compute_feat_loss(self, acts_exp, acts_remain, acts, reg_choice, reg_coef_1, reg_coef_2, norm_choice='l2'):
        """
        compute mask prediction and regularization loss
        :param acts_exp: f(s*M), f is the policy network
        :param acts_remain: f(s*(1-M))
        :param acts: true actions given by the policy network
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param temp: temperature
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :return: prediction loss and regularization loss
        """

        if self.mask_model.act_distribution == 'cat':
            acts = acts.type(torch.long) - 1
            loss_func = torch.nn.CrossEntropyLoss()
            loss_exp = loss_func(acts_exp, acts)
            # loss_remain = -loss_func(acts_remain, acts)
        else:
            acts_exp_diff = acts - acts_exp
            # acts_remain_diff = acts - acts_remain

            if norm_choice == 'l1':
                # print('Use MAE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=1))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=1))
            elif norm_choice == 'inf':
                # print('Use infinity loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=float('inf')))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=float('inf')))
            else:
                # print('Use MSE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=2))
                # loss_remain = -torch.mean(torch.linalg.norm(acts_remain_diff, dim=1, ord=2))

        if reg_choice == 'elasticnet':
            # print('Using elasticnet regularization.')
            loss_reg_mask = elasticnet_loss(self.mask_model.logit_p)
        else:
            # print('Using l1 regularization.')
            loss_reg_mask = torch.sum(torch.abs(self.mask_model.logit_p))

        if len(self.mask_model.logit_p.shape)==3:
            loss_smooth_mask = smoothness_loss(self.mask_model.logit_p)
        elif len(self.mask_model.logit_p.shape)==1:
            loss_smooth_mask = continuity_loss(self.mask_model.logit_p)
        else:
            raise TypeError('Only support image or vector observation...')

        loss_mask = loss_exp + reg_coef_1 * loss_reg_mask + reg_coef_2 * loss_smooth_mask

        return loss_mask, loss_exp, loss_reg_mask + loss_smooth_mask

    def compute_step_loss(self, prediction, target, features=None):
        """
        compute prediction and regularization loss
        :param prediction: predicted rewards
        :param target: original rewards
        :return: prediction loss and regularization loss
        """
        if self.step_model.weight_x:
            pred_loss = -self.mll(prediction, target, input_encoding=features)  # approximated ELBO.
        else:
            pred_loss = -self.mll(prediction, target)  # approximated ELBO.

        if self.step_model.weight_x and self.likelihood_type == 'classification':
            # lasso.
            features_sum = features.detach().sum(-1)
            weight_output = self.step_model.likelihood.weight_encoder(features_sum)
            pred_reg_loss = torch.norm(weight_output, p=1) # lasso
        else:
            pred_reg_loss = torch.norm(self.step_model.likelihood.mixing_weights, p=1) # lasso

        return pred_loss, pred_reg_loss

    def save(self, save_path):
        state_dict = self.step_model.state_dict()
        mask_dict = self.mask_model.state_dict()
        torch.save({'model': state_dict, 'mask': mask_dict}, save_path)
        return 0

    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        self.step_model.load_state_dict(model_dict)
        self.mask_model.load_state_dict(dicts['mask'])

        return self.step_model, self.mask_model

    def train_mask(self, obs, fused_obs, acts, hs, cs, mask_batch_size, reg_choice, reg_coef_1, reg_coef_2, temp=0.1,
                   norm_choice='l2'):
            """
            Update the explanation parameters based on feature loss
            :param obs: flatten observations
            :param fused_obs: fused observations
            :param acts: target actions
            :param hs: hidden states
            :param cs: hidden states
            :param mask_batch_size: batch size
            :param reg_choice: regularization term to use ['l1', 'elasticnet']
            :param reg_coef_1: coefficient of the shape regularization
            :param reg_coef_2: coefficient of the smoothness regularization
            :param temp: temperature
            :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
            :return: loss.
            """

            if obs.shape[0] % mask_batch_size == 0:
                n_batch = int(obs.shape[0] / mask_batch_size)
            else:
                n_batch = int(obs.shape[0] / mask_batch_size) + 1

            loss = 0
            loss_exp = 0
            loss_reg = 0

            # for batch in tqdm.tqdm(range(n_batch)):
            for batch in range(n_batch):
                batch_obs = obs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_fused_obs = fused_obs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_acts = acts[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_hs = hs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]
                batch_cs = cs[batch * mask_batch_size: min((batch + 1) * mask_batch_size, obs.shape[0]), ]

                if torch.cuda.is_available():
                    batch_obs, batch_fused_obs, batch_acts, batch_hs, batch_cs = \
                        batch_obs.cuda(), batch_fused_obs.cuda(), batch_acts.cuda(), batch_hs.cuda(), batch_cs.cuda()

                obs_exp, obs_remain, acts_exp, acts_remain = self.mask_model(batch_obs[0:1, ], batch_fused_obs[0:1],
                                                                             (batch_hs[0], batch_cs[0]), temp)
                for i in range(batch_obs.shape[0] - 1):
                    obs_exp_tmp, obs_remain_tmp, acts_exp_tmp, acts_remain_tmp = \
                        self.mask_model(batch_obs[i + 1:i + 2], batch_fused_obs[i + 1:i + 2],
                                        (batch_hs[i + 1], batch_cs[i + 1]), temp)

                    obs_exp = torch.cat((obs_exp, obs_exp_tmp))
                    obs_remain = torch.cat((obs_remain, obs_remain_tmp))
                    acts_exp = torch.cat((acts_exp, acts_exp_tmp))
                    acts_remain = torch.cat((acts_remain, acts_remain_tmp))

                self.feat_optimizer.zero_grad()

                loss_batch, loss_exp_batch, loss_reg_batch = \
                    self.compute_feat_loss(acts_exp, acts_remain, batch_acts, reg_choice, reg_coef_1, reg_coef_2,
                                           norm_choice)
                loss_batch.backward()
                self.feat_optimizer.step()

                loss += loss_batch.cpu().detach().numpy()
                loss_exp += loss_exp_batch.cpu().detach().numpy()
                loss_reg += loss_reg_batch.cpu().detach().numpy()

            loss = loss / float(obs.shape[0])
            loss_exp = loss_exp / float(obs.shape[0])
            loss_reg = loss_reg / float(obs.shape[0])

            return loss, loss_exp, loss_reg

    def train(self, n_epoch, train_idx, step_batch_size, mask_batch_size, traj_path, reg_choice, reg_coef_1, reg_coef_2,
              reg_coef_w=0.01, norm_choice='l2', decay_weight=0.1, lambda_patience=10, lambda_multiplier=1.5,
              save_path=None, likelihood_sample_size=8):

        """
        Training function
        :param train_idx: training traj index
        :param step_batch_size: step prediction training batch size
        :param mask_batch_size: feature prediction training batch size
        :param traj_path: training traj path
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param reg_coef_w: coefficient of the regularization loss on weight (w)
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :param decay_weight: lr decay weight
        :param lambda_patience: lambda update patience
        :param lambda_multiplier: lambda update multiplier
        :param save_path: model save path
        :param likelihood_sample_size:
        :return: trained model
        """

        step_scheduler = optim.lr_scheduler.MultiStepLR(self.step_optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                        gamma=decay_weight)

        feat_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.feat_optimizer,
                                                              milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                              gamma=decay_weight)

        if train_idx.shape[0] % step_batch_size == 0:
            n_batch = int(train_idx.shape[0] / step_batch_size)
        else:
            n_batch = int(train_idx.shape[0] / step_batch_size) + 1

        lambda_up_counter = 0
        lambda_down_counter = 0

        loss_feat_exp_last = 0
        loss_exp_best = math.inf
        best_model_iter = 0

        batch_sample_num = step_batch_size*self.seq_len

        for epoch in range(1, n_epoch + 1):
            self.step_model.train()
            self.mask_model.train()

            print('{} out of {} epochs.'.format(epoch, n_epoch))
            mse = 0
            loss = 0
            loss_feat_exp = 0
            loss_feat_reg = 0
            loss_step_exp = 0
            loss_step_reg = 0

            preds_all = []
            rewards_all = []

            with gpytorch.settings.use_toeplitz(False):
                with gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
                    for batch in tqdm.tqdm(range(n_batch)):

                        batch_obs = []
                        batch_acts = []
                        batch_rewards = []
                        batch_hs = []
                        batch_cs = []

                        for idx in train_idx[
                                   batch * step_batch_size:min((batch + 1) * step_batch_size, train_idx.shape[0]), ]:
                            batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                            batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                            batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])
                            # batch_hs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['h'])
                            # batch_cs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['c'])

                        batch_obs_flatten = np.array(batch_obs)
                        batch_obs_flatten = batch_obs_flatten.reshape(batch_sample_num, *batch_obs_flatten.shape[2:])

                        batch_acts = np.array(batch_acts)
                        batch_acts = batch_acts.reshape(batch_sample_num, *batch_acts.shape[2:])

                        # batch_hs = np.array(batch_hs)
                        # batch_hs = batch_hs.reshape(batch_sample_num, *batch_hs.shape[2:])
                        #
                        # batch_cs = np.array(batch_cs)
                        # batch_cs = batch_cs.reshape(batch_sample_num, *batch_cs.shape[2:])

                        # for debugging.
                        batch_hs = np.zeros((batch_sample_num, 1, 256))
                        batch_cs = np.zeros((batch_sample_num, 1, 256))

                        # Remove the padding state action pair where action=0
                        nonzero_idx = np.unique(np.where(batch_acts != 0)[0])
                        batch_obs_flatten_non_padding = batch_obs_flatten[nonzero_idx,]
                        batch_acts = batch_acts[nonzero_idx,]
                        batch_hs = batch_hs[nonzero_idx,]
                        batch_cs = batch_cs[nonzero_idx,]

                        # todo: Support only one channel, add three channels support
                        # Generate fused images
                        if len(batch_obs_flatten.shape) == 4:
                            if self.fused_choice == 'mean':
                                batch_fused_obs = np.mean(batch_obs_flatten, axis=0)[None, :, :, :].repeat(batch_sample_num, 0)
                            elif self.fused_choice == 'random':
                                batch_fused_obs = np.random.normal(loc=0, scale=0.1,
                                                                   size=batch_obs_flatten.shape[1:])[None, :, :, :].repeat(
                                    batch_sample_num, 0)
                            elif self.fused_choice == 'blur':
                                batch_fused_obs = cv2.GaussianBlur(batch_obs_flatten[:, :, :, :], (5, 5),
                                                                   cv2.BORDER_DEFAULT)
                            else:
                                batch_fused_obs = np.zeros(batch_obs_flatten.shape[1:])[None, :, :, :].repeat(batch_sample_num, 0)
                            # batch_fused_obs = batch_fused_obs[:, None, :, :]
                        elif len(batch_obs_flatten.shape) == 2:
                            if self.fused_choice == 'mean':
                                batch_fused_obs = np.mean(batch_obs_flatten, axis=0)
                            elif self.fused_choice == 'random':
                                batch_fused_obs = np.random.normal(loc=0, scale=0.1, size=batch_obs_flatten.shape[1:])
                            elif self.fused_choice == 'blur':
                                raise TypeError("Non-image observation does not blur...")
                            else:
                                batch_fused_obs = np.zeros(batch_obs_flatten.shape[1:])
                            batch_fused_obs = batch_fused_obs[None,].repeat(batch_sample_num, 0)
                        else:
                            raise TypeError('Only support image or vector observation...')
                        batch_fused_obs_non_padding = batch_fused_obs[nonzero_idx,]

                        batch_fused_obs = torch.tensor(batch_fused_obs, dtype=torch.float32)
                        batch_fused_obs_non_padding = torch.tensor(batch_fused_obs_non_padding, dtype=torch.float32)

                        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
                        batch_obs_flatten = torch.tensor(batch_obs_flatten, dtype=torch.float32)
                        batch_obs_flatten_non_padding = torch.tensor(batch_obs_flatten_non_padding, dtype=torch.float32)

                        acts = torch.tensor(batch_acts, dtype=torch.float32)
                        hs = torch.tensor(batch_hs, dtype=torch.float32)
                        cs = torch.tensor(batch_cs, dtype=torch.float32)

                        if self.step_model.likelihood_type == 'classification':
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                        else:
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                        if torch.cuda.is_available():
                            obs, batch_obs_flatten, batch_obs_flatten_non_padding = \
                                obs.cuda(), batch_obs_flatten.cuda(), batch_obs_flatten_non_padding.cuda()

                            batch_fused_obs, batch_fused_obs_non_padding = \
                                batch_fused_obs.cuda(), batch_fused_obs_non_padding.cuda()

                            acts, rewards, hs, cs = acts.cuda(), rewards.cuda(), hs.cuda(), cs.cuda()

                        loss_feat, loss_feat_exp_batch, loss_feat_reg_batch = \
                            self.train_mask(batch_obs_flatten_non_padding, batch_fused_obs_non_padding, acts, hs, cs,
                                            mask_batch_size, reg_choice, reg_coef_1, reg_coef_2, self.temp, norm_choice)

                        loss_feat_exp += loss_feat_exp_batch
                        loss_feat_reg += loss_feat_reg_batch

                        self.step_optimizer.zero_grad()

                        obs_exp = self.mask_model(batch_obs_flatten, batch_fused_obs, temp=self.temp,
                                                  compute_obs_only=True)

                        obs = obs_exp.reshape(obs.shape)
                        output, features = self.step_model.model(obs)  # marginal variational posterior, q(f|x).
                        pred_loss_batch, pred_reg_loss_batch = self.compute_step_loss(output, rewards, features)

                        if self.step_model.weight_x:
                            output = self.step_model.likelihood(output, input_encoding=features)
                        else:
                            output = self.step_model.likelihood(output) # y = E_{f_* ~ q(f_*)}[y|f_*])

                        loss_step_batch = pred_loss_batch + reg_coef_w * pred_reg_loss_batch
                        loss_step_batch.backward()
                        self.step_optimizer.step()

                        loss = loss + loss_step_batch.cpu().detach().numpy() + loss_feat
                        loss_step_exp += pred_loss_batch.cpu().detach().numpy()
                        loss_step_reg += pred_reg_loss_batch.cpu().detach().numpy()

                        if self.step_model.likelihood_type == 'classification':
                            preds = output.probs.mean(0).argmax(-1)
                            preds_all.extend(preds.cpu().detach().numpy().tolist())
                            rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                        else:
                            preds = output.mean
                            mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

            loss = loss / float(n_batch)
            loss_feat_exp = loss_feat_exp / float(n_batch)
            loss_feat_reg = loss_feat_reg / float(n_batch)
            loss_step_exp = loss_step_exp / float(n_batch)
            loss_step_reg = loss_step_reg / float(n_batch)

            # check cost modification
            if epoch > n_epoch/2 and loss_feat_exp <= loss_feat_exp_last:
                lambda_up_counter += 1
                if lambda_up_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 * lambda_multiplier
                    reg_coef_2 = reg_coef_2 * lambda_multiplier
                    lambda_up_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))
            elif epoch > n_epoch/2 and loss_feat_exp > loss_feat_exp_last:
                lambda_down_counter += 1
                if lambda_down_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 / lambda_multiplier
                    reg_coef_2 = reg_coef_2 / lambda_multiplier
                    lambda_down_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))

            loss_feat_exp_last = loss_feat_exp

            print("Epoch %d/%d: loss = %.5f feat_explanation_loss = %.5f feat_reg_loss = %.5f "
                  "step_prediction loss = %.5f step_reg_loss = %.5f"
                  % (epoch, n_epoch, loss, loss_feat_exp, loss_feat_reg, loss_step_exp, loss_step_reg))

            if self.step_model.likelihood_type == 'classification':
                preds_all = np.array(preds_all)
                rewards_all = np.array(rewards_all)
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
                acc = accuracy_score(rewards_all, preds_all)
                for cls in range(len(precision)):
                    print('Train results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                          format(cls, precision[cls], recall[cls], f1[cls], acc))
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
                print('Overall training results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(precision, recall, f1, acc))
            else:
                print('Train MSE: {}'.format(mse / float(train_idx.shape[0]))).cpu().detach().numpy()

            if loss_exp_best > (loss_feat_exp + loss_step_exp):
                loss_exp_best = loss_feat_exp + loss_step_exp
                best_model_iter = epoch

            feat_scheduler.step()
            step_scheduler.step()

            self.save(save_path)

            p = self.get_explanations(class_id=0)
            out_img = Image.fromarray(np.squeeze((p*255).astype(np.uint8), axis=0))
            out_img.save('{}.png'.format('_'.join(save_path.split('_')[:-1])))

        return best_model_iter

    def reward_pred_test(self, test_idx, batch_size, traj_path, likelihood_sample_size=16, use_mask=True):

        """ testing function
        :param test_idx: testing traj index
        :param batch_size: testing batch size
        :param traj_path: testing traj path
        :param likelihood_sample_size: number of samples from the likelihood
        :param use_mask: number of samples from the likelihood
        :return: prediction error
        """

        # Specify that the model is in eval mode.
        self.step_model.eval()
        self.mask_model.eval()

        mse = 0
        preds_all = []
        rewards_all = []

        batch_sample_num = batch_size * self.seq_len

        if test_idx.shape[0] % batch_size == 0:
            n_batch = int(test_idx.shape[0] / batch_size)
        else:
            n_batch = int(test_idx.shape[0] / batch_size) + 1

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
            for batch in range(n_batch):
                batch_obs = []
                batch_rewards = []

                for idx in test_idx[batch * batch_size:min((batch + 1) * batch_size, test_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                    batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

                obs_flatten = np.array(batch_obs)
                obs_flatten = obs_flatten.reshape(batch_sample_num, *obs_flatten.shape[2:])

                # todo: Support only one channel, add three channels support
                # Generate fused images
                if len(obs_flatten.shape) == 4:
                    if self.fused_choice == 'mean':
                        fused_obs = np.mean(obs_flatten, axis=0)[None, :, :, :].repeat(batch_sample_num, 0)
                    elif self.fused_choice == 'random':
                        fused_obs = np.random.normal(loc=0, scale=0.1,
                                                            size=obs_flatten.shape[1:])[None, :, :, :].repeat(
                            batch_sample_num, 0)
                    elif self.fused_choice == 'blur':
                        fused_obs = cv2.GaussianBlur(obs_flatten[:, :, :, :], (5, 5),
                                                            cv2.BORDER_DEFAULT)
                    else:
                        fused_obs = np.zeros(obs_flatten.shape[1:])[None, :, :, :].repeat(batch_sample_num, 0)
                elif len(obs_flatten.shape) == 2:
                    if self.fused_choice == 'mean':
                        fused_obs = np.mean(obs_flatten, axis=0)
                    elif self.fused_choice == 'random':
                        fused_obs = np.random.normal(loc=0, scale=0.1, size=obs_flatten.shape[1:])
                    elif self.fused_choice == 'blur':
                        raise TypeError("Non-image observation does not blur...")
                    else:
                        fused_obs = np.zeros(obs_flatten.shape[1:])
                    fused_obs = fused_obs[None,].repeat(batch_sample_num, 0)
                else:
                    raise TypeError('Only support image or vector observation...')

                obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
                obs_flatten = torch.tensor(obs_flatten, dtype=torch.float32)
                fused_obs = torch.tensor(fused_obs, dtype=torch.float32)

                if self.step_model.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, obs_flatten, fused_obs, rewards = obs.cuda(), obs_flatten.cuda(), \
                                                           fused_obs.cuda(), rewards.cuda()

                if use_mask:
                    obs_exp = self.mask_model(obs_flatten, fused_obs, temp=self.temp, compute_obs_only=True)
                    obs = obs_exp.reshape(obs.shape)

                output, features = self.step_model.model(obs)  # marginal variational posterior, q(f|x).
                if self.step_model.weight_x:
                    output = self.step_model.likelihood(output, input_encoding=features)
                else:
                    output = self.step_model.likelihood(output) # y = E_{f_* ~ q(f_*)}[y|f_*])

                if self.likelihood_type == 'classification':
                    preds = output.probs.mean(0).argmax(-1)  # y = E_{f_* ~ q(f_*)}[y|f_*])
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    preds = output.mean
                    mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

            if self.step_model.likelihood_type == 'classification':
                preds_all = np.array(preds_all)
                rewards_all = np.array(rewards_all)
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
                acc = accuracy_score(rewards_all, preds_all)
                for cls in range(len(precision)):
                    print('Test results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                          format(cls, precision[cls], recall[cls], f1[cls], acc))
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
                print('Overall test results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(precision, recall, f1, acc))
            else:
                print('Test MSE: {}'.format(mse / float(test_idx.shape[0])))

            return 0

    # def get_explanations(self, class_id, normalize=True, model_path=None, load_model=False):
    #     """
    #     get explanation for a specific class
    #     :param class_id: the ID of a class
    #     :param normalize: normalize the explanation or not
    #     :param model_path: model path
    #     :param load_model: load model or not
    #     :return: step and mask explanation
    #     """

    #     if load_model:
    #         self.step_model, self.mask_model = self.load(model_path)

    #     self.step_model.eval()
    #     self.mask_model.eval()

    #     importance_all = self.step_model.likelihood.mixing_weights
    #     importance_all = importance_all.transpose(1, 0)

    #     importance_all = importance_all.cpu().detach().numpy()

    #     if importance_all.shape[-1] > 1:
    #         saliency = importance_all[:, class_id]
    #     else:
    #         saliency = np.squeeze(importance_all, -1)

    #     if normalize:
    #         saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-16)

    #     p = self.mask_model.compute_p()

    #     return saliency, p.cpu().detach().numpy()
    
    def get_explanations(self, class_id, normalize=True, model_path=None, load_model=False):
        """
        get explanation for a specific class
        :param class_id: the ID of a class
        :param normalize: normalize the explanation or not
        :param model_path: model path
        :param load_model: load model or not
        :return: step and mask explanation
        """

        if load_model:
            self.step_model, self.mask_model = self.load(model_path)

        self.step_model.eval()
        self.mask_model.eval()

        p = self.mask_model.compute_p()

        return p.cpu().detach().numpy()

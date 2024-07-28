import cv2
import tqdm
import math
import torch
import numpy as np


def concrete_transformation(theta, batch_size, temp=1.0 / 10.0, epsilon=torch.tensor(1e-10)):
    """ Use concrete distribution to approximate binary output
    :param theta: Bernoulli distribution parameters
    :param batch_size: size of samples
    :param temp: temperature
    :return: approximated binary output
    """
    unif_noise = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size, *theta.shape)))

    reverse_theta = torch.ones_like(theta) - theta
    reverse_unif_noise = torch.ones_like(unif_noise) - unif_noise

    log_alpha = torch.log(theta + epsilon) - torch.log(reverse_theta + epsilon)  # log(\theta/(1-\theta))
    log_u = torch.log(unif_noise + epsilon) - torch.log(reverse_unif_noise + epsilon)  # log(U/(1-U))

    log_u = log_u.to(log_alpha.device)

    logit = (log_u + log_alpha) / temp

    return torch.sigmoid(logit).type(torch.float32)


def gumble_transformation(theta, batch_size, temp=1.0 / 10.0, epsilon=torch.tensor(1e-8)):
    """ Use concrete distribution to approximate multiclass output
    :param theta: Multinoulli distribution parameters
    :param batch_size: size of samples
    :param temp: temperature
    :return: approximated binary output
    """
    unif_noise = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size, *theta.shape)))

    gumbel = - torch.log(- torch.log(unif_noise + epsilon) + epsilon)
    logit = (torch.log(theta + epsilon) + gumbel) / temp

    return torch.nn.Softmax(dim=-1)(logit).type(torch.float32)


def smoothness_loss(tensor, border_penalty=0.4):
    """
    :param tensor: input tensor with a shape of [W, H, C] and type of 'float'
    :param border_penalty: border penalty
    :return: loss value
    """
    x_loss = torch.sum((tensor[1:, :, :] - tensor[:-1, :, :]) ** 2)
    y_loss = torch.sum((tensor[:, 1:, :] - tensor[:, :-1, :]) ** 2)
    if border_penalty > 0:
        border = float(border_penalty) * (torch.sum(tensor[-1, :, :] ** 2 + tensor[0, :, :] ** 2) +
                                          torch.sum(tensor[:, -1, :] ** 2 + tensor[:, 0, :] ** 2))
    else:
        border = 0.
    return torch.mean(x_loss) + torch.mean(y_loss) + torch.mean(border)


def continuity_loss(tensor):
    """
    fused lass regularization
    :param tensor: tensor (dim,)
    :return: loss value
    """
    l_padded_mask = torch.cat([tensor[0:1], tensor])
    r_padded_mask = torch.cat([tensor, tensor[-1:]])
    continuity_loss = torch.mean(torch.sum(torch.abs(l_padded_mask - r_padded_mask)))
    return continuity_loss


def elasticnet_loss(tensor):
    loss_l1 = torch.sum(torch.abs(tensor))
    loss_l2 = torch.sqrt(torch.sum(torch.square(tensor)))
    return loss_l1 + loss_l2


class MaskObsModel(torch.nn.Module):
    def __init__(self, policy, act_distrubtion, input_shape, mask_shape, initializer='one', normalize_choice='sigmoid',
                 upsampling_mode='nearest', epsilon=1e-8):
        """
        :param policy: policy network
        :param act_distribution: action distribution "normal" or "cat"
        :param input_shape: shape of the input observation [c, w, h] or [d]
        :param mask_shape: shape of the explanation mask [c, w, h] or [d]
        :param require_hidden: whether the policy requires hidden states
        :param initializer: initializer to use ['zero', 'one', 'uniform', 'normal']
        :param normalize_choice: how to normalize the variable to between zero and one ['sigmoid', 'tanh', 'clip']
        :param upsampling_mode: upsampling mode ['nearest', 'bilinear']
        :param epsilon: same number to prevent numerical error
        """

        super(MaskObsModel, self).__init__()

        self.policy = policy
        self.act_distribution = act_distrubtion
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.initializer = initializer
        self.normalize_choice = normalize_choice
        self.upsampling_mode = upsampling_mode
        self.epsilon = torch.tensor(epsilon)

        # Remove the gradient in pretrained policy network
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False

        # Define the variable before normalization.
        if self.initializer == 'zero':
            print('Initializing logit p as all zeros.')
            tensor_logit_p = torch.zeros(mask_shape) + 1e-9
        elif self.initializer == 'one':
            print('Initializing logit p as all ones.')
            tensor_logit_p = torch.ones(mask_shape)
        elif self.initializer == 'uniform':
            print('Initializing logit p from uniform(-1, 1).')
            tensor_logit_p = torch.from_numpy(np.random.uniform(-1, 1, mask_shape))
        elif self.initializer == 'normal':
            print('Initializing logit p from N(0, 1).')
            tensor_logit_p = torch.randn(mask_shape)
        else:
            print('Initializing logit p as all zeros.')
            tensor_logit_p = torch.zeros(mask_shape)

        self.logit_p = torch.nn.Parameter(tensor_logit_p)

    def forward(self, obs, fused_obs, hidden_states=None, temp=0.1, compute_obs_only=False):
        """
        Compute the masked observations
        :param obs: input observations
        :param fused_obs: values to fill in the masked observations
        :param hidden_states: hidden_states (h_t, c_t)
        :param temp: temperature
        :return: masked observations and corresponding actions.
        """

        # normalize logit_p to between zero and one -> p.
        if self.normalize_choice == 'sigmoid':
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)
        elif self.normalize_choice == 'tanh':
            # print('Using tanh normalization.')
            p_mask_size = (torch.tanh(self.logit_p + torch.ones_like(self.logit_p))) / (2 + self.epsilon)
        elif self.normalize_choice == 'clip':
            # print('Using clip normalization.')
            p_mask_size = torch.clamp(self.logit_p, 0.0, 1.0).clone()
        else:
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)

        # resize variables
        if len(self.input_shape) > 1 and self.input_shape[1] != self.mask_shape[1]:
            # print('Upsampling the mask from %d to %d.' % (self.mask_shape[1], self.input_shape[1]))
            p = torch.nn.Upsample(size=self.input_shape[1:], mode=self.upsampling_mode)(p_mask_size[None,:])[0]
        else:
            p = p_mask_size

        batch_size = obs.shape[0]
        mask = concrete_transformation(p, batch_size, temp, self.epsilon)
        reverse_mask = torch.ones_like(mask) - mask

        # compute masked samples and reverse masked samples.
        obs_exp = torch.mul(obs, mask) + torch.mul(fused_obs, reverse_mask)

        # compute outputs and prepare the target label
        if compute_obs_only:
            return obs_exp
        else:
            obs_remain = torch.mul(obs, reverse_mask) + torch.mul(fused_obs, mask)
            acts_exp = self.policy(obs_exp)
            acts_remain = self.policy(obs_remain)
            return obs_exp, obs_remain, acts_exp, acts_remain

    def compute_p(self):
        # normalize logit_p to between zero and one -> p.
        if self.normalize_choice == 'sigmoid':
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)
        elif self.normalize_choice == 'tanh':
            # print('Using tanh normalization.')
            p_mask_size = (torch.tanh(self.logit_p + torch.ones_like(self.logit_p))) / (2 + self.epsilon)
        elif self.normalize_choice == 'clip':
            # print('Using clip normalization.')
            p_mask_size = torch.clamp(self.logit_p, 0.0, 1.0).clone()
        else:
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)

        # resize variables
        if len(self.input_shape) > 1 and self.input_shape[1] != self.mask_shape[1]:
            # print('Upsampling the mask from %d to %d.' % (self.mask_shape[1], self.input_shape[1]))
            p = torch.nn.Upsample(size=self.input_shape[1:], mode=self.upsampling_mode)(p_mask_size[None,:])[0]
        else:
            p = p_mask_size

        return p

    def get_visual_mask(self, temp=0.1):
         # normalize logit_p to between zero and one -> p.
        if self.normalize_choice == 'sigmoid':
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)
        elif self.normalize_choice == 'tanh':
            # print('Using tanh normalization.')
            p_mask_size = (torch.tanh(self.logit_p + torch.ones_like(self.logit_p))) / (2 + self.epsilon)
        elif self.normalize_choice == 'clip':
            # print('Using clip normalization.')
            p_mask_size = torch.clamp(self.logit_p, 0.0, 1.0).clone()
        else:
            # print('Using sigmoid normalization.')
            p_mask_size = torch.sigmoid(self.logit_p)

        # resize variables
        if len(self.input_shape) > 1 and self.input_shape[1] != self.mask_shape[1]:
            # print('Upsampling the mask from %d to %d.' % (self.mask_shape[1], self.input_shape[1]))
            p = torch.nn.Upsample(size=self.input_shape[1:], mode=self.upsampling_mode)(p_mask_size[None,:])[0]
        else:
            p = p_mask_size

        mask = concrete_transformation(p, 1, temp, self.epsilon)[0]
        
        return mask


class MaskFeatExp(object):
    def __init__(self, policy, act_distribution, input_shape, mask_shape, lr, require_hidden=True, initializer='one',
                 normalize_choice='sigmoid', upsampling_mode='nearest', epsilon=1e-8):
        """
        :param policy: policy network
        :param act_distribution: action distribution "normal" or "cat"
        :param input_shape: shape of the input observation
        :param mask_shape: shape of the explanation mask
        :param lr: learning rate
        :param require_hidden: require hidden states or not
        :param initializer: initializer to use ['zero', 'one', 'uniform', 'normal']
        :param normalize_choice: how to normalize the variable to between zero and one ['sigmoid', 'tanh', 'clip']
        :param upsampling_mode: upsampling mode ['nearest', 'bilinear']
        :param epsilon: same number to prevent numerical error
        """

        self.mask_model = MaskObsModel(policy, act_distribution, input_shape, mask_shape, initializer,
                                       normalize_choice, upsampling_mode, epsilon)

        self.optimizer = torch.optim.Adam({self.mask_model.logit_p}, lr=lr)

        if torch.cuda.is_available():
            self.mask_model = self.mask_model.cuda()
            self.mask_model.policy = self.mask_model.policy.cuda()

    def compute_loss(self, obs, fused_obs, acts, reg_choice, reg_coef_1, reg_coef_2, hidden=None,
                     temp=0.1, norm_choice='l2'):
        """
        Compute the loss value
        :param obs: input observations
        :param fused_obs: values to fill in the masked observations
        :param acts: true actions given by the policy network
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param temp: temperature
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :return: loss.
        """

        obs_exp, obs_remain, acts_exp, acts_remain = self.mask_model(obs[0:1,], fused_obs[0:1],
                                                                     (hidden[0][0], hidden[1][0]), temp)
        for i in range(obs.shape[0]-1):
            obs_exp_tmp, obs_remain_tmp, acts_exp_tmp, acts_remain_tmp = self.mask_model(obs[i+1:i+2], fused_obs[i+1:i+2],
                                                                                         (hidden[0][i+1], hidden[1][i+1]),
                                                                                         temp)
            obs_exp = torch.cat((obs_exp, obs_exp_tmp))
            obs_remain = torch.cat((obs_remain, obs_remain_tmp))
            acts_exp = torch.cat((acts_exp, acts_exp_tmp))
            acts_remain = torch.cat((acts_remain, acts_remain_tmp))

        # define and compute the network loss and regularization loss.
        if self.mask_model.act_distribution == 'cat':
            acts = acts.type(torch.long) - 1
            loss_func = torch.nn.CrossEntropyLoss()
            loss_exp = loss_func(acts_exp, acts)
            # loss_remain = -loss_func(acts_remain, acts)
        else:
            acts_exp_diff = acts - acts_exp
            acts_remain_diff = acts - acts_remain

            if norm_choice == 'l2':
                # print('Use MSE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=2))
                # loss_remain = -torch.mean(torch.linalg.vector_norm(acts_remain_diff, dim=1, ord=2))
            elif norm_choice == 'l1':
                # print('Use MAE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=1))
                # loss_remain = -torch.mean(torch.linalg.vector_norm(acts_remain_diff, dim=1, ord=1))
            elif norm_choice == 'inf':
                # print('Use infinity loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord=float('inf')))
                # loss_remain = -torch.mean(torch.linalg.vector_norm(acts_remain_diff, dim=1, ord=torch.inf))
            else:
                # print('Use MSE loss.')
                loss_exp = torch.mean(torch.linalg.norm(acts_exp_diff, dim=1, ord='2'))
                # loss_remain = -torch.mean(torch.linalg.vector_norm(acts_remain_diff, dim=1, ord='2'))

        if reg_choice == 'l1':
            # print('Using l1 regularization.')
            loss_reg_mask = torch.sum(torch.abs(self.mask_model.logit_p))
        elif reg_choice == 'elasticnet':
            # print('Using elasticnet regularization.')
            loss_reg_mask = elasticnet_loss(self.mask_model.logit_p)
        else:
            print('Using l1 regularization.')
            loss_reg_mask = torch.sum(torch.abs(self.mask_model.logit_p))

        if len(self.mask_model.logit_p.shape)==3:
            loss_smooth_mask = smoothness_loss(self.mask_model.logit_p)
        elif len(self.mask_model.logit_p.shape)==1:
            loss_smooth_mask = continuity_loss(self.mask_model.logit_p)
        else:
            raise TypeError('Only support image or vector observation...')

        # loss = loss_exp + loss_remain + reg_coef_1 * loss_reg_mask + reg_coef_2 * loss_smooth_mask
        loss = loss_exp + reg_coef_1 * loss_reg_mask + reg_coef_2 * loss_smooth_mask

        return loss, loss_exp, loss_reg_mask, loss_smooth_mask

    def get_params(self):
        return self.mask_model.compute_p().cpu().detach().numpy()

    def train(self, train_idx, traj_path, step_idx, batch_size, n_epochs, reg_choice, reg_coef_1, reg_coef_2, temp=0.1,
              norm_choice='l2', fused_choice='None', lambda_patience=10, lambda_multiplier=1.5, decay_weight=0.1,
              iteration_threshold=1e-4, early_stop_patience=10, display_interval=10):

        """
        Train the explanation parameters
        :param train_idx: training traj index
        :param traj_path: training traj path
        :param step_idx: important step_idx
        :param batch_size: training batch size
        :param decay_weight: learning rate decay weight
        :param n_epochs: training epochs
        :param reg_choice: regularization term to use ['l1', 'elasticnet']
        :param reg_coef_1: coefficient of the shape regularization
        :param reg_coef_2: coefficient of the smoothness regularization
        :param temp: temperature
        :param norm_choice: loss norm choice ['l2', 'l1', 'inf']
        :param fused_choice: values to fill in the masked part ['mean', 'random', 'blur']
        :param lambda_patience: lambda update patience
        :param lambda_multiplier: lambda update multiplier
        :param iteration_threshold: early stop count threshold
        :param early_stop_patience: early stop wait patience
        :param display_interval: print information interval
        :return: The solved explanation mask.
        """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                         gamma=decay_weight)

        total_train_samples = train_idx.shape[0] * step_idx.shape[0]
        batch_samples = batch_size * step_idx.shape[0]

        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
            n_batch = int(train_idx.shape[0] / batch_size) + 1

        print('********************************')
        print('Strat training...')
        # Check model parameters.
        # print(self.model.evaluate(obs))
        lambda_up_counter = 0
        lambda_down_counter = 0
        early_stop_counter = 0

        loss_last = 0
        loss_exp_last = 0
        loss_exp_best = math.inf
        best_p = self.get_params()
        for epoch in range(1, n_epochs + 1):
            loss = 0
            loss_exp = 0
            loss_sparsity = 0
            loss_smooth = 0

            for batch in tqdm.tqdm(range(n_batch)):
                batch_obs = []
                batch_acts = []
                batch_hs = []
                batch_cs = []

                for idx in train_idx[batch * batch_size:min((batch + 1) * batch_size, train_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'][step_idx,])
                    batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'][step_idx,])
                    # batch_hs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['h'][step_idx,])
                    # batch_cs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['c'][step_idx,])

                batch_obs = np.array(batch_obs)
                batch_obs = batch_obs.reshape(batch_samples, *batch_obs.shape[2:])

                batch_acts = np.array(batch_acts)
                batch_acts = batch_acts.reshape(batch_samples, *batch_acts.shape[2:])

                # batch_hs = np.array(batch_hs)
                # batch_hs = batch_hs.reshape(batch_samples, *batch_hs.shape[2:])

                # batch_cs = np.array(batch_cs)
                # batch_cs = batch_cs.reshape(batch_samples, *batch_cs.shape[2:])

                batch_hs = np.zeros((batch_samples, 1, 256))
                batch_cs = np.zeros((batch_samples, 1, 256))

                # Remove the padding state action pair where action=0
                nonzero_idx = np.unique(np.where(batch_acts != 0)[0])
                batch_obs_non_padding = batch_obs[nonzero_idx,]
                batch_acts = batch_acts[nonzero_idx,]
                batch_hs = batch_hs[nonzero_idx,]
                batch_cs = batch_cs[nonzero_idx,]

                obs = torch.tensor(batch_obs_non_padding, dtype=torch.float32)
                acts = torch.tensor(batch_acts, dtype=torch.float32)
                hs = torch.tensor(batch_hs, dtype=torch.float32)
                cs = torch.tensor(batch_cs, dtype=torch.float32)

                # todo: Support only one channel, add three channels support
                # Generate fused images
                if len(batch_obs.shape) == 4:
                    if fused_choice == 'mean':
                        fused_obs = np.mean(batch_obs, axis=0)[None, :, :, :].repeat(batch_samples, 0)
                    elif fused_choice == 'random':
                        fused_obs = np.random.normal(loc=0, scale=0.1, size=batch_obs.shape[1:])[None, :, :, :].repeat(batch_samples, 0)
                    elif fused_choice == 'blur':
                        fused_obs = cv2.GaussianBlur(batch_obs[:,:,:,:], (5, 5), cv2.BORDER_DEFAULT)
                    else:
                        fused_obs = np.zeros(batch_obs.shape[1:])[None, :, :, :].repeat(batch_samples, 0)
                elif len(batch_obs.shape) == 2:
                    if fused_choice == 'mean':
                        fused_obs = np.mean(batch_obs, axis=0)
                    elif fused_choice == 'random':
                        fused_obs = np.random.normal(loc=0, scale=0.1, size=batch_obs.shape[1:])
                    elif fused_choice == 'blur':
                        raise TypeError("Non-image observation does not blur...")
                    else:
                        fused_obs = np.zeros(batch_obs.shape[1:])
                    fused_obs = fused_obs[None, ].repeat(batch_samples, 0)
                else:
                    raise TypeError('Only support image or vector observation...')
                fused_obs = torch.tensor(fused_obs, dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, fused_obs, acts, hs, cs = obs.cuda(), fused_obs.cuda(), acts.cuda(), hs.cuda(), cs.cuda()

                self.optimizer.zero_grad()
                loss_batch, loss_exp_batch, loss_sparse_batch, loss_smooth_batch = \
                    self.compute_loss(obs, fused_obs, acts, reg_choice, reg_coef_1, reg_coef_2, (hs, cs),
                                      temp, norm_choice)
                loss_batch.backward()
                self.optimizer.step()
                loss += loss_batch.cpu().detach().numpy()
                loss_exp += loss_exp_batch.cpu().detach().numpy()
                loss_sparsity += loss_sparse_batch.cpu().detach().numpy()
                loss_smooth += loss_smooth_batch.cpu().detach().numpy()
                # Check model parameters.
                # print(self.model.evaluate(obs))

            loss = loss / float(n_batch)
            loss_exp = loss_exp / float(n_batch)
            loss_sparsity = loss_sparsity / float(n_batch)
            loss_smooth = loss_smooth / float(n_batch)

            # check cost modification
            if epoch > n_epochs/2 and loss_exp <= loss_exp_last:
                lambda_up_counter += 1
                if lambda_up_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 * lambda_multiplier
                    reg_coef_2 = reg_coef_2 * lambda_multiplier
                    lambda_up_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))
            elif epoch > n_epochs/2 and loss_exp > loss_exp_last:
                lambda_down_counter += 1
                if lambda_down_counter >= lambda_patience:
                    reg_coef_1 = reg_coef_1 / lambda_multiplier
                    reg_coef_2 = reg_coef_2 / lambda_multiplier
                    lambda_down_counter = 0
                    # print('Updating lambda1 to %.8f and lambda2 to %.8f'% (self.lambda_1, self.lambda_2))

            if (np.abs(loss - loss_last) < iteration_threshold) or \
                    (np.abs(loss_exp - loss_exp_last) < iteration_threshold):
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print('Reach the threshold and stop training at iteration %d/%d.' % (epoch, n_epochs))
                best_p = self.get_params()
                break

            loss_last = loss
            loss_exp_last = loss_exp

            if epoch % display_interval == 0:
                print("Epoch %d/%d: loss = %.5f explanation_loss = %.5f sparse_loss = %.5f smoothness_loss = %.5f "
                      % (epoch, n_epochs, loss, loss_exp, loss_sparsity, loss_smooth))

                if loss_exp_best > loss_exp:
                    loss_exp_best = loss_exp
                    best_p = self.get_params()
            scheduler.step()

        return best_p


class SaliencyFeatExp(object):
    def __init__(self, model):
        """
        :param model: policy network.
        """
        self.model = model

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def compute_gradient_input(self, obs):
        """
        compute gradient
        :param obs: input observations
        :return: the gradient of predicted actions w.r.t observations.
        """

        self.model.eval()
        obs.requires_grad_()

        if torch.cuda.is_available():
            obs = obs.cuda()
        pred_acts = self.model(obs)
        # todo: sum up the gradient dimensions [num_act, obs_dim]
        grad = torch.autograd.grad(torch.unbind(pred_acts), obs, retain_graph=True)[0]

        return grad.cpu()

    def get_explanations_by_tensor(self, obs, saliency_method='integrated_gradient', n_samples=2,
                                   stdev_spread=0.15, normalize=True):
        """
        Compute saliency explanation
        :param obs: input observations
        :param saliency_method: choice of saliency method
        :param n_samples: number of reference samples
        :param stdev_spread: std spread
        :param normalize: normalize the resulted gradient
        :return: time step importance.
        """

        if saliency_method == 'gradient':
            print('Using vanilla gradient.')
            saliency = self.compute_gradient_input(obs)

        elif saliency_method == 'integrated_gradient':
            print('Using integrated gradient.')
            baseline = torch.zeros_like(obs)
            assert baseline.shape == obs.shape
            x_diff = obs - baseline
            saliency = torch.zeros_like(obs)
            for alpha in np.linspace(0, 1, n_samples):
                x_step = baseline + alpha * x_diff
                grads = self.compute_gradient_input(x_step)
                saliency += grads
            saliency = saliency * x_diff

        elif saliency_method == 'unifintgrad':
            print('Using Unifintgrad.')
            baseline = torch.rand(obs.shape)
            baseline = (torch.max(obs) - torch.min(obs)) * baseline + torch.min(obs)
            assert baseline.shape == obs.shape
            x_diff = obs - baseline
            saliency = torch.zeros_like(obs)
            for alpha in np.linspace(0, 1, n_samples):
                x_step = baseline + alpha * x_diff
                grads = self.compute_gradient_input(x_step)
                saliency += grads
            saliency = saliency * x_diff

        elif saliency_method == 'smoothgrad':
            print('Using smooth gradient.')
            stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
            saliency = torch.zeros_like(obs)
            for x in range(n_samples):
                noise = torch.normal(0, stdev, obs.shape)
                noisy_data = obs + noise
                grads = self.compute_gradient_input(noisy_data)
                saliency = saliency + grads
            saliency = saliency / n_samples

        elif saliency_method == 'expgrad':
            print('Using Expgrad.')
            stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
            saliency = torch.zeros_like(obs)
            for x in range(n_samples):
                noise = torch.normal(0, stdev, obs.shape)
                noisy_data = obs + noise * torch.rand(1)[0]
                grads = self.compute_gradient_input(noisy_data)
                saliency = saliency + grads * noise

        elif saliency_method == 'vargrad':
            print('Using vargrad.')
            saliency = []
            stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
            for x in range(n_samples):
                noise = torch.normal(0, stdev, obs.shape)
                noisy_data = obs + noise
                grads = self.compute_gradient_input(noisy_data)
                saliency.append(grads[None, ...])

            saliency = torch.cat(saliency, dim=0)
            saliency = torch.var(saliency, dim=0)

        else:
            print('Using vanilla gradient.')
            saliency = self.compute_gradient_input(obs)

        saliency = saliency.cpu().detach().numpy()

        # todo: check the dimension.
        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) / \
                       (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None] + 1e-8)

        return saliency

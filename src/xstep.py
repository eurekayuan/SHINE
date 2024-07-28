import tqdm
import torch
import gpytorch
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils import CnnRnnEncoder, MlpRnnEncoder
from src.utils import DGPXRLModel, CustomizedGaussianLikelihood, CustomizedSoftmaxLikelihood, NNSoftmaxLikelihood


class DGaussianModel(torch.nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, input_channels, likelihood_type, encoder_type='MLP',
                 dropout_rate=0.25, num_class=None, rnn_cell_type='GRU', normalize=False):
        """ Reward prediction model
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param encoder_type: the encoder type
        :param dropout_rate: MLP dropout rate
        :param num_class: number of classes
        :param rnn_cell_type: the RNN cell type
        :param normalize: whether to normalize the input
        """

        super(DGaussianModel, self).__init__()

        self.encoder_type = encoder_type
        self.likelihood_type = likelihood_type

        if self.encoder_type == 'CNN':
            self.encoder = CnnRnnEncoder(seq_len, input_dim, input_channels, hiddens[-1], rnn_cell_type, normalize)
        else:
            self.encoder = MlpRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type, normalize)

        self.f_mean_net = torch.nn.Linear(hiddens[-1], 1)
        self.f_std_net = torch.nn.Linear(hiddens[-1], 1)

        self.mix_weight = torch.nn.Parameter(torch.randn(num_class, seq_len))
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, obs):
        """
        Compute the predicted reward
        :param obs: input observations
        :return: predicted reward.
        """

        obs_embedding, _ = self.encoder(obs) # (batch_size, seq_len, hidden_dim)
        mean = self.f_mean_net(obs_embedding).squeeze(-1) # (batch_size, seq_len)
        std = self.f_std_net(obs_embedding).squeeze(-1) # (batch_size, seq_len)

        f = self.gaussian_transformation(mean, std)
        output = f @ self.mix_weight.t()

        if self.likelihood_type == 'classification':
            pred = self.softmax(output) # y = E_{f_* ~ q(f_*)}[y|f_*]
        else:
            pred = output

        return pred

    @staticmethod
    def gaussian_transformation(mean, std):
        """
        sample v from N(0, 1) and compute f = mean + v * std -> f ~ N(mean, std^2)
        :param mean: mean vector
        :param std: std vector
        :return: f
        """
        v = torch.randn(mean.shape).to(mean.device)
        samples = mean + v * std
        return samples


class DGaussianStepExp(object):
    def __init__(self, seq_len, input_dim, hiddens, input_channels, likelihood_type, lr, encoder_type='MLP',
                dropout_rate=0.25, num_class=None, rnn_cell_type='GRU', normalize=False):

        """ reward prediction
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param lr: learning rate
        :param encoder_type: the encoder type
        :param dropout_rate: MLP dropout rate
        :param num_class: number of classes
        :param rnn_cell_type: the RNN cell type
        :param normalize: whether to normalize the input
        """
        self.seq_len = seq_len

        self.model = DGaussianModel(seq_len, input_dim, hiddens, input_channels, likelihood_type, encoder_type,
                                    dropout_rate, num_class, rnn_cell_type, normalize)

        self.optimizer = optim.Adam([
            {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
            {'params': self.model.f_mean_net.parameters()},
            {'params': self.model.f_std_net.parameters()},
            {'params': self.model.mix_weight}], lr=lr, weight_decay=0)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def compute_loss(self, prediction, target):
        """
        compute prediction loss
        :param prediction: predicted rewards
        :param target: original rewards
        :return: prediction loss and regularization loss
        """

        if self.model.likelihood_type == 'classification':
            loss = torch.nn.CrossEntropyLoss()(prediction, target)
        else:
            diff = target - prediction
            loss = torch.mean(torch.linalg.norm(diff, dim=1, ord=2))

        reg_loss = torch.sum(torch.abs(self.model.mix_weight))

        return loss, reg_loss

    def save(self, save_path):
        state_dict = self.model.state_dict()
        torch.save({'model': state_dict}, save_path)
        return 0

    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        self.model.load_state_dict(model_dict)
        return self.model

    def train(self, n_epoch, train_idx, batch_size, traj_path, reg_weight=0.01, decay_weight=0.1, save_path=None):
        """
        Training function
        :param n_epoch: training epoch
        :param train_idx: training traj index
        :param batch_size: training batch size
        :param traj_path: training traj path
        :param reg_weight: weight of the regularization loss
        :param decay_weight: lr decay weight
        :param save_path: model save path
        :return: trained model
        """

        self.model.train()

        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                   gamma=decay_weight)

        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
            n_batch = int(train_idx.shape[0] / batch_size) + 1

        for epoch in range(1, n_epoch + 1):
            print('{} out of {} epochs.'.format(epoch, n_epoch))
            mse = 0
            loss_sum = 0
            loss_reg_sum = 0
            preds_all = []
            rewards_all = []

            for batch in tqdm.tqdm(range(n_batch)):
                batch_obs = []
                batch_rewards = []
                for idx in train_idx[batch * batch_size:min((batch + 1) * batch_size, train_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                    batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

                obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)

                if self.model.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, rewards = obs.cuda(), rewards.cuda()

                self.optimizer.zero_grad()

                preds = self.model(obs)
                loss_pred, loss_reg = self.compute_loss(preds, rewards)
                loss = loss_pred + reg_weight * loss_reg

                loss.backward()
                self.optimizer.step()
                loss_sum += loss_pred.cpu().detach().numpy()
                loss_reg_sum += loss_reg.cpu().detach().numpy()

                if self.model.likelihood_type == 'classification':
                    preds = preds.argmax(-1)
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

            print('Prediction loss {}'.format(loss_sum/n_batch))
            print('Regularization loss {}'.format(loss_reg_sum/n_batch))

            if self.model.likelihood_type == 'classification':
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
                print('Train MSE: {}'.format(mse / float(train_idx.shape[0])))

            scheduler.step()
            self.save(save_path + '_epoch_' + str(epoch) + '.data')

        return self.model

    def test(self, test_idx, batch_size, traj_path):
        """ testing function
        :param test_idx: training traj index
        :param batch_size: training batch size
        :param traj_path: training traj path
        :return: prediction error
        """

        # Specify that the model is in eval mode.
        self.model.eval()

        mse = 0
        preds_all = []
        rewards_all = []

        if test_idx.shape[0] % batch_size == 0:
            n_batch = int(test_idx.shape[0] / batch_size)
        else:
            n_batch = int(test_idx.shape[0] / batch_size) + 1

        for batch in range(n_batch):
            batch_obs = []
            batch_rewards = []
            for idx in test_idx[batch * batch_size:min((batch + 1) * batch_size, test_idx.shape[0]), ]:
                batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

            obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)

            if self.model.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, rewards = obs.cuda(), rewards.cuda()

            preds = self.model(obs)

            if self.model.likelihood_type == 'classification':
                preds = preds.argmax(-1)
                preds_all.extend(preds.cpu().detach().numpy().tolist())
                rewards_all.extend(rewards.cpu().detach().numpy().tolist())
            else:
                mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

        if self.model.likelihood_type == 'classification':
            preds_all = np.array(preds_all)
            rewards_all = np.array(rewards_all)
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
            acc = accuracy_score(rewards_all, preds_all)
            for cls in range(len(precision)):
                print('Test results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(cls, precision[cls], recall[cls], f1[cls], acc))
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
            print('Overall testing results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                  format(precision, recall, f1, acc))
        else:
            print('Test MSE: {}'.format(mse / float(test_idx.shape[0])))

        return 0

    def get_explanations(self, class_id, normalize=True):
        """
        get explanation for a specific class
        :param class_id: the ID of a class
        :param normalize: normalize the explanation or not
        :return: explanation
        """

        importance_all = self.model.mix_weight
        importance_all = importance_all.transpose(1, 0)

        importance_all = importance_all.cpu().detach().numpy()

        if importance_all.shape[-1] > 1:
            saliency = importance_all[:, class_id]
        else:
            saliency = np.squeeze(importance_all, -1)

        if normalize:
            saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-16)

        return saliency

# 1. (Capturing the sequential dependency) Use a RNN (seq2seq) inside the RBF kernel function.
# 2. (Non-Gaussian likelihood) Use the Variational inference with the variational distribution as q(u)~N(\mu, LL^T).
# 3. Additive Gaussian process with two kernels - data (trajectory/individual) level and feature level.
# 4. Standard variational strategy with whitening and SoR.
#    (Faster way of computing p(f|u)) Use the inducing point technique with the SoR approximation of
#    p(f|u)=K_{x,z}K_{z,z}u and p(f_*|u)=K_{x_*,z}K_{z,z}u.
# 5. Standard variational strategy with SoR.
# 6. Natural Gradient Descent: Second-order optimization with Hessian replaced by Fisher Information matrix.
#    Optimization in the distribution space with the KL divergence as the distance measure. Better choice for
#    Variational distribution parameter.
# 7. Standard variational strategy with Contour Integral Quadrature to approximate K_{zz}^{-1/2},
#    Use it together with NGD.
# 8. Grid variational strategy with KSI.
# (Faster way of computing K_{x,z} in p(f|u)) KISS-GP: Use the local kernel interpolation technique
# (Structured kernel interpolation) to approximate K_{x,z}K_{z,z} with the interpolation matrix M.
# 9. Orthogonally decoupled VGPs. Using a different set of inducing points for the mean and covariance functions.
#    Use more inducing points for the mean and fewer inducing points for the covariance.


class DGPModel(torch.nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, input_channels, likelihood_type, num_inducing_points,
                 encoder_type='MLP', inducing_points=None, mean_inducing_points=None, dropout_rate=0.25,
                 num_class=None, rnn_cell_type='GRU', normalize=False, grid_bounds=None, using_ngd=False,
                 using_ksi=False, using_ciq=False, using_sor=False, using_OrthogonallyDecouple=False,
                 weight_x=False):
        """
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param num_inducing_points: number of inducing points
        :param encoder_type: encoder type ('MLP' or 'CNN')
        :param inducing_points: inducing points at the latent space Z (num_inducing_points, 2*hiddens[-1])
        :param mean_inducing_points: mean inducing points, used for orthogonally decoupled VGP
        :param dropout_rate: MLP dropout rate
        :param rnn_cell_type: the RNN cell type
        :param normalize: whether to normalize the input
        :param grid_bounds: grid bounds
        :param using_ngd: Whether to use natural gradient descent
        :param using_ksi: Whether to use KSI approximation, using this with other options as False
        :param using_ciq: Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD
        :param using_sor: Whether to use SoR approximation, not applicable for KSI and CIQ
        :param using_OrthogonallyDecouple
        :param weight_x: whether the mixing weights depend on inputs
        """

        super(DGPModel, self).__init__()

        self.likelihood_type = likelihood_type
        self.using_ngd = using_ngd
        self.weight_x = weight_x

        # Build the likelihood layer (Regression and classification).
        if self.likelihood_type == 'regression':
            print('Conduct regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len, weight_x=weight_x,
                                                           hidden_dims=2 * hiddens[-1])
        elif self.likelihood_type == 'classification':
            print('Conduct classification and use softmaxLikelihood')
            if self.weight_x:
                print('Varying the mixing weights with an ')
                self.likelihood = NNSoftmaxLikelihood(num_features=seq_len, num_classes=num_class,
                                                      input_encoding_dim=hiddens[-1] * 2)
            else:
                self.likelihood = CustomizedSoftmaxLikelihood(num_features=seq_len, num_classes=num_class)
        else:
            print('Default choice is regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len)

        # Compute the loss (ELBO) likelihood + KL divergence.
        self.model = DGPXRLModel(seq_len=seq_len, input_dim=input_dim, input_channels=input_channels,
                                 hiddens=hiddens, likelihood_type=likelihood_type, encoder_type=encoder_type,
                                 dropout_rate=dropout_rate, rnn_cell_type=rnn_cell_type, normalize=normalize,
                                 num_inducing_points=num_inducing_points, inducing_points=inducing_points,
                                 mean_inducing_points=mean_inducing_points, grid_bounds=grid_bounds,
                                 using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq,
                                 using_sor=using_sor, using_OrthogonallyDecouple=using_OrthogonallyDecouple)
        # print(self.model)

    def forward(self, obs):
        """
        Compute the predicted reward
        :param obs: input observations
        :return: predicted reward.
        """

        f_predicted, features = self.model(obs) # f_predicted: marginal variational posterior.

        if self.weight_x:
            output = self.likelihood(f_predicted, input_encoding=features)
        else:
            output = self.likelihood(
                f_predicted)  # y = E_{f_* ~ q(f_*)}[y|f_*].

        if self.likelihood_type == 'classification':
            pred = output.probs.mean(0).argmax(-1)  # y = E_{f_* ~ q(f_*)}[y|f_*].
        else:
            pred = output.mean

        return pred


class DGPStepExp(object):
    def __init__(self, train_len, seq_len, input_dim, hiddens, input_channels, likelihood_type, lr,
                 optimizer_type, n_epoch, gamma, num_inducing_points, encoder_type='MLP', inducing_points=None,
                 mean_inducing_points=None, dropout_rate=0.25, num_class=None, rnn_cell_type='GRU', normalize=False,
                 grid_bounds=None, using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                 using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.01):
        """
        :param train_len: training data length
        :param seq_len: trajectory length
        :param input_dim: input state/action dimension
        :param hiddens: hidden layer dimentions
        :param input_channels: input channels
        :param likelihood_type: likelihood type
        :param lr: learning rate
        :param optimizer_type
        :param n_epoch
        :param gamma
        :param num_inducing_points: number of inducing points
        :param encoder_type: encoder type ('MLP' or 'CNN')
        :param inducing_points: inducing points at the latent space Z (num_inducing_points, 2*hiddens[-1])
        :param mean_inducing_points: mean inducing points, used for orthogonally decoupled VGP
        :param dropout_rate: MLP dropout rate
        :param rnn_cell_type: the RNN cell type
        :param normalize: whether to normalize the input
        :param grid_bounds: grid bounds
        :param using_ngd: Whether to use natural gradient descent
        :param using_ksi: Whether to use KSI approximation, using this with other options as False
        :param using_ciq: Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD
        :param using_sor: Whether to use SoR approximation, not applicable for KSI and CIQ
        :param using_OrthogonallyDecouple
        :param weight_x: whether the mixing weights depend on inputs
        :param lambda_1: coefficient before the lasso/local linear regularization (here we time it to lr)
        """

        self.train_len = train_len
        self.likelihood_type = likelihood_type
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.n_epoch = n_epoch
        self.gamma = gamma
        self.using_ngd = using_ngd
        self.weight_x = weight_x

        # Build the likelihood layer (Regression and classification).
        if self.likelihood_type == 'regression':
            print('Conduct regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len, weight_x=weight_x, hidden_dims=2*hiddens[-1])
        elif self.likelihood_type == 'classification':
            print('Conduct classification and use softmaxLikelihood')
            if self.weight_x:
                print('Varying the mixing weights with an ')
                self.likelihood = NNSoftmaxLikelihood(num_features=seq_len, num_classes=num_class,
                                                      input_encoding_dim=hiddens[-1]*2)
            else:
                self.likelihood = CustomizedSoftmaxLikelihood(num_features=seq_len, num_classes=num_class)
        else:
            print('Default choice is regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len)

        # Compute the loss (ELBO) likelihood + KL divergence.
        self.model = DGPXRLModel(seq_len=seq_len, input_dim=input_dim, input_channels=input_channels, hiddens=hiddens,
                                 likelihood_type=likelihood_type, encoder_type=encoder_type, dropout_rate=dropout_rate,
                                 rnn_cell_type=rnn_cell_type, normalize=normalize, num_inducing_points=num_inducing_points,
                                 inducing_points=inducing_points, mean_inducing_points=mean_inducing_points,
                                 grid_bounds=grid_bounds, using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq,
                                 using_sor=using_sor, using_OrthogonallyDecouple=using_OrthogonallyDecouple)
        # print(self.model)

        # First, sampling from q(f) with shape [n_sample, n_data].
        # Then, the likelihood function times it with the mixing weight and get the marginal likelihood p(y|f).
        # VariationalELBO will call _ApproximateMarginalLogLikelihood, which then computes the marginal likelihood by
        # calling the likelihood function (the expected_log_prob in the likelihood class)
        # and the KL divergence (VariationalStrategy.kl_divergence()).
        # ELBO = E_{q(f)}[p(y|f)] - KL[q(u)||p(u)].
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer,
                                                 num_data=self.train_len)

        # Define the optimizer over the parameters.
        # (RNN parameters, RBF kernel parameters, Z, variational parameters, mixing weight).
        if self.using_ngd:
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.model.gp_layer.variational_parameters(),
                                                                num_data=self.train_len, lr=self.lr*10)
            if self.optimizer_type == 'adam':
                self.hyperparameter_optimizer = optim.Adam([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            else:
                self.hyperparameter_optimizer = optim.SGD([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            # Learning rate decay schedule.
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.variational_ngd_optimizer,
                                                            milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                            gamma=self.gamma)
            self.scheduler_hyperparameter = optim.lr_scheduler.MultiStepLR(self.hyperparameter_optimizer,
                                                                           milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                                           gamma=self.gamma)
        else:
            if self.optimizer_type == 'adam':
                self.optimizer = optim.Adam([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.model.gp_layer.variational_parameters()},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            else:
                self.optimizer = optim.SGD([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.model.gp_layer.variational_parameters()},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr,
                    momentum=0.9, nesterov=True, weight_decay=0)

            # Learning rate decay schedule.
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                            gamma=self.gamma)

        self.likelihood_regular_optimizer = optim.Adam([{'params': self.likelihood.parameters()}],
                                                       lr=self.lr * lambda_1, weight_decay=0) # penalize the lr with lambda_1.

        self.scheduler_regular = optim.lr_scheduler.MultiStepLR(self.likelihood_regular_optimizer,
                                                                milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                                gamma=self.gamma)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def save(self, save_path):
        state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, save_path)
        return 0

    # Load a pretrained model.
    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        likelihood_dict = dicts['likelihood']
        self.model.load_state_dict(model_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        return self.model, self.likelihood

    def train(self, train_idx, batch_size, traj_path, save_path=None, likelihood_sample_size=8):
        """
        Training function
        :param train_idx: training traj index
        :param batch_size: training batch size
        :param traj_path: training traj path
        :param save_path: model save path
        :param likelihood_sample_size:
        :return: trained model
        """

        self.model.train()
        self.likelihood.train()

        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
            n_batch = int(train_idx.shape[0] / batch_size) + 1

        best_acc = 0
        for epoch in range(1, self.n_epoch + 1):
            print('{} out of {} epochs.'.format(epoch, self.n_epoch))
            mse = 0
            mae = 0
            loss_sum = 0
            loss_reg_sum = 0
            preds_all = []
            rewards_all = []
            with gpytorch.settings.use_toeplitz(False):
                with gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
                    for batch in tqdm.tqdm(range(n_batch)):
                        batch_obs = []
                        batch_rewards = []
                        for idx in train_idx[batch * batch_size:min((batch + 1) * batch_size, train_idx.shape[0]), ]:
                            batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                            batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

                        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)

                        if self.likelihood_type == 'classification':
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                        else:
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                        if torch.cuda.is_available():
                            obs, rewards = obs.cuda(), rewards.cuda()

                        if self.using_ngd:
                            self.variational_ngd_optimizer.zero_grad()
                            self.hyperparameter_optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()

                        output, features = self.model(obs)  # marginal variational posterior, q(f|x).

                        if self.weight_x:
                            loss = -self.mll(output, rewards, input_encoding=features)  # approximated ELBO.
                        else:
                            loss = -self.mll(output, rewards)  # approximated ELBO.

                        loss.backward()

                        if self.using_ngd:
                            self.variational_ngd_optimizer.step()
                            self.hyperparameter_optimizer.step()
                        else:
                            self.optimizer.step()

                        loss_sum += loss.item()

                        self.likelihood_regular_optimizer.zero_grad()
                        if self.weight_x and self.likelihood_type == 'classification':
                            # lasso.
                            output, features = self.model(obs)  # marginal variational posterior, q(f|x).
                            features_sum = features.detach().sum(-1)
                            weight_output = self.likelihood.weight_encoder(features_sum)
                            lasso_term = torch.norm(weight_output, p=1) # lasso
                            lasso_term.backward()
                            loss_reg_sum += lasso_term
                        else:
                            lasso_term = torch.norm(self.likelihood.mixing_weights, p=1) # lasso
                            lasso_term.backward()
                            loss_reg_sum += lasso_term
                            self.likelihood_regular_optimizer.step()

                        if self.weight_x:
                            output = self.likelihood(output, input_encoding=features)
                        else:
                            output = self.likelihood(output) # y = E_{f_* ~ q(f_*)}[y|f_*])

                        if self.likelihood_type == 'classification':
                            preds = output.probs.mean(0).argmax(-1)  # y = E_{f_* ~ q(f_*)}[y|f_*])
                            preds_all.extend(preds.cpu().detach().numpy().tolist())
                            rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                        else:
                            preds = output.mean
                            mae += torch.sum(torch.abs(preds - rewards)).cpu().detach().numpy()
                            mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

                    print('ELBO loss {}'.format(loss_sum/n_batch))
                    print('Regularization loss {}'.format(loss_reg_sum/n_batch))

                    if self.likelihood_type == 'classification':
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
                        print('Train MAE: {}'.format(mae / float(self.train_len)))
                        print('Train MSE: {}'.format(mse / float(self.train_len)))

            if self.using_ngd:
                self.scheduler_hyperparameter.step()

            self.scheduler.step()
            self.scheduler_regular.step()
            if self.likelihood_type == 'classification':
                if acc > best_acc:
                    best_acc = acc
                    self.save(save_path)
            else:
                self.save(save_path + '_' + str(epoch) + '_model.data')

        return self.model

    def test(self, test_idx, batch_size, traj_path, likelihood_sample_size=16):
        """
        Testing function
        :param test_idx: training traj index
        :param batch_size: training batch size
        :param traj_path: training traj path
        :param likelihood_sample_size:
        :return: prediction error
        """

        # Specify that the model is in eval mode.
        self.model.eval()
        self.likelihood.eval()

        mse = 0
        mae = 0
        preds_all = []
        rewards_all = []

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

                obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)

                if self.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, rewards = obs.cuda(), rewards.cuda()

                f_predicted, features = self.model(obs)
                if self.weight_x:
                    output = self.likelihood(f_predicted, input_encoding=features)
                else:
                    output = self.likelihood(f_predicted)

                if self.likelihood_type == 'classification':
                    preds = output.probs.mean(0).argmax(-1)
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    preds = output.mean.detach()
                    mae += torch.sum(torch.abs(preds - rewards)).cpu().detach().numpy()
                    mse += torch.sum(torch.square(preds - rewards)).cpu().detach().numpy()

        if self.likelihood_type == 'classification':
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
            print('Test MAE: {}'.format(mae / float(test_idx.shape[0])))
            print('Test MSE: {}'.format(mse / float(test_idx.shape[0])))

        return 0

    def get_explanations(self, class_id, normalize=True):
        """
        get explanation for a specific class
        :param class_id: the ID of a class
        :param normalize: normalize the explanation or not
        :return: explanation
        """

        importance_all = self.likelihood.mixing_weights
        importance_all = importance_all.transpose(1, 0)

        importance_all = importance_all.cpu().detach().numpy()

        if importance_all.shape[-1] > 1:
            saliency = importance_all[:, class_id]
        else:
            saliency = np.squeeze(importance_all, -1)

        if normalize:
            saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-16)

        return saliency

    def get_explanations_per_traj(self, exp_idx, batch_size, traj_path, normalize=True):
        """
        get explanation for each input traj
        :param exp_idx: training traj index
        :param batch_size: training batch size
        :param traj_path: training traj path
        :param normalize: normalize
        :return: time step importance
        """

        self.model.eval()
        self.likelihood.eval()
        n_batch = int(exp_idx.shape[0] / batch_size)
        final_rewards = []

        for batch in range(n_batch):
            batch_obs = []
            batch_rewards = []
            for idx in exp_idx[batch * batch_size:(batch + 1) * batch_size, ]:
                batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                batch_rewards.append(int(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards']))

            final_rewards += batch_rewards
            obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)

            if self.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs = obs.cuda()

            step_embedding, traj_embedding = self.model.encoder(obs)  # (N, T, P) -> (N, T, D), (N, D).
            traj_embedding = traj_embedding[:, None, :].repeat(1, obs.shape[1], 1)  # (N, D) -> (N, T, D)
            features = torch.cat([step_embedding, traj_embedding], dim=-1)  # (N, T, 2D)
            # features = features.view(obs.size(0) * obs.size(1), features.size(-1))
            covar_all = self.model.gp_layer.covar_module(features)
            covar_step = self.model.gp_layer.step_kernel(features)
            covar_traj = self.model.gp_layer.traj_kernel(features)

            if self.weight_x:
                input_encoding = features.sum(-1)
                importance_all = self.likelihood.weight_encoder(input_encoding)
                importance_all = importance_all.reshape(importance_all.shape[0], self.likelihood.num_features,
                                                        self.likelihood.num_classes)
            else:
                importance_all = self.likelihood.mixing_weights
                importance_all = importance_all.transpose(1, 0)

            importance_all = importance_all.cpu().detach().numpy()

            if len(importance_all.shape) == 2:
                importance_all = np.repeat(importance_all[None, ...], rewards.shape[0], axis=0)

            if importance_all.shape[-1] > 1:
                importance = importance_all[list(range(rewards.shape[0])), :, rewards]
            else:
                importance = np.squeeze(importance_all, -1)

            if batch == 0:
                saliency_all = importance
                covar_all_all = covar_all.numpy()[None, ...]
                covar_step_all = covar_step.numpy()[None, ...]
                covar_traj_all = covar_traj.numpy()[None, ...]
            else:
                saliency_all = np.vstack((saliency_all, importance))
                covar_all_all = np.concatenate((covar_all_all, covar_all.numpy()[None, ...]))
                covar_step_all = np.concatenate((covar_step_all, covar_all.numpy()[None, ...]))
                covar_traj_all = np.concatenate((covar_traj_all, covar_all.numpy()[None, ...]))

        if normalize:
            saliency_all = (saliency_all - np.min(saliency_all, axis=1)[:, None]) \
                         / (np.max(saliency_all, axis=1)[:, None] - np.min(saliency_all, axis=1)[:, None] + 1e-16)

        return saliency_all, (covar_all_all, covar_traj_all, covar_step_all), final_rewards


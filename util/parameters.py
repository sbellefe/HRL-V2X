import math
import torch as th
from torch.nn import functional as F

class SharedParams:
    def __init__(self):
        super(SharedParams, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        """global hyperparams""" #TODO add testing params
        self.num_trials = 5
        self.total_train_episodes = 5000 #100000  # number of control episodes
        self.t_max = 10        # max timesteps (communication intervals) per control interval
        self.k_max = 10                 #number of control intervals per episode (AoI only)
        self.num_agents = 4
        self.test_interval = 100
        self.test_episodes = 10

        """global environment parameters"""
        self.multi_location = True
        self.fast_fading = True
        self.include_AoI = False
        self.partial_observability = False
        self.single_loc_idx = 25.0 #only used for NFIG, SIG_SL
        self.multi_loc_test_idx = range(35, 45)   #only used for SIG_ML, POSIG


class ParametersMAPPO(SharedParams):
    def __init__(self):
        super(ParametersMAPPO, self).__init__()

        # training loop hyperparameters
        self.buffer_episodes = 32  # or "batch_size" num episodes in batch buffer
        self.opt_epochs = 10    #num optimization epochs per batch buffer
        # self.mini_batch_size = 320
        self.num_mini_batches = 1
        self.train_iterations = math.ceil(self.total_train_episodes / self.buffer_episodes) #top-lvl loop index

        # network hyperparameters
        self.actor_hidden_dim = (128, 128)
        self.critic_hidden_dim = (128, 128)
        self.lr_actor = 5e-4 #3e-4
        self.lr_critic = 5e-4#1e-3

        # training value hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.99
        self.entropy_coef = 0.001
        self.eps_clip = 0.2

class ParametersDAC(SharedParams):
    def __init__(self):
        super(ParametersDAC, self).__init__()

        #Network hidden dimensions
        dim = 64
        self.feature_dim = 64
        self.option_hidden_units = (dim, dim) #hidden neurons for sub-policy and beta networks
        self.actor_hidden_units = (dim, dim) #hidden neurons for master policy network
        self.critic_hidden_units = (dim, dim) #hidden neurons for critic network

        #hidden neuron activation functions lambda x: F.relu(x) or F.tanh(x)
        self.pi_l_activation = lambda x: F.tanh(x)     #option policies
        self.beta_activation = lambda x: F.tanh(x)     #option termination
        self.pi_h_activation = lambda x: F.tanh(x)     #master policy
        self.critic_activation = lambda x: F.tanh(x)     #critic

        # training loop hyperparameters
        self.num_options = 4
        self.buffer_episodes = 5  # num episodes in batch buffer
        self.opt_epochs = 5  # num optimization epochs per batch buffer per mdp
        self.mini_batch_size = 64
        self.train_iterations = math.ceil(self.total_train_episodes / self.buffer_episodes)  # top-lvl loop index

        # training value hyperparameters
        self.lr_ha = 3e-4       #high actor (pi_W) learning rate
        self.lr_la = 3e-4       #low actor (pi_w) learning rate
        self.lr_critic = 1e-3       #critic learning rate
        self.lr_beta = 1e-4     #beta network learning rate
        self.lr_phi = 5e-4      #shared feature network learning rate
        self.eps_clip = 0.2     #ppo clipping parameter
        self.gamma = 0.99       #discount
        self.gae_lambda = 0.99      #GAE smoothing param
        self.entropy_coef_h = 0.01  #high MDP exploration entropy coefficient
        self.entropy_coef_l = 0.01  #low MDP exploration entropy coefficient


class ParametersOC(SharedParams):
    def __init__(self):
        super(ParametersOC, self).__init__()

        # training loop hyperparameters
        self.buffer_size = 20000 #max timesteps stored in buffer
        self.batch_size = 32        #minibatch size for critic optimization
        self.target_update_freq = 1000    #number of timesteps between hard critic updates
        self.critic_optim_freq = 4       #number of timesteps between critic SGD optimizations

        # training value and network hyperparameters
        self.hidden_dim = (64,64)   #hidden neurons for q, beta, subpolicy networks
        self.feature_dim = 64    #shared state representation
        self.lr_q = 1e-3    #Q network (critic + master policy) learning rate
        self.lr_beta = 1e-4  # beta network learning rate
        self.lr_la = 3e-4  # low actor (sub-policies) learning rate
        self.lr_phi = 5e-4  # feature network learning rate
        self.gamma = 0.99
        self.beta_reg = 0.01    #Regularization to decrease termination prob
        self.entropy_coef = 0.01 #Regularization to increase policy entropy
        self.num_options = 4
        self.temp = 1.0   #Action distribution softmax tempurature param (1 does nothing)
        self.gradient_clip = 0.5  # gradient clip norm for updates

        # exploration epsilon decay from 'start' to 'end' in 'decay' timesteps
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 20000
        self.eps_test = 0.005

        self.t_tot = 0  # timestep counter for epsilon calculation
        """"""

    @property
    def epsilon(self):
        """dynamically compute epsilon based on current step as params attribute"""
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  math.exp(-1. * self.t_tot / self.eps_decay)
        return epsilon
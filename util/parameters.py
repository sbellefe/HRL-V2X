import math
import torch as th
from torch.nn import functional as F
from Envs.env_params import V2Xparams #TODO: migrate relevant parts here

class SharedParams(V2Xparams):
    def __init__(self):
        super(SharedParams, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        """global hyperparams"""
        self.num_trials = 5
        self.total_train_episodes = 20000 #100000  # number of control episodes
        self.t_max = 10             # maximum number of time for control
        self.num_agents = 1

        """global environment parameters"""
        self.multi_location = True
        self.fast_fading = True
        self.single_loc_idx = 25.0    #only used for NFIG, SIG_SL
        self.multi_loc_idx = [35, 45]
        #TODO: Make game_mode automatically calculated ??
        self.game_mode = 2 # 1:Chanel only, 2:++Queue, 3:++AoI, 4:POSIG


        # self.num_veh = self.num_agents * 2





        self.t_max_control = 120
        self.test_interval = 1000  # test every 10 episodes
        self.test_episodes = 10  # test 10 episodes and get average results

class ParametersV2X_N:
    def __init__(self):

        self.num_agents = 1     #number of V2V links
        self.t_max_control = 10  # control interval length (T)


        #FROM MARL env_params (not all included yet)
        # Driving Scenario Parameters
        self.n_veh = 0                          # number of vehicles in total
        self.n_veh_platoon = [2, 2, 2, 2]       # number of vehicles in each platoon
        self.n_lane = 1                         # number of lanes in the platoon
        self.number_of_SC = 4                   # number of sub-channels (SCs) for V2X communications

        # Control Parameters
        self.nb_episodes_control = 100000   # number of control episodes
        # self.t_max_control = 10             # maximum number of time for control
        self.nb_episodes_Test = 2           # number of Test episodes
        self.NQ = 1                         # number of buffer size at each agent
        self.CAM_size = 25600               # number of bits per cooperative awareness message (''CAM), in bits'
        self.gamma_control = 0.99979        # discount factor of CL part
        self.CL_pretrained = False          # using locally pretrained model
        self.AoI_max = 6                    # maximum value of AoI at each agent
        self.track_er_position_init = 1.5   # initialization of tracking position error
        self.track_er_velocity_init = -1.0  # initialization of tracking velocity error
        self.er_position_norm_range = 2.0   # normalization of position tracking error
        self.er_velocity_norm_range = 1.5   # normalization of velocity tracking error
        self.acceleration_norm_range = 2.6  # normalization of acceleration

        # Communication Parameters
        self.game_mode = 2  # 1: channel aware. 2: channel and queue aware. 3: channel, queue and AoI aware. 4: partial observability
        self.n_step_per_episode_communication = 10  # number of communication time intervals within each control time unit
        self.BW_per_SC = 1000000  # bandwidth of each sub-channel, in Hz
        self.reward_w1 = 0.001  # reward weight corresponding to V2I
        self.reward_w2 = 0.1  # reward weight corresponding to V2V data rate
        self.reward_w3 = 1  # reward weight corresponding to AoI
        self.reward_VoI = 100.0  # reward weight corresponding to V2V data rate
        self.reward_G = 5.0  # reward weight corresponding to V2V data rate
        self.QMS = 2  # queue management strategies (QMS) = 1 OR 2


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
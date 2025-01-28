import math
import torch as th

class ParametersPPO:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # training loop hyperparameters
        self.num_trials = 5
        self.total_train_episodes = 1000
        self.buffer_episodes = 10  # num episodes in batch buffer
        self.t_max = 1000    # max episode length
        self.opt_epochs = 10
        self.mini_batch_size = 64
        self.train_iterations = math.ceil(self.total_train_episodes / self.buffer_episodes) #top-lvl loop index
        self.test_interval = 10  # test every 10 episodes
        self.test_episodes = 10  # test 10 episodes and get average results

        # training value hyperparameters
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.99
        self.entropy_coef = 0.01
        self.eps_clip = 0.2
import sys
import numpy as np
import torch as th
from torch import nn
from torch.fx.passes.infra.pass_base import PassResult
from torch.nn import functional as F
from torch.distributions import Categorical

from helpers.dac_helper import pre_process

class DAC_Actor(nn.Module):
    def __init__(self, params, mdp):
        super(DAC_Actor, self).__init__()
        self.mdp = mdp
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_options = params.num_options
        self.eps_clip = params.eps_clip

        if self.mdp == 'high':
            self.fc1 = nn.Linear(self.state_dim, params.actor_hidden_dim)
            self.fc2 = nn.Linear(params.actor_hidden_dim, params.actor_hidden_dim)
            self.fc3 = nn.Linear(params.actor_hidden_dim, self.num_options)
        elif self.mdp == 'low':
            self.fc1 = nn.Linear(self.state_dim+1, params.actor_hidden_dim)
            self.fc2 = nn.Linear(params.actor_hidden_dim, params.actor_hidden_dim)
            self.fc3 = nn.Linear(params.actor_hidden_dim, self.action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        x = state
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def actor_loss(self,  logp, old_logp, advantages):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(logp - old_logp)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        # Calculate the minimum surrogate loss for clipping
        loss = -th.min(surr1, surr2).mean()
        return loss

    def get_state(self, *args):
        if self.mdp == 'high':
            """high MDP state is raw state"""
            obs, = args
            state_h = pre_process(obs)

            return state_h
        elif self.mdp == 'low':
            obs, option = args

            # print(f"option: {option.shape}\n"
            #       f"obs: {obs.shape}\n\n")
            # sys.exit()

            obs_l = np.concatenate((obs, np.array(option)))
            state_l = pre_process(obs_l)
            # print(f"option: {option.shape}\n"
            #       f"obs: {obs.shape}\n"
            #       f"state_l: {state_l.shape}\n")
            return state_l
        else: raise NotImplementedError

    def get_policy(self, *args):
        """compute policy, action/option, logp, for high or low MDP
                args:
                    High MDP: state_h, beta, prev_option
                        Return: option, logp_high, pi_hat
                    Low MDP: state_l
                        Return: action, logp_l, pi_bar
                 """
        if self.mdp == 'high':
            #unpack args and pass state through network
            state_h, beta, prev_option = args
            logits = self(state_h)


            #compute pi_hat by factoring in beta termination
            probs = F.softmax(logits, dim=-1)
            mask = th.zeros_like(probs)
            mask[0, prev_option[0]] = 1
            pi_hat = (1 - beta) * mask + beta * probs

            #sample option from pi_hat, compute log prob
            dist = Categorical(probs=pi_hat)
            option = dist.sample()
            logp_h = dist.log_prob(option)

            return option, logp_h, pi_hat
        elif self.mdp == 'low':
            #unpack args and pass state through network
            state_l, = args

            logits = self(state_l)

            #sample action from distribution, compute log prob
            pi_bar = Categorical(logits=logits)
            action = pi_bar.sample()
            logp_l = pi_bar.log_prob(action)

            return action, logp_l, pi_bar



class DAC_Critic(nn.Module):
    def __init__(self, params):
        super(DAC_Critic, self).__init__()
        self.state_dim = params.state_dim
        self.num_options = params.num_options
        self.eps_clip = params.eps_clip

        self.fc1 = nn.Linear(self.state_dim+1, params.actor_hidden_dim)
        self.fc2 = nn.Linear(params.critic_hidden_dim, params.critic_hidden_dim)
        self.fc3 = nn.Linear(params.critic_hidden_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        x = state
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

    # def get_state(self, state_h, option):
    # #     state_l = np.concatenate((state, np.array(option)))
    # #     print(f"option: {option}\n"
    # #           f"obs: {obs.shape}\n")
    # obs_l = np.concatenate((obs, np.array(option)))
    # state_l = pre_process(obs_l)
    #
    #     obs_l = np.concatenate((obs, np.array(option)))
    #     state_l = pre_process(obs_l)
    #     return state_l

    def get_values(self, state_l, pi_hat):
        #critic estimates low MDP value function
        v_l = self(state_l)
        state_h = state_l[:, :-1]
        batch_size = state_l.shape[0]

        #compute high MDP value function
        v_h = 0
        for op in range(self.num_options):
            # state_l_temp = self.get_state(obs, th.tensor([op]))
            # print(f"state_h: {state_h.shape}\n"
            #       f"state_l: {state_l.shape}\n"
            #       f"op: {th.tensor([[op]]).shape}\n"
            #       f"option: {th.full((state_h.size(0), 1), op, dtype=th.long).shape}")
            option = th.full((batch_size, 1), op, dtype=th.long)
            state_l_temp = th.cat((state_h, option), dim=1)
            v_l_temp = self(state_l_temp)
            v_h += v_l_temp * pi_hat[0][op]
        return v_h, v_l

    def critic_loss(self, values, old_values, returns):
        # value_clip = old_values + th.clamp(values - old_values, -eps_clip, eps_clip)
        value_clip = th.clamp(values, old_values - self.eps_clip, old_values + self.eps_clip)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss

class DAC_Beta(nn.Module):
    def __init__(self, params):
        super(DAC_Beta, self).__init__()
        self.state_dim = params.state_dim
        self.eps_clip = params.eps_clip

        self.fc1 = nn.Linear(self.state_dim+1, params.beta_hidden_dim)
        self.fc2 = nn.Linear(params.beta_hidden_dim, params.beta_hidden_dim)
        self.fc3 = nn.Linear(params.beta_hidden_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        beta = th.sigmoid(self.fc3(x))
        return beta

    def get_state(self, next_obs, option):
        """state beta is next_obs augmented with current option"""
        obs_beta = np.concatenate((next_obs, np.array(option)))
        state_beta = pre_process(obs_beta)
        return state_beta

    def beta_loss(self,state_beta_batch, beta_value_batch, advantage_batch):
        '''
               beta_value_batch_clone = beta_value_batch.clone().detach()
               #beta_value_batch_clamp = torch.clamp(beta_value_batch_clone, min=0.0, max=1.0)
               # loss = - beta * advantage
               loss_1 = torch.mean(beta_value_batch_clone)
               #loss_2 = -torch.mean(beta_value_batch.detach() * advantage_batch.detach())
               return loss_1
               '''
        beta_value_batch = self(state_beta_batch)  # re-generate logits by the current sampled state
        # loss_1 = torch.mean(beta_value_batch)
        loss = -th.mean(beta_value_batch * advantage_batch)
        return loss
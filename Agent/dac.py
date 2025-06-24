import sys
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

class DAC_SingleOptionNet(nn.Module):
    """option policy and termination network for a single option"""
    def __init__(self, params):
        super(DAC_SingleOptionNet,self).__init__()
        self.gate = params.pi_l_activation
        input_dim = params.state_dim + params.num_agents
        hidden = params.option_hidden_units
        output_dim = params.action_dim

        #define option policy network
        self.pi_fc1 = nn.Linear(input_dim, hidden[0])
        self.pi_fc2 = nn.Linear(hidden[0], hidden[1])
        self.pi_fc3 = nn.Linear(hidden[1], output_dim)

    def forward(self, x):
        """accepts raw state tensor as input"""
        # x = phi
        #pass through option policy network
        x_pi = self.gate(self.pi_fc1(x))
        x_pi = self.gate(self.pi_fc2(x_pi))
        logits_pi = self.pi_fc3(x_pi)
        pi_w = F.softmax(logits_pi, dim=-1)
        return pi_w #Shape: [batch_size, action_dim]


class DAC_Actors(nn.Module):
    """'Actors' = high-policy, low-policy, beta """
    def __init__(self, params):
        super(DAC_Actors, self).__init__()
        #store used parameters
        self.pi_gate = params.pi_h_activation
        self.beta_gate = params.beta_activation
        self.eps_clip = params.eps_clip
        input_dim = params.state_dim + params.num_agents
        hidden_a = params.actor_hidden_units
        output_dim = params.num_options


        #define shared state representation network
        # self.phi_fc1 = nn.Linear(self.state_dim, self.feature_dim)

        #define high actor network
        self.actor_fc1 = nn.Linear(input_dim, hidden_a[0])
        self.actor_fc2 = nn.Linear(hidden_a[0], hidden_a[1])
        self.actor_fc3 = nn.Linear(hidden_a[1], output_dim)

        #define option networks (low actor)
        self.options = nn.ModuleList([DAC_SingleOptionNet(params) for _ in range(output_dim)])

        #define option termination network
        self.beta_fc1 = nn.Linear(input_dim, hidden_a[0])
        self.beta_fc2 = nn.Linear(hidden_a[0], hidden_a[1])
        self.beta_fc3 = nn.Linear(hidden_a[1], output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """state: raw state as tensor.shape[batch_size, state_dim]"""
        x = state
        # x = self.phi_fc1(x) #get shared state representation

        #pass through each option network and stack outputs
        pi_w = []
        for option in self.options:
            pi_w_ = option(x)
            pi_w.append(pi_w_)
        pi_w = th.stack(pi_w, dim=1)

        #pass through master policy network
        x_pi = self.pi_gate(self.actor_fc1(x))
        x_pi = self.pi_gate(self.actor_fc2(x_pi))
        logits_pi = self.actor_fc3(x_pi)
        pi_W = F.softmax(logits_pi, dim=-1)

        #pass through beta net
        x_beta = self.beta_gate(self.beta_fc1(x))
        x_beta = self.beta_gate(self.beta_fc2(x_beta))
        logit_beta = self.beta_fc3(x_beta)
        beta_w = F.sigmoid(logit_beta)

        return {'pi_w': pi_w,           #option policies: shape[batch_size, num_options, action_dim]
                'betas': beta_w,        #option betas: shape[batch_size, num_options]
                'pi_W': pi_W,}           #master policy: shape[batch_size, num_options]
                # 'q_W': q }               #critic option value: shape[batch_size, num_options]

    def actor_loss(self, new_logp, old_logp, advantages):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(new_logp - old_logp)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        # Calculate the minimum surrogate loss for clipping
        loss = -th.min(surr1, surr2).mean()
        return loss


class DAC_Critic(nn.Module):
    def __init__(self, params):
        super(DAC_Critic, self).__init__()
        self.gate = params.critic_activation
        self.eps_clip = params.eps_clip
        input_dim = params.state_dim
        hidden = params.critic_hidden_units
        output_dim = params.num_options

        # define critic network
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

    def critic_loss(self, new_values, old_values, returns):
        #clip the difference between old and new values
        value_clip = old_values + th.clamp(new_values - old_values, -self.eps_clip, self.eps_clip)
        # value_clip = th.clamp(new_values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (new_values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss


#OLDDD
class DAC_Network(nn.Module):
    def __init__(self, params):
        super(DAC_Network, self).__init__()
        # store used parameters
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_options = params.num_options
        self.hidden_a = params.actor_hidden_units
        self.hidden_c = params.critic_hidden_units
        self.pi_gate = params.pi_h_activation
        self.critic_gate = params.critic_activation
        self.beta_gate = params.beta_activation
        self.feature_dim = params.feature_dim

        # define shared state representation network
        self.phi_fc1 = nn.Linear(self.state_dim, self.feature_dim)

        # define high actor network
        self.actor_fc1 = nn.Linear(self.feature_dim, self.hidden_a[0])
        self.actor_fc2 = nn.Linear(self.hidden_a[0], self.hidden_a[1])
        self.actor_fc3 = nn.Linear(self.hidden_a[1], self.num_options)

        # define option networks (low actor)
        self.options = nn.ModuleList([DAC_SingleOptionNet(params) for _ in range(self.num_options)])

        # define option termination network
        self.beta_fc1 = nn.Linear(self.feature_dim, self.hidden_a[0])
        self.beta_fc2 = nn.Linear(self.hidden_a[0], self.hidden_a[1])
        self.beta_fc3 = nn.Linear(self.hidden_a[1], self.num_options)

        # define critic network
        self.critic_fc1 = nn.Linear(self.feature_dim, self.hidden_c[0])
        self.critic_fc2 = nn.Linear(self.hidden_c[0], self.hidden_c[1])
        self.critic_fc3 = nn.Linear(self.hidden_c[1], self.num_options)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """state: raw state as tensor.shape[batch_size, state_dim]"""
        x = state
        x = self.phi_fc1(x)  # get shared state representation

        # pass through each option network and stack outputs
        pi_w = []
        for option in self.options:
            pi_w_ = option(x)
            pi_w.append(pi_w_)
        pi_w = th.stack(pi_w, dim=1)

        # pass through master policy network
        x_pi = self.pi_gate(self.actor_fc1(x))
        x_pi = self.pi_gate(self.actor_fc2(x_pi))
        logits_pi = self.actor_fc3(x_pi)
        pi_W = F.softmax(logits_pi, dim=-1)

        # pass through beta net
        x_beta = self.beta_gate(self.beta_fc1(x))
        x_beta = self.beta_gate(self.beta_fc2(x_beta))
        logit_beta = self.beta_fc3(x_beta)
        beta_w = F.sigmoid(logit_beta)

        # pass through critic network
        x_q = self.critic_gate(self.critic_fc1(x))
        x_q = self.critic_gate(self.critic_fc2(x_q))
        q = self.critic_fc3(x_q)

        return {'pi_w': pi_w,  # option policies: tensor.shape[batch_size, num_options, action_dim]
                'betas': beta_w,  # option betas: tensor.shape[batch_size, num_options]
                'pi_W': pi_W,  # master policy: tensor.shape[batch_size, num_options]
                'q_W': q}  # critic option value: tensor.shape[batch_size, num_options]

    @staticmethod
    def actor_loss(new_logp, old_logp, advantages, eps_clip):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(new_logp - old_logp)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        # Calculate the minimum surrogate loss for clipping
        loss = -th.min(surr1, surr2).mean()
        return loss

    @staticmethod
    def critic_loss(new_values, old_values, returns, eps_clip):
        # clip the difference between old and new values
        value_clip = old_values + th.clamp(new_values - old_values, -eps_clip, eps_clip)
        # value_clip = th.clamp(new_values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (new_values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss
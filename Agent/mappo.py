import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class MAPPO_Actor(nn.Module):
    def __init__(self, params):
        super(MAPPO_Actor, self).__init__()
        self.output_dim = params.action_dim
        self.partial = params.partial_observability
        self.eps_clip = params.eps_clip
        hidden = params.actor_hidden_dim

        #hidden layer determination
        if not self.partial:
            input_dim = params.state_dim + params.num_agents
            self.fc_in = nn.Linear(input_dim, hidden[0])  # input layer
            self.fc2 = nn.Linear(hidden[0], hidden[1])
            self.fc_out = nn.Linear(hidden[1], self.output_dim)
        else:
            # input_dim = params.obs_dim + params.action_dim + params.num_agents
            input_dim = params.obs_dim + params.num_agents
            self.fc_in = nn.Linear(input_dim, hidden[0])  # input layer
            self.gru = nn.GRU(hidden[0], hidden[1], batch_first=True)
            self.fc_out = nn.Linear(hidden[1], self.output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, *args):
        if not self.partial:
            x, = args
            x = th.tanh(self.fc_in(x))
            x = th.tanh(self.fc2(x))
            logits = self.fc_out(x)
            return logits
        else:
            x, h = args
            x = th.tanh(self.fc_in(x))
            x, h_new = self.gru(x, h)   #x: [batch, seq, hidden]
            logits = self.fc_out(x)     #h_new: [1, batch, hidden]
            return logits, h_new


    @staticmethod
    def select_action(logits):
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        return action, logp #, action_dist

    def actor_loss(self, new_logp, old_logp, advantages):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(new_logp - old_logp)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        # Calculate the minimum surrogate loss for clipping
        loss = -th.min(surr1, surr2).mean()
        return loss

class MAPPO_Critic(nn.Module):
    def __init__(self, params):
        super(MAPPO_Critic, self).__init__()
        self.partial = params.partial_observability
        self.eps_clip = params.eps_clip
        hidden = params.critic_hidden_dim

        if not self.partial:
            input_dim = params.state_dim
            self.fc_in = nn.Linear(input_dim, hidden[0])  # input layer
            self.fc2 = nn.Linear(hidden[0], hidden[1])
            self.fc_out = nn.Linear(hidden[1], 1)
        else:
            # input_dim = params.state_dim + params.action_dim * params.num_agents + params.num_agents
            input_dim = params.state_dim + params.num_agents
            self.fc_in = nn.Linear(input_dim, hidden[0])  # input layer
            self.gru = nn.GRU(hidden[0], hidden[1], batch_first=True)
            self.fc_out = nn.Linear(hidden[1], 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, *args):
        if not self.partial:
            x, = args
            x = th.tanh(self.fc_in(x))
            x = th.tanh(self.fc2(x))
            value = self.fc_out(x)
            return value
        else:
            x, h = args
            x = th.tanh(self.fc_in(x))
            x, h_new = self.gru(x, h)
            value = self.fc_out(x)
            return value, h_new


    def critic_loss(self, new_values, old_values, returns):
        value_clip = old_values + th.clamp(new_values - old_values, -self.eps_clip, self.eps_clip)
        # value_clip = th.clamp(new_values, old_values - eps_clip, old_values + eps_clip)

        loss_unclipped = (new_values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss
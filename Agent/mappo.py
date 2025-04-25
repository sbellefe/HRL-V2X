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
        hidden = params.actor_hidden_dim

        if not self.partial:
            input_dim = params.state_dim + params.num_agents
            self.gru = nn.GRU(hidden[0], hidden[1], batch_first=True)
        else:
            input_dim = params.obs_dim + params.action_dim + params.num_agents

        self.fc_in = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc_out = nn.Linear(hidden[1], self.output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, *args):
        """Fully observable: forward(state, agent_id)
          - state:  [batch, state_dim]
          - agent_id: [batch, num_agents]
          returns logits [batch, action_dim]

        Partially observable GRU: forward(x, hidden_state)
          - x:            [batch, obs_dim + action_dim + num_agents]
          - hidden_state: [1, batch, hidden[1]]
          returns (logits [batch, action_dim], new_hidden [1, batch, hidden[1]])
        """
        if self.partial:
            x, h = args
            x = th.tanh(self.fc_in(x))
            # GRU expects [B, T, features]; here T=1
            x, h_new = self.gru(x.unsqueeze(1), h)
            logits = self.fc_out(x.squeeze(1))
            return logits, h_new

        else:
            state, agent_id = args
            x = th.cat([state, agent_id], dim=-1)
            x = th.tanh(self.fc_in(x))
            x = th.tanh(self.fc2(x))
            logits = self.fc_out(x)
            return logits


    def forward_Z(self, state, agent_id):
        """Create state for agent_index and pass through
            network to get logits"""
        x = th.cat([state, agent_id], dim=-1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

    # def mapping_action2RRA(self, action):
    #
    #     if isinstance(action, th.Tensor):
    #         action = action.cpu().numpy()
    #
    #     if self.state_type == 'simplified_version':
    #
    #         if action < self.action_dim - 1:
    #             # convert action index to power allocation and SC allocation
    #             SC_index = self.ag_idx  # self.n_SC - self.ag_idx - 1
    #             Power_level_index = action % self.n_pw_levels
    #         else:
    #             SC_index = -1
    #             Power_level_index = -1
    #     else:
    #         if action < self.action_dim - 1:
    #             # convert action index to power allocation and SC allocation
    #             # SC_index = int(np.floor(action.cpu().numpy() / self.n_pw_levels))
    #             SC_index = int(np.floor(action / self.n_pw_levels))
    #             Power_level_index = action % self.n_pw_levels
    #         else:
    #             SC_index = -1
    #             Power_level_index = -1
    #
    #     return SC_index, Power_level_index

    @staticmethod
    def select_action(logits):
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        return action, logp #, action_dist

    def actor_loss(self, logp, old_logp, advantages, eps_clip):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(logp - old_logp)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        # Calculate the minimum surrogate loss for clipping
        loss = -th.min(surr1, surr2).mean()
        return loss

class MAPPO_Critic(nn.Module):
    def __init__(self, params):
        super(MAPPO_Critic, self).__init__()
        self.state_dim = params.state_dim + params.num_agents
        self.action_dim = params.action_dim
        self.hidden_dim = params.critic_hidden_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim[0])
        self.fc2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.fc3 = nn.Linear(self.hidden_dim[1],1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state, agent_id):
        """Create state for agent_index and pass through
            network to get logits"""
        x = th.cat([state, agent_id], dim=-1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

    def critic_loss(self, new_values, old_values, returns, eps_clip):
        # value_clip = old_values + th.clamp(values - old_values, -eps_clip, eps_clip)
        value_clip = th.clamp(new_values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (new_values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss
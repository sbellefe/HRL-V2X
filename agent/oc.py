import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical, Bernoulli
#TODO: rework OC implementation to match original paper

import sys
from math import exp
import numpy as np
from helpers.oc_helper import pre_process

class SingleOptionMLP(nn.Module):
    """Class for a single option policy"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleOptionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim)
        )

    def forward(self, x):
        return self.net(x)


class OptionCritic(nn.Module):
    def __init__(self, params):
        super(OptionCritic, self).__init__()
        self.device = params.device

        #define shared state representation network
        self.features = nn.Linear(params.state_dim, params.feature_dim)

        #define Q-network
        self.Q_net = nn.Sequential(
            nn.Linear(params.feature_dim, params.hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(params.hidden_dim[0], params.hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(params.hidden_dim[1], params.num_options)
        )

        #define beta-network
        self.beta_net = nn.Sequential(
            nn.Linear(params.feature_dim, params.hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(params.hidden_dim[0], params.hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(params.hidden_dim[1], params.num_options)
        )

        #define sub-policy networks
        self.options = nn.ModuleList([
            SingleOptionMLP(params.feature_dim, params.hidden_dim, params.action_dim)
            for _ in range(params.num_options)
        ])

        self.to(self.device)
        self.train(mode=True)

    def get_state(self, obs):
        """pass observation through shared phi network to get state"""
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        """pass state through Q-network"""
        Q = self.Q_net(state)
        return Q

    def get_betas(self, state):
        """pass state through beta network to get all betas.
         used for initial state where no current option"""
        betas = self.beta_net(state).sigmoid()
        return betas

    def get_beta(self, state, current_option):
        """pass state through beta network and slice to get current beta,
            select epsilon-greedy next-option. Used for non-initial states"""

        #get termination probability for current option
        beta_prob = self.beta_net(state)[:,current_option].sigmoid()

        #sample beta from termination probability and convert to bool
        beta = bool(Bernoulli(beta_prob).sample().item())

        #compute greedy next option
        Q = self.get_Q(state)
        greedy_next_option = Q.argmax(dim=1).item()

        return beta, greedy_next_option

    def get_action(self, state, option, temperature):
        """"gets action info for given option from parameterized sub policy
                    Inputs:
                        Current State, Option
                    Returns:
                        Action: sampled from parameterized sub-policy network output distribution
                        logp: log probability density/mass function evaluated at sampled action
                        entropy: computed entropy of the output distribution
                """
        #pass thru current option network
        logits = self.options[option](state)

        # compute probability over actions using temp for exploration
        action_dist = (logits / temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)  # create discrete distribution

        action = action_dist.sample()  # sample action from distribution stochastically
        logp = action_dist.log_prob(action)  # compute log prob of selected action
        entropy = action_dist.entropy()  # compute entropy of distribution

        return action.item(), logp, entropy

    def greedy_option(self, state):
        """chose greedy option for current state"""
        Q = self.get_Q(state)
        greedy_option = Q.argmax(dim=-1).item()
        return greedy_option

    def critic_loss(self, agent_prime, batch, params):
        """Computes squared TD error between Q and the update target for a
            minibatch of samples,
                Inputs:
                    agent: nn.Module type OptionCritic to be trained
                    agent_prime: target model
                    batch: tuple(obs, option, reward, next_obs, done)
                Returns:
                    critic_loss: squared TD error
               """
        #unpack and format batch, create terminal mask
        obs, options, rewards, next_obs, dones = batch
        batch_idx = th.arange(len(options)).long()  # Tensor of batch indices [0, 1, ..., batch_size-1]
        options = th.LongTensor(options).to(params.device)  # Convert to a tensor of integers
        rewards = th.FloatTensor(rewards).to(params.device)  # Convert to a floating-point tensor
        masks = 1 - th.FloatTensor(dones).to(params.device)  # Create mask for terminal states (0=Done)

        # compute current Q-values, ie Q(s,w1), Q(s,w2)...
        states = self.get_state(pre_process(obs)).squeeze(0)
        Q = self.get_Q(states)

        # compute next state Q-values with target network, ie Q'(s',w1), Q'(s',w2)...
        next_states_prime = agent_prime.get_state(pre_process(next_obs)).squeeze(0)  # compute current state (from obs)
        next_Q_prime = agent_prime.get_Q(next_states_prime)  # detach?

        # Compute probability each option terminates in s', ie beta_w1(s'), beta_w2(s')...
        next_states = self.get_state(pre_process(next_obs)).squeeze(0)
        next_states_betas = self.get_betas(next_states).detach()
        next_states_beta = next_states_betas[batch_idx, options]  # index for specific options executed in batch

        # Now we can calculate the update target gt
        gt = rewards + masks * params.gamma * \
             ((1 - next_states_beta) * next_Q_prime[batch_idx, options] + next_states_beta *
              next_Q_prime.max(dim=-1)[0])

        # compute loss as TD error
        td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
        return td_err

    def actor_loss(self, agent_prime, obs, option, logp, entropy, reward, done, next_obs, params):
        """ compute actor loss used to train option-policy and beta networks every time-step
            """
        #pass obs and next_obs through networks to get states
        state = self.get_state(pre_process(obs))
        next_state = self.get_state(pre_process(next_obs))  # get next state from next obs
        next_state_prime = agent_prime.get_state(pre_process(next_obs))  # get target next state

        # get termination prob for current_state + current_option
        beta = self.get_betas(state)[:, option]

        # get termination prob for next_state + current_option
        next_beta = self.get_betas(next_state)[:, option].detach()

        # compute Q(s,o) and Q'(s',o)
        Q = self.get_Q(state).detach().squeeze()
        next_Q_prime = agent_prime.get_Q(next_state_prime).detach().squeeze()

        # "One-step off-policy update target"
        gt = reward + (1 - done) * params.gamma * \
             ((1 - next_beta) * next_Q_prime[option] + next_beta * next_Q_prime.max(dim=-1)[0])

        # The beta loss
        beta_loss = beta * (Q[option].detach() - Q.max(dim=-1)[0].detach() + params.beta_reg) * (1 - done)

        # actor-critic policy/beta gradient loss with entropy regularization
        policy_loss = -logp * (gt.detach() - Q[option]) - params.entropy_coef * entropy
        actor_loss = beta_loss + policy_loss
        return actor_loss



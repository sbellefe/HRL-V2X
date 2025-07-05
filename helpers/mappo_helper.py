import torch as th
import numpy as np
import matplotlib.pyplot as plt
import sys


class RolloutBuffer_FO:
    def __init__(self, params):
        self.device = params.device
        self.batch_size = params.buffer_episodes  # episodes per batch
        self.t_max = params.t_max  # timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim

        #preallocate tensors, flat transitions for the entire batch (length = batch_size * t_max)
        BT, N = self.batch_size * self.t_max, self.N
        self.global_states = th.zeros((BT, self.state_dim), device=self.device)
        self.joint_actions = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.global_rewards = th.zeros((BT,), device=self.device)
        self.values = th.zeros((BT,), device=self.device)
        self.logps = th.zeros((BT, N), device=self.device)
        self.advantages = None  # will be [BT,]
        self.returns = None     # will be [BT,]

        # internal write indices
        self.t_steps = 0  # 0..BT-1 within a batch
        self.n_eps = 0  # 0..B-1
        self.ep_tot, self.n_roll = 0, 0 # multiple rollout counters

        # reuse dataset / dataloader
        self.dataset = None
        self.dataloader = None

    def push(self, *args):
        s, a, r, v, lp, = args
        idx = self.t_steps
        self.global_states[idx] = s
        self.joint_actions[idx] = a
        self.global_rewards[idx] = r
        self.values[idx] = v
        self.logps[idx] = lp

        #increment counters
        self.t_steps += 1
        if self.t_steps % self.t_max == 0:
            self.n_eps += 1
            self.ep_tot += 1

    def process_batch(self):
        self.compute_GAE()

        s = self.global_states
        a = self.joint_actions
        lp = self.logps
        v = self.values
        rtrn = self.returns
        adv = self.advantages

        # prepare dataset and dataloader once
        if self.dataset is None:
            self.dataset = th.utils.data.TensorDataset(s, a, lp, v, rtrn, adv)
            mb_size = len(self.dataset) // self.num_mb
            self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=mb_size, shuffle=True)
        else:
            # overwrite tensors in-place
            self.dataset.tensors = (s, a, lp, v, rtrn, adv)

        # reset for next batch
        self.t_steps = 0
        self.n_eps = 0
        self.n_roll += 1

        return self.dataloader

    def compute_GAE(self):
        B, T, N = self.batch_size, self.t_max, self.N

        #reshape variables
        r = self.global_rewards.view(B, T)
        v = self.values.view(B, T)

        # pad values with zeros for last state [T]→[T+1]
        fv = th.zeros((B, 1), device=self.device)
        v_pad = th.cat([v, fv], dim=1)  # [B, T+1]

        # Compute deltas
        delta = r + self.gamma * v_pad[:, 1:] - v

        # Allocate advantage & return arrays in episode form
        adv = th.zeros_like(delta)    # [B, T]
        ret = th.zeros_like(r)          # [B, T]

        #Backward pass: for each t = T-1 … 0
        for t in reversed(range(T)):
            if t == T - 1:
                # terminal step: no future advantage or return
                adv[:, t] = delta[:, t]
                ret[:, t] = r[:, t]
            else:
                # GAE recurrence and discounted return
                adv[:, t] = (delta[:, t] + self.gamma * self.gae_lambda * adv[:, t+1])
                ret[:, t] = r[:, t] + self.gamma * ret[:, t+1]

        #Flatten back to [B*T, …] for DataLoader
        self.advantages = adv.view(B*T)     # [B*T]
        self.returns = ret.view(B*T)        # [B*T]

        # normalize
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)

class RolloutBuffer_PO:
    def __init__(self, params):
        self.device = params.device
        self.batch_size = params.buffer_episodes    #episodes per batch
        self.t_max = params.t_max                   #timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim

        #preallocate tensors, flat transitions for the entire batch (length = batch_size * t_max)
        BT, N = self.batch_size * self.t_max, self.N
        self.observations = th.zeros((BT, N, self.obs_dim), device=self.device)
        self.fp_global_states = th.zeros((BT, N, self.state_dim), device=self.device)
        self.joint_actions = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.global_rewards = th.zeros((BT,), device=self.device)
        self.values = th.zeros((BT, N), device=self.device)
        self.logps = th.zeros((BT, N), device=self.device)
        self.advantages = None  # will be [BT, N]
        self.returns = None     # will be [BT,]

        # internal write indices
        self.t_steps = 0  # 0..BT-1 within a batch
        self.n_eps = 0  # 0..B-1
        self.ep_steps = 0
        self.ep_tot, self.n_roll = 0, 0 # multiple rollout counters

        # reuse dataset / dataloader
        self.dataset = None
        self.dataloader = None

    def push(self, *args):
        obs, fpgs, a, r, v, lp = args
        idx = self.t_steps
        self.observations[idx] = obs
        self.fp_global_states[idx] = fpgs
        self.joint_actions[idx] = a
        self.global_rewards[idx] = r
        self.values[idx] = v
        self.logps[idx] = lp

        # increment counters
        self.t_steps += 1
        self.ep_steps += 1
        if self.t_steps % self.t_max == 0:
            self.n_eps += 1
            self.ep_tot += 1
            self.ep_steps = 0

    def process_batch(self):
        self.compute_GAE()

        B, T, N = self.batch_size, self.t_max, self.N

        obs = self.observations.view(B, T, N, self.obs_dim)
        fpgs = self.fp_global_states.view(B, T, N, self.state_dim)
        a = self.joint_actions.view(B, T, N)
        v = self.values.view(B, T, N)
        lp = self.logps.view(B, T, N)
        rtrn = self.returns.view(B, T)
        adv = self.advantages.view(B, T, N)

        if self.dataset is None:
            self.dataset = th.utils.data.TensorDataset(obs, fpgs, a, v, lp, rtrn, adv)
            mb_size = len(self.dataset) // self.num_mb
            self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=mb_size, shuffle=True)
        else:
            # overwrite tensors in-place
            self.dataset.tensors = (obs, fpgs, a, v, lp, rtrn, adv)

        # reset for next batch
        self.t_steps = 0
        self.n_eps = 0
        self.n_roll += 1

        # self.plot()

        return self.dataloader

    def compute_GAE(self):
        B, T, N = self.batch_size, self.t_max, self.N

        # reshape variables
        r = self.global_rewards.view(B, T)
        v = self.values.view(B, T, N)

        # pad values with zeros for last state [T]→[T+1]
        fv = th.zeros((B, 1, N), device=self.device)
        v_pad = th.cat([v, fv], dim=1)  # [B, T+1, N]

        # Compute deltas
        delta = r.unsqueeze(-1) + self.gamma * v_pad[:, 1:, :] - v

        adv = th.zeros_like(delta)  # [B, T, N]
        ret = th.zeros_like(r)       # [B, T]

        #Backward pass: for each t = T-1 … 0
        for t in reversed(range(T)):
            if t == T - 1:
                # terminal step: no future advantage or return
                adv[:, t] = delta[:, t]
                ret[:, t] = r[:, t]
            else:
                # GAE recurrence and discounted return
                adv[:, t] = delta[:, t] + self.gamma * self.gae_lambda * adv[:, t+1]
                ret[:, t] = r[:, t] + self.gamma * ret[:, t+1]

        adv = adv.flatten()   # [B*T*N]
        ret = ret.flatten()        # [B*T]
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.returns = (ret - ret.mean()) / (ret.std() + 1e-8)



    def plot(self):
        """
        Plot the batch returns, values, and advantages.
        Creates a 3x1 subplot figure.
        """
        # Ensure numpy arrays for compatibility with matplotlib
        returns = self.returns.cpu().numpy()  # shape [BT,]
        values = self.values.cpu().numpy()  # shape [BT, N]
        advantages = self.advantages.cpu().numpy()  # shape [BT, N]

        fig, axes = plt.subplots(3, 1, figsize=(20, 12))

        # 1) Returns over the batch
        axes[0].plot(returns)
        axes[0].set_title("Batch Returns")
        # axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Return")

        # 2) Value estimates for each agent
        for agent_idx in range(self.N):
            axes[1].plot(values[:, agent_idx], label=f"Agent {agent_idx}")
        axes[1].set_title("Value Estimates per Agent")
        # axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Value")
        # axes[1].legend()

        # 3) Advantages for each agent
        for agent_idx in range(self.N):
            axes[2].plot(advantages[:, agent_idx], label=f"Agent {agent_idx}")
        axes[2].set_title("Advantages per Agent")
        # axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Advantage")
        axes[2].legend()

        plt.tight_layout()
        plt.show()


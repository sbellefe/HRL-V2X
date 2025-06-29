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

ag, ep_idx = 0,0
class NEWRolloutBuffer_PO:
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

        # pre-allocate tensors on device
        B, T, N = self.batch_size, self.t_max, self.N
        self.observations = th.zeros(B, T, N, self.obs_dim, device=self.device)
        self.fp_global_states = th.zeros(B, T, N, self.state_dim, device=self.device)
        self.global_rewards = th.zeros(B, T, device=self.device)
        self.joint_actions = th.zeros(B, T, N, device=self.device, dtype=th.long)
        self.logps = th.zeros(B, T, N, device=self.device)
        self.values = th.zeros(B, T+1, N, device=self.device)
        self.prev_actions = th.zeros(B, T, N, self.action_dim, device=self.device)

        self.t = 0
        self.ep = 0

    def push(self, *args):
        obs, fpgs, a, r, lp, v, prev_a = args
        assert 0 <= self.ep < self.batch_size, f"ep out of range: {self.ep}"
        assert 0 <= self.t  < self.t_max, f"t out of range: {self.t}"

        idx = (self.ep, self.t)
        self.observations[idx] = obs
        self.fp_global_states[idx] = fpgs
        self.joint_actions[idx] = a
        self.global_rewards[idx] = r
        self.logps[idx] = lp
        self.values[idx] = v
        self.prev_actions[idx] = prev_a
        # print(f"NEW: t= {self.t} prev_action: {prev_a}")


        #increment TODO add from process episode
        self.t += 1

    def process_episode(self):
        assert self.t == self.t_max, f"expected t={self.t_max}, got {self.t}"
        self.values[self.ep, self.t_max] = 0.0
        self.ep += 1
        self.t = 0


    def process_batch(self):
        B, T, N = self.batch_size, self.t_max, self.N
        assert self.ep == B, f"expected {B} eps, got {self.ep}"

        # 1) compute discounted returns: shape [B, T]
        returns = th.zeros_like(self.global_rewards)
        # start from last timestep
        returns[:, -1] = self.global_rewards[:, -1]
        for t in range(T-2, -1, -1):
            returns[:, t] = self.global_rewards[:, t] + self.gamma * returns[:, t+1]

        # 2) compute GAE advantages: shape [B, T, N]
        # deltas = r_t + gamma * V_{t+1} - V_t
        deltas = self.global_rewards.unsqueeze(-1) + \
                 self.gamma * self.values[:, 1:, :] - \
                 self.values[:, :-1, :]
        advantages = th.zeros_like(deltas)
        last_gae = th.zeros(B, N, device=self.device)
        for t in range(T-1, -1, -1):
            last_gae = deltas[:, t, :] + self.gamma * self.gae_lambda * last_gae
            advantages[:, t, :] = last_gae

        # normalize advantages and returns across all entries
        adv_flat = advantages.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.view(B, T, N)

        ret_flat = returns.view(-1)
        ret_flat = (ret_flat - ret_flat.mean()) / (ret_flat.std() + 1e-8)
        returns = ret_flat.view(B, T)

        # # right before `# 3) build dataset of full episodes`
        # ep0_r = self.global_rewards[ep_idx].cpu().numpy()  # [T]
        # ep0_vals = self.values[ep_idx, :-1, ag].cpu().numpy()  # pick agent 0: [T]
        # ep0_ret = returns[ep_idx].detach().cpu().numpy()  # [T]
        # ep0_adv = advantages[ep_idx].detach().cpu().numpy()[:, ag]  # [T]
        # ep0_lp = self.logps[ep_idx].detach().cpu().numpy()[:, ag]  # [T]
        #
        # print(f"NEW ep{ep_idx} rewards:   ", ep0_r)
        # print(f"NEW ep{ep_idx} agent {ag} values:    ", ep0_vals)
        # print(f"NEW ep{ep_idx} returns:   ", ep0_ret)
        # print(f"NEW ep{ep_idx} agent {ag} advantages:", ep0_adv)
        # print(f"NEW ep{ep_idx} agent {ag} logps:", ep0_lp)

        # 3) build dataset of full episodes
        dataset = th.utils.data.TensorDataset(
            self.observations,
            self.fp_global_states,
            self.joint_actions,
            self.logps,
            # use values up to T (exclude extra final slot)
            self.values[:, :-1, :],
            self.prev_actions,
            returns,
            advantages
        )
        mb_size = len(dataset) // self.num_mb
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)
        # dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)


        # reset for next rollout
        self.ep = 0
        self.t = 0

        return dataloader


class OLDRolloutBuffer_PO:
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

        self.observations = []
        self.fp_global_states = []
        self.joint_action = []
        self.global_reward = []
        self.logps = []
        self.values = []
        self.prev_actions = []

        #per episode returns and advantages
        self.returns = []
        self.advantages = []

        self.n_eps = 0
        self.t_steps = 0
        self.t_tot, self.ep_tot = 0, 0
        self.n_roll = 0

    def push(self, *args):
        """Add one timestep transition to the buffer. Transition variable shapes: (N = num_agents)"""
        obs, fpgs, actions, reward, logps, values, prev_action = args
        self.observations.append(obs)                   #tensor [N, obs_dim]
        self.fp_global_states.append(fpgs)              #tensor [N, state_dim]
        self.joint_action.append(actions)               #tensor [N]
        self.global_reward.append(reward)               #float
        self.logps.append(logps)                        #tensor [N]
        self.values.append(values)                      #tensor [N]
        self.prev_actions.append(prev_action.clone())          #tensor [N * action_dim]
        # print(f"OLD: t= {self.t_steps} prev_action: {prev_action}")

        self.t_steps += 1
        self.t_tot += 1

    def process_episode(self, final_values=None):
        """Call at end of each episode. Compute GAE and discounted returns
            for the last t_steps transitions."""

        # final_values = None
        T, N = self.t_steps, self.N

        # get the last T rewards as tensor
        rewards = th.tensor(self.global_reward[-T:], dtype=th.float32, device=self.device)  # [T]


        values = th.stack(self.values[-T:], dim=0) # [T, N]
        if final_values is None:
            final_values = th.zeros_like(values[-1:])  # [1, N]
        else:
            final_values = final_values.unsqueeze(0) # [1, N]

        adv = th.zeros_like(values)             # [T] or [T, N]
        ret = th.zeros_like(rewards)            # [T]
        gae = th.zeros_like(final_values)       # [1] or [N]

        values = th.cat([values, final_values], dim=0)  # [T+1] or [T+1, N]

        R = 0.0
        for i in reversed(range(T)):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[i] = gae
            R = rewards[i] + self.gamma * R
            ret[i] = R

        self.advantages.append(adv)  #[T] or [T, N]
        self.returns.append(ret)     #[T]
        self.n_eps += 1
        self.ep_tot += 1
        self.t_steps = 0

    def process_batch(self):
        """After batch_size episodes, reshape step‐level lists into
            [batch_size, t_max, ...] tensors, build a DataLoader, then clear buffer"""

        B, T, N = self.batch_size, self.t_max, self.N
        # stack and reshape
        obs = th.stack(self.observations, dim=0).view(B, T, N, self.obs_dim)
        fpgs = th.stack(self.fp_global_states, dim=0).view(B, T, N, self.state_dim)
        actions = th.stack(self.joint_action, dim=0).view(B, T, N)
        logps = th.stack(self.logps, dim=0).view(B, T, N)
        values = th.stack(self.values, dim=0).view(B, T, N)
        prev_action = th.stack(self.prev_actions, dim=0).view(B, T, N, self.action_dim)
        returns = th.stack(self.returns, dim=0).view(B, T)
        advantages = th.stack(self.advantages, dim=0).view(B, T, N)

        # print(f"\n\n\n\n\n prev_action[0,:,0,:]: {prev_action[0,:,0,:]}")
        # print(f"th.stack(self.prev_action, dim=0).shape: {th.stack(self.prev_actions, dim=0).shape}")

        # normalize across all entries
        adv = advantages.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        advantages = adv.view(B, T, N)

        ret = returns.flatten()
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        returns = ret.view(B, T)

        # # right before `# 3) build dataset of full episodes`
        # ep0_r = th.tensor(self.global_reward[:T]).cpu().numpy()  # [T]
        # ep0_vals = values[0, :, 0].cpu().numpy()  # pick agent 0: [T]
        # ep0_ret = returns[0].detach().cpu().numpy()  # [T]
        # ep0_adv = advantages[0].detach().cpu().numpy()[:, 0]  # [T]
        # ep0_lp = logps[0].detach().cpu().numpy()[:, 0]  # [T]
        #
        # print("OLD ep0 rewards:   ", ep0_r)
        # print("OLD ep0 values:    ", ep0_vals)
        # print("OLD ep0 returns:   ", ep0_ret)
        # print("OLD ep0 advantages:", ep0_adv)
        # print("OLD ep0 logps:", ep0_lp)

        # dataset of full episodes
        dataset = th.utils.data.TensorDataset(obs, fpgs, actions, logps, values, prev_action, returns, advantages)
        mb_size = len(dataset) // self.num_mb
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)
        # dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)


        self.reset_buffer()

        return dataloader


    def reset_buffer(self):
        self.joint_action.clear()
        self.global_reward.clear()
        self.logps.clear()
        self.values.clear()

        self.observations.clear()
        self.fp_global_states.clear()
        self.prev_actions.clear()

        self.returns.clear()
        self.advantages.clear()
        self.n_eps = 0
        self.n_roll += 1


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

class ZZZZRolloutBuffer:
    def __init__(self, params):
        self.device = params.device
        self.PO = params.partial_observability
        self.batch_size = params.buffer_episodes    #episodes per batch
        self.t_max = params.t_max                   #timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches   #for RNN
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.plot = False

        # Flat lists of transitions for the entire batch (length = batch_size * t_max)
        if not self.PO:
            self.global_state = []
            self.joint_action = []
            self.global_reward = []
            self.logps = []
            self.values = []
        else:
            self.observations = []
            self.fp_global_states = []
            self.joint_action = []
            self.global_reward = []
            self.logps = []
            self.values = []
            self.prev_actions = []

        #per episode returns and advantages
        self.returns = []
        self.advantages = []

        self.n_eps = 0
        self.t_steps = 0
        self.t_tot, self.ep_tot = 0, 0
        self.n_roll = 0

        #Env params stored here (PLOTTING ONLY)
        self.V2V_power_dB_list = [23, 15, 5]
        self.num_power_levels = len(self.V2V_power_dB_list)
        self.num_SC = 4


    def push(self, *args):
        """Add one timestep transition to the buffer. Transition variable shapes: (N = num_agents)"""

        if not self.PO:
            global_state, actions, reward, logps, value = args
            self.global_state.append(global_state)  #tensor shape [1, state_dim]
            self.joint_action.append(actions)       #tensor shape [N]
            self.global_reward.append(reward)       #float
            self.logps.append(logps)                #tensor shape [N]
            self.values.append(value)               #tensor shape [1]
        else:
            obs, fpgs, actions, reward, logps, values, prev_action = args
            self.observations.append(obs)                   #tensor [N, obs_dim]
            self.fp_global_states.append(fpgs)              #tensor [N, state_dim]
            self.joint_action.append(actions)               #tensor [N]
            self.global_reward.append(reward)               #float
            self.logps.append(logps)                        #tensor [N]
            self.values.append(values)                      #tensor [N]
            self.prev_actions.append(prev_action)          #tensor [N * action_dim]

        self.t_steps += 1
        self.t_tot += 1

    def process_episode(self, final_values=None):
        """Call at end of each episode. Compute GAE and discounted returns
            for the last t_steps transitions."""

        # final_values = None
        T, N = self.t_steps, self.N

        # get the last T rewards as tensor
        rewards = th.tensor(self.global_reward[-T:], dtype=th.float32, device=self.device)  # [T]

        # final_values = None
        if not self.PO:
            values = th.cat(self.values[-T:], dim=0)    # [T]
            if final_values is None:
                final_values = th.zeros_like(values[-1:])  # [1]
        else:
            values = th.stack(self.values[-T:], dim=0) # [T, N]
            if final_values is None:
                final_values = th.zeros_like(values[-1:])  # [1, N]
            else:
                final_values = final_values.unsqueeze(0) # [1, N]

        adv = th.zeros_like(values)             # [T] or [T, N]
        ret = th.zeros_like(rewards)            # [T]
        gae = th.zeros_like(final_values)       # [1] or [N]

        values = th.cat([values, final_values], dim=0)  # [T+1] or [T+1, N]

        R = 0.0
        for i in reversed(range(T)):
            delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[i] = gae
            R = rewards[i] + self.gamma * R
            ret[i] = R

        self.advantages.append(adv)  #[T] or [T, N]
        self.returns.append(ret)     #[T]
        self.n_eps += 1
        self.ep_tot += 1
        self.t_steps = 0


    def process_batch(self):
        """After batch_size episodes, reshape step‐level lists into
            [batch_size, t_max, ...] tensors, build a DataLoader, then clear buffer"""

        B, T, N = self.batch_size, self.t_max, self.N


        if not self.PO:
            # SIG: flatten across all time-steps [B*T, ...]
            states = th.stack(self.global_state, dim=0).squeeze(1)
            actions = th.stack(self.joint_action, dim=0)  # [B*T, N]
            logps = th.stack(self.logps, dim=0)  # [B*T, N]
            values = th.stack(self.values, dim=0).squeeze(-1)  # [B*T]
            returns = th.stack(self.returns, dim=0).view(-1)  # [B*T]
            advantages = th.stack(self.advantages, dim=0).view(-1)  # [B*T]

            # normalize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            dataset = th.utils.data.TensorDataset(states, actions, logps, values, returns, advantages)
            mb_size = len(dataset) // self.num_mb
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)

        else:
            # stack and reshape
            obs = th.stack(self.observations, dim=0).view(B, T, N, self.obs_dim)
            fpgs = th.stack(self.fp_global_states, dim=0).view(B, T, N, self.state_dim)
            actions = th.stack(self.joint_action, dim=0).view(B, T, N)
            logps = th.stack(self.logps, dim=0).view(B, T, N)
            values = th.stack(self.values, dim=0).view(B, T, N)
            prev_action = th.stack(self.prev_actions, dim=0).view(B, T, N, self.action_dim)
            returns = th.stack(self.returns, dim=0).view(B, T)
            advantages = th.stack(self.advantages, dim=0).view(B, T, N)

            # normalize across all entries
            adv = advantages.flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            advantages = adv.view(B, T, N)

            ret = returns.flatten()
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
            returns = ret.view(B, T)

            # dataset of full episodes
            dataset = th.utils.data.TensorDataset(obs, fpgs, actions, logps, values, prev_action, returns, advantages)
            mb_size = len(dataset) // self.num_mb
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)

        if self.plot and self.n_roll % 6 == 0:
            self.plot_rollout(dataset)

        self.reset_buffer()

        return dataloader

    def reset_buffer(self):
        self.joint_action.clear()
        self.global_reward.clear()
        self.logps.clear()
        self.values.clear()
        if self.PO:
            self.observations.clear()
            self.fp_global_states.clear()
            self.prev_actions.clear()
        else:
            self.global_state.clear()
        self.returns.clear()
        self.advantages.clear()
        self.n_eps = 0
        self.n_roll += 1

    def plot_rollout(self, dataset):
        """MAY NOT WORK"""
        P = self.num_power_levels
        S = self.num_SC
        null_action = self.action_dim - 1

        if self.PO:
            actions, obs, fpgs, prev, logps, values, returns, advantages = dataset.tensors
            B, T, N = values.shape

            # flatten
            time = np.arange(B * T)
            returns_series = returns.view(-1).detach().cpu().numpy()
            values_series = values.view(B * T, N).detach().cpu().numpy()
            adv_series = advantages.view(B * T, N).detach().cpu().numpy()
            actions_series = actions.view(B * T, N).detach().cpu().numpy()
            logps_series = logps.view(B * T, N).detach().cpu().numpy()


            # figure + axes
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            fig.suptitle(f"Rollout {self.n_roll+1} from episode {self.ep_tot-self.batch_size+1} to {self.ep_tot}. {self.t_tot} total completed steps")


            # 1) Returns
            axes[0].plot(time, returns_series)
            axes[0].set_ylabel("Returns")

            # 2) Values
            for i in range(N):
                axes[1].plot(time, values_series[:, i], label=f"Agent {i+1}")
            axes[1].set_ylabel("Values")
            axes[1].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 3) Advantages
            for i in range(N):
                axes[2].plot(time, adv_series[:, i], label=f"Agent {i+1}")
            axes[2].set_ylabel("Advantages")
            axes[2].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 4) Actions: y-axis = subchannels (0..P-1) + no-transmit (index P)
            ax = axes[3]

            # Prepare scatter data
            times = np.repeat(time, N)
            agents = np.tile(np.arange(N), B*T)
            acts = actions_series.flatten()

            # Map to y-values: subchannel = act//P; null -> P
            y_vals = np.where(acts == null_action, S, acts // P)

            # Power index: 0..P-1  →  use for alpha & size
            pw_idx = np.where(acts == null_action,
                              0,  # give null-action lowest alpha/size
                              acts % P)

            alpha_vals = (pw_idx + 1) / P  # low→transparent, high→opaque
            size_vals = 20 + pw_idx * 10  # bigger for higher power

            # Scatter: one big batch—color encodes agent
            for agent in range(N):
                mask = agents == agent
                ax.scatter(
                    times[mask],
                    y_vals[mask],
                    c=f"C{agent}",
                    alpha=alpha_vals[mask],
                    s=size_vals[mask],
                    label=f"Agent {agent + 1}",
                    edgecolors='none'  # a bit cleaner
                )

            # Y‐axis ticks & labels
            ax.set_yticks(np.arange(S + 1))
            ax.set_yticklabels([f"ch {i+1}" for i in range(S)] + ["N/A"])
            ax.set_ylabel("Sub-channel")

            # Single legend for agents
            ax.legend(fontsize='x-small', ncol=N, loc='upper right', bbox_to_anchor=(1.0, 0.95))

            # 5) Log‐probs
            for i in range(N):
                axes[4].plot(time, logps_series[:, i], label=f"Agent {i}")
            axes[4].set_ylabel("Log-probs")
            axes[4].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 6) Blank placeholder
            # axes[5].axis("off")

            # X-ticks on the bottom axis, every T steps
            # X‐axis ticks & limits
            axes[-1].set_xlim(-10, B*T + 10)
            # major every 10*T, minor every T
            axes[-1].set_xticks(np.arange(0, B*T + 1, 10 * T))
            axes[-1].set_xticks(np.arange(0, B*T + 1, T), minor=True)
            axes[-1].set_xlabel("Time steps (minor=T, major=10×T)")

            plt.tight_layout(rect=(0, 0, 1, 0.96))  # leave space for suptitle
            fig.subplots_adjust(bottom=0.08)
            plt.show()
        else:
            states, actions, logps, values, returns, advantages = dataset.tensors
            B, N = actions.shape

            # flatten
            time = np.arange(B)
            returns_series = returns.detach().cpu().numpy()
            values_series = values.detach().cpu().numpy()
            adv_series = advantages.detach().cpu().numpy()
            actions_series = actions.detach().cpu().numpy()
            logps_series = logps.detach().cpu().numpy()

            # figure + axes
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            fig.suptitle(
                f"Rollout {self.n_roll + 1} from episode {self.ep_tot - self.batch_size + 1} to {self.ep_tot}. {self.t_tot} total completed steps")

            # 1) Returns
            axes[0].plot(time, returns_series)
            axes[0].set_ylabel("Returns")

            # # 2) Values
            axes[1].plot(time, values_series, color='g')
            axes[1].set_ylabel("Values")
            # for i in range(N):
            #     axes[1].plot(time, values_series[:, i], label=f"Agent {i + 1}")
            # axes[1].set_ylabel("Values")
            # axes[1].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 3) Advantages
            axes[2].plot(time, adv_series, color='r')
            axes[2].set_ylabel("Advantages")
            # for i in range(N):
            #     axes[2].plot(time, adv_series[:, i], label=f"Agent {i+1}")
            # axes[2].set_ylabel("Advantages")
            # axes[2].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 4) Actions: y-axis = subchannels (0..P-1) + no-transmit (index P)
            ax = axes[3]
            P = self.num_power_levels
            S = self.num_SC
            null_action = self.action_dim - 1

            # Prepare scatter data
            times = np.repeat(time, N)
            agents = np.tile(np.arange(N), B)
            acts = actions_series.flatten()

            # Map to y-values: subchannel = act//P; null -> P
            y_vals = np.where(acts == null_action, S, acts // P)

            # Power index: 0..P-1  →  use for alpha & size
            pw_idx = np.where(acts == null_action,
                              0,  # give null-action lowest alpha/size
                              acts % P)

            alpha_vals = (pw_idx + 1) / P  # low→transparent, high→opaque
            size_vals = 20 + pw_idx * 10  # bigger for higher power

            # Scatter: one big batch—color encodes agent
            for agent in range(N):
                mask = agents == agent
                ax.scatter(
                    times[mask],
                    y_vals[mask],
                    c=f"C{agent}",
                    alpha=alpha_vals[mask],
                    s=size_vals[mask],
                    label=f"Agent {agent + 1}",
                    edgecolors='none'  # a bit cleaner
                )

            # Y‐axis ticks & labels
            ax.set_yticks(np.arange(S + 1))
            ax.set_yticklabels([f"ch {i + 1}" for i in range(S)] + ["N/A"])
            ax.set_ylabel("Sub-channel")

            # Single legend for agents
            ax.legend(fontsize='x-small', ncol=N, loc='upper right', bbox_to_anchor=(1.0, 0.95))

            # 5) Log‐probs
            for i in range(N):
                axes[4].plot(time, logps_series[:, i], label=f"Agent {i}")
            axes[4].set_ylabel("Log-probs")
            axes[4].legend(fontsize='x-small', ncol=N, loc='upper right')

            # 6) Blank placeholder
            # axes[5].axis("off")

            # X-ticks on the bottom axis, every T steps
            # X‐axis ticks & limits
            axes[-1].set_xlim(-10, B + 10)
            # major every 10*T, minor every T
            axes[-1].set_xticks(np.arange(0, B + 1, 100))
            axes[-1].set_xticks(np.arange(0, B + 1, 10), minor=True)
            axes[-1].set_xlabel("Time steps (minor=T, major=10×T)")

            plt.tight_layout(rect=(0, 0, 1, 0.96))  # leave space for suptitle
            fig.subplots_adjust(bottom=0.08)
            plt.show()

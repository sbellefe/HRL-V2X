import torch as th
import numpy as np
import sys

def pre_process(obs):
    state = th.FloatTensor(obs).unsqueeze(0)
    return state

class RolloutBuffer:
    def __init__(self, params):
        self.device = params.device
        self.PO = params.partial_observability
        self.batch_size = params.buffer_episodes    #episodes per batch
        self.t_max = params.t_max                   #timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim

        # Flat lists of transitions for the entire batch (length = batch_size * t_max)
        self.global_state = []
        self.joint_action = []
        self.global_reward = []
        self.logps = []
        self.values = []
        if self.PO:
            self.observations = []
            self.fp_global_states = []
            self.prev_actions = []

        #per episode returns and advantages
        self.returns = []
        self.advantages = []

        self.n_eps = 0
        self.t_steps = 0

    def push(self, *, joint_action, global_reward, logps, values, global_state=None,
             observations=None, fp_global_states=None, prev_actions=None):
        """Add one timestep transition to the buffer. Transition variable shapes: (N = num_agents)
            **Communal**
            1.  joint_action: tensor, shape [N]
            2.  global_reward: float
            3.  logps: tensor, shape [N]

            **FO Specific**
            1. global_state: tensor, shape [1, state_dim]
            2. values: tensor, shape [1]

            **PO Specific**
            1. fp_global_states: dict length [N] of tensors shape [1, state_dim]
            2. local_states: dict length [N] of tensors shape [1, obs_dim]
            3. prev_actions: list length [N] of tensors shape [1, action_dim] (one-hot actions)
            4. values: tensor, shape [N]
            """

        self.joint_action.append(joint_action)
        self.global_reward.append(global_reward)
        self.logps.append(logps)
        self.values.append(values)

        if self.PO and observations is not None:
            self.observations.append(observations)
            self.fp_global_states.append(fp_global_states)
            self.prev_actions.append(prev_actions)
        else:
            self.global_state.append(global_state)

        self.t_steps += 1

    def process_episode(self, final_values=None):
        """Call at end of each episode. Compute GAE and discounted returns
            for the last t_steps transitions"""

        T = self.t_steps

        # stack rewards & values
        rewards = th.tensor(self.global_reward[-T:], dtype=th.float32, device=self.device)  # [T]
        if self.PO: #propegate rewards
            rewards_4adv = rewards.unsqueeze(-1).expand(T, self.N)     # [T, N]
        else:
            rewards_4adv = rewards.unsqueeze(-1)     # [T, 1]
        vals = th.stack(self.values[-T:], dim=0)  # [T, N] or [T,1]

        # bootstrap final value
        if final_values is None:
            final_values = th.zeros_like(vals[-1]) # [N] or [1]
        vals = th.cat([vals, final_values.unsqueeze(0)], dim=0)  # [T+1, N] or [T+1, 1]

        # compute advantage and return
        adv = th.zeros_like(rewards_4adv)        #[T, N] or [T, 1]
        gae = th.zeros_like(final_values)   #[N] or [1]
        ret = th.zeros_like(rewards)        #[T]
        R = 0.0

        for i in reversed(range(T)):
            delta = rewards_4adv[i] + self.gamma * vals[i+1] - vals[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[i] = gae

            R = rewards[i] + self.gamma * R
            ret[i] = R

            # print(f"step {i} | delta = {delta} | gae = {gae} | R = {R}")

        self.advantages.append(adv)  # for whole episode: [T,1] or [T,N]
        self.returns.append(ret)    # for whole episode: [T]
        self.n_eps += 1
        self.t_steps = 0




    def process_batch(self):
        """After batch_size episodes, reshape step‐level lists into
            [batch_size, t_max, ...] tensors, build a DataLoader, then clear buffer"""

        B, T, N = self.batch_size, self.t_max, self.N
        # print(th.stack(self.joint_action, dim=0).shape, f"bum, ep-{self.n_eps}")

        if self.PO:
            # POSIG stack episodes: [B, T, ...]
            actions = th.stack(self.joint_action, dim=0).view(B, T, N)
            logps = th.stack(self.logps, dim=0).view(B, T, N)
            values = th.stack(self.values, dim=0).view(B, T, N)
            returns = th.stack(self.returns, dim=0)        # [B, T, N]
            advantages = th.stack(self.advantages, dim=0)  # [B, T, N]

            # PO‐specific: obs, fpgs, prev
            obs = th.stack([
                th.stack([step[a].squeeze(0) for a in sorted(step.keys())], dim=0)
                for step in self.observations
            ], dim=0).view(B, T, N, self.obs_dim)

            fpgs = th.stack([
                th.stack([step[a].squeeze(0) for a in sorted(step.keys())], dim=0)
                for step in self.fp_global_states
            ], dim=0).view(B, T, N, self.state_dim)

            prev = th.stack([
                th.stack([a.squeeze(0) for a in step], dim=0)
                for step in self.prev_actions
            ], dim=0).view(B, T, N, self.action_dim)

            # dataset of full episodes
            dataset = th.utils.data.TensorDataset(actions, obs, fpgs, prev, logps, values, returns, advantages)
        else:
            # SIG: flatten across all time-steps [B*T, ...]
            states = th.stack(self.global_state, dim=0)  # [B*T, state_dim]
            actions = th.stack(self.joint_action, dim=0)  # [B*T, N]
            logps = th.stack(self.logps, dim=0)  # [B*T, N]
            values = th.stack(self.values, dim=0)  # [B*T, 1]
            returns = th.stack(self.returns, dim=0).view(-1, 1)  # [B*T, 1]
            advantages = th.stack(self.advantages, dim=0).view(-1, 1)  # [B*T, 1]

            dataset = th.utils.data.TensorDataset(states, actions, logps, values, returns, advantages)


        mini_batch_size = len(dataset) // self.num_mb
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

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



class zzBatchProcessing:
    def __init__(self):
        pass

    def collate_batch(self, batch_buffer, device):
        """process buffer into batch tensors once buffer is full"""
        batch_state = []
        batch_obs = []
        batch_action = []
        batch_prev_action = []
        batch_logp = []
        batch_value = []
        batch_rtrn = []
        batch_adv = []
        batch_h_a, batch_h_c = [], []

        # print(f"len(batch_buffer) = {len(batch_buffer)}")
        # print(f"len(batch_buffer[0][0]) = {len(batch_buffer[0][0])}")

        for data in batch_buffer:
            state, obs, action, prev_action, logp, value, rtrn, adv, h_a, h_c = data

            state = th.stack(state, dim=0).squeeze(1).to(device)  # Shape: [batch_size, state_dim]
            action = th.tensor(action).to(device) #Shape: [batch_size, num_agents]
            logp = th.tensor(logp).to(device)   #Shape: [batch_size, num_agents]
            if value[0].shape[0] == 1:  #SIG
                value = th.stack([v.squeeze() for v in value], dim=0).to(device)  # [T] → [T, 1]
            else:  # POSIG: list of [N,1] → want [batch_size, N]
                value = th.stack([v.squeeze(-1) for v in value], dim=0).to(device)  # [T, N]

            # print(f"state: {state.shape}\n"
            #       # f"obs: {obs[0]}\n"
            #       f"action: {action.shape}\n"
            #       # f"prev_action: {prev_action[0]}\n"
            #       f"logp: {logp.shape}\n"
            #       f"value: {value.shape}\n"
            #       f"rtrn: {rtrn.shape}\n"
            #       f"adv: {adv.shape}\n")

            batch_state.append(state)
            batch_action.append(action)
            batch_logp.append(logp)
            batch_value.append(value)
            batch_rtrn.append(rtrn)
            batch_adv.append(adv)


                # value = th.stack([
                #     th.stack([v.squeeze(0) for v in timestep], dim=0)  # [num_agents, action_dim]
                #     for timestep in value
                # ], dim=0)  # [batch_size, num_agents, action_dim]

            #POSIG ONLY
            if prev_action[0] is not None:
                prev_action = th.stack([
                    th.stack([a.squeeze(0) for a in timestep], dim=0)  # [num_agents, action_dim]
                    for timestep in prev_action
                ], dim=0)  # [batch_size, num_agents, action_dim]
                # print(f"prev_action: {prev_action.shape}")
                batch_prev_action.append(prev_action)

                # prev_action = th.stack(prev_action, dim=0).to(device)
            if obs[0] is not None:
                obs = th.stack([
                    th.stack([obs_t[agent_id].squeeze(0) for agent_id in sorted(obs_t.keys())], dim=0)
                    for obs_t in obs], dim=0)  # final shape: [batch_size, num_agents, obs_dim]
                # print(f"obs: {obs.shape}")
                batch_obs.append(obs)

            # print(f"ha: {h_a[0].shape}\n"
            #       f"hc: {h_c[0].shape}\n")
            # print(f"ha: {len(h_a)}\n"
            #       f"h_c: {len(h_c)}\n")
            # sys.exit()


            if h_a[0] is not None:
                h_a = th.stack(h_a, dim=0) #[batch, num_agents, hidden_dim]
                # h_a = th.stack([th.stack([h for h in time], dim=1)  # [num_agents, action_dim]
                #      for time in h_a], dim=0)

                # print(f"h_a.shape: {h_a.shape}")
                # sys.exit()
                # h_a = th.stack([th.stack([h for h in ], dim=0)  # [num_agents, action_dim]
                #     for timestep in h_a], dim=0)
                batch_h_a.append(h_a)
            if h_c[0] is not None:
                h_c = th.stack(h_c, dim=0) #[batch, num_agents, hidden_dim]
                batch_h_c.append(h_c)





        # convert to tensors
        batch_state = th.cat(batch_state, dim=0)
        batch_obs = th.cat(batch_obs, dim=0) if len(batch_obs) > 0 else None
        batch_action = th.cat(batch_action, dim=0)
        batch_prev_action = th.cat(batch_prev_action, dim=0) if len(batch_prev_action) > 0 else None
        batch_logp = th.cat(batch_logp, dim=0)
        batch_value = th.cat(batch_value, dim=0)
        batch_rtrn = th.cat(batch_rtrn, dim=0)
        batch_adv = th.cat(batch_adv, dim=0)
        batch_h_a = th.cat(batch_h_a, dim=0) if len(batch_h_a) > 0 else None
        batch_h_c = th.cat(batch_h_c, dim=0) if len(batch_h_c) > 0 else None

        # print(f"batch_h_a.shape: {batch_h_a.shape}")

        # normalize advantages
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()

        return (batch_state, batch_obs, batch_action, batch_prev_action,
                batch_logp, batch_value, batch_rtrn, batch_adv, batch_h_a, batch_h_c )



def _OLD_compute_GAE(individual_rewards, global_rewards, values, gamma, gae_lambda, device):
    """Compute GAE and returns for a single episode.
    Args:
        individual_rewards: Tensor, shape [T, N]
        values:             Tensor, shape [T] or [T, N]
    Returns:
        returns:    Tensor, shape [T, N] or [T]
        advantages: Tensor, shape [T, N] or [T]
    """

    individual_rewards = np.array(individual_rewards)
    individual_rewards = th.tensor(individual_rewards, dtype=th.float32, device=device) #shape [T, N]
    # print(f"individual_rewards: {individual_rewards.shape}\n"
    #       f"values[0].shape: {values[0].shape}\n")

    #format values / rewards as tensors:
    if values[0].shape[0] == 1: #SIG
        values = th.tensor(values, dtype=th.float32, device=device) #shape [T]
        values = th.cat([values, th.tensor([0.0])], dim=0) #shape [T+1]
        rewards = th.tensor(global_rewards, dtype=th.float32, device=device)   #shape [T]
        # rewards = individual_rewards.mean(dim=-1)   #shape [T]
    else:   #POSIG
        values = th.stack(values, dim=0).squeeze(-1).to(device) #shape [T, N]
        values = th.cat([values, values.new_zeros(1, values.size(1))], dim=0) #shape [T+1, N]
        rewards = individual_rewards    #shape [T, N]
    # print(f"values: {values.shape}\n"
    #       f"rewards: {rewards.shape}\n")


    # — initialize buffers for returns & advantages —
    T = rewards.size(0)
    if rewards.dim() == 1:
        # single-vector case
        returns    = th.zeros(T,    dtype=th.float32, device=device)
        advantages = th.zeros(T,    dtype=th.float32, device=device)
        gae = 0.0
        R   = 0.0
    else:
        # multi-agent case
        N = rewards.size(1)
        returns    = th.zeros((T, N), dtype=th.float32, device=device)
        advantages = th.zeros((T, N), dtype=th.float32, device=device)
        gae = th.zeros(N, device=device)
        R   = th.zeros(N, device=device)

    # — backward pass —
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

        R = rewards[t] + gamma * R
        returns[t] = R

    return returns, advantages


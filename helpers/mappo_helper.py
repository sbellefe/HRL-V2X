import torch as th
import numpy as np
import matplotlib.pyplot as plt
import sys

def pre_process(obs):   #NOT USED
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
        self.plot = True

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
        self.t_tot, self.ep_tot = 0, 0
        self.n_roll = 0

        #Env params stored here, I see why it is annoying to have them all attached to enviro
        self.V2V_power_dB_list = [23, 15, 5]
        self.num_power_levels = len(self.V2V_power_dB_list)
        self.num_SC = 4


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
        self.t_tot += 1

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
        self.ep_tot += 1
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

            # normalize advantages
            adv = advantages.flatten()
            adv = (adv - adv.mean()) / adv.std()
            advantages = adv.view(B, T, N)

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

            # normalize advantages
            adv = advantages.flatten()
            adv = (adv - adv.mean()) / adv.std()
            advantages = adv.view(B, T, N)


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

            # normalize advantages
            adv = advantages.squeeze(-1)  # [B*T]
            adv = (adv - adv.mean()) / adv.std()
            advantages = adv.unqueeze(-1)

            dataset = th.utils.data.TensorDataset(states, actions, logps, values, returns, advantages)



        mini_batch_size = len(dataset) // self.num_mb
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        if self.plot:
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
        P = self.num_power_levels
        S = self.num_SC
        null_action = self.action_dim - 1

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

                    # # 3) Actions as scatter
                    # ax = axes[3]
                    # ax.set_ylabel("Agent → Action")
                    # # meta-parameters
                    # P = self.num_power_levels
                    # power_list = self.V2V_power_dB_list
                    # null_action = self.action_dim - 1
                    # # flatten for scatter
                    # times = np.repeat(time, N)
                    # agents = np.tile(np.arange(N), B*T)
                    # acts = actions_series.flatten()
                    # # compute channel & power index arrays
                    # sc = (acts // P)
                    # pw_idx = (acts % P)
                    #
                    # # choose a distinct marker per power‐level
                    # markers = ['o', 's', '^', 'D', 'v'][:P]  # extend if P>4
                    # # scatter each (channel, power) combination
                    # for ch in np.unique(sc[sc >= 0]):
                    #     for pi, pw in enumerate(power_list):
                    #         mask = (sc == ch) & (pw_idx == pi)
                    #         if not mask.any(): continue
                    #         ax.scatter(
                    #             times[mask],
                    #             agents[mask],
                    #             c=[f"C{ch}"],  # one color per channel
                    #             marker=markers[pi],  # shape per power
                    #             s=20,
                    #             label=f"ch{ch}, {pw}dB"
                    #         )
                    # # null‐action
                    # null_mask = (acts == null_action)
                    # if null_mask.any():
                    #     ax.scatter(
                    #         times[null_mask],
                    #         agents[null_mask],
                    #         c='lightgray',
                    #         marker='x',
                    #         s=20,
                    #         label="no-transmit"
                    #     )
                    #
                    # ax.set_yticks(np.arange(N))
                    # ax.set_yticklabels([f"{i}" for i in range(N)])
                    # ax.legend(fontsize='small', ncol=4, loc='upper right')

        # (batch_states, batch_actions, batch_pi_hat,
        #  batch_options, batch_prev_options,
        #  batch_v_h, batch_v_l, batch_logp_h, batch_logp_l,
        #  batch_rtrn, batch_adv_h, batch_adv_l, batch_pi_bar, batch_betas) = processed_buffer
        #
        # fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        # fig.suptitle(f"Buffer {self.counter} Data Visualization")#, fontsize=16)
        #
        # # High-Level Policy Distribution (pi_hat) - Show average probability per option
        # avg_pi_hat = batch_pi_hat.mean(dim=0).cpu().numpy()
        # axes[0, 0].bar(range(len(avg_pi_hat)), avg_pi_hat, color='blue')
        # axes[0, 0].set_title("High-Level Policy Distribution (pi_hat)")
        # axes[0, 0].set_xlabel("Option Index")
        # axes[0, 0].set_ylabel("Avg Probability")
        #
        # # Log Prob of Selected Options (logp_h), Actions (logp_l)
        # axes[0, 1].hist(batch_logp_h.cpu().numpy(), bins=30, color='red', alpha=0.5, label="Selected Options (logp_h)")
        # axes[0, 1].hist(batch_logp_l.cpu().numpy(), bins=30, color='blue', alpha=0.5, label="Selected Actions (logp_l)")
        # axes[0, 1].set_title("Log Probability of")
        # axes[0, 1].set_xlabel("Log Probability")
        # axes[0, 1].set_ylabel("Frequency")
        # axes[0,1].legend()
        #
        # #Plot option Terminations: batch_betas [batch_size, 4]
        # for i in range(batch_betas.shape[1]):
        #     axes[0, 2].plot(range(len(batch_betas)), batch_betas[:, i].cpu().numpy(),label=f"Option {i}", alpha=0.7,  linestyle='dotted')
        # axes[0, 2].set_title("Option Termination Probabilities (betas)")
        # axes[0, 2].set_ylabel("Termination Probability")
        # axes[0, 2].set_xlabel("Buffer Index")
        # axes[0, 2].legend()
        #
        # # High-Level Advantage Estimates (Adv_H)
        # axes[1, 0].plot(batch_adv_h.cpu().numpy(), color='blue')
        # axes[1, 0].set_title("High MDP Advantage Estimates (Adv_H)")
        # axes[1, 0].set_xlabel("Buffer index")
        # axes[1, 0].set_ylabel("Advantage Value")
        #
        # # Low-Level Advantage Estimates (Adv_L)
        # axes[1, 1].plot(batch_adv_l.cpu().numpy(), color='green')
        # axes[1, 1].set_title("Low MDP Advantage Estimates (Adv_L)")
        # axes[1, 1].set_xlabel("Buffer index")
        # axes[1, 1].set_ylabel("Advantage Value")
        #
        # # Discounted Returns (batch_rtrn)
        # axes[1, 2].plot(batch_rtrn.cpu().numpy(), color='orange')
        # axes[1, 2].set_title("Discounted Returns")
        # axes[1, 2].set_xlabel("Buffer index")
        # axes[1, 2].set_ylabel("Return Value")
        #
        # # High-Level Value Function Estimates (V_H)
        # axes[2, 0].plot(batch_v_h.cpu().numpy(), color='purple')
        # axes[2, 0].set_title("High-Level Value Function Estimates (V_H)")
        # axes[2, 0].set_xlabel("Buffer index")
        # axes[2, 0].set_ylabel("Value")
        #
        # # Low-Level Value Function Estimates (V_L)
        # axes[2, 1].plot(batch_v_l.cpu().numpy(), color='brown')
        # axes[2, 1].set_title("Low-Level Value Function Estimates (V_L)")
        # axes[2, 1].set_xlabel("Buffer index")
        # axes[2, 1].set_ylabel("Value")
        #
        # axes[2, 2].scatter(range(len(batch_options)), batch_options.cpu().numpy(), label="Current Option", alpha=0.7, s=50)
        # axes[2, 2].scatter(range(len(batch_prev_options)), batch_prev_options.cpu().numpy(),label="Previous Option", alpha=0.5, s=4)
        # axes[2, 2].set_title("Option Switching Behavior")
        # axes[2, 2].set_xlabel("Buffer Index")
        # axes[2, 2].set_ylabel("Option Index")
        # axes[2, 2].legend()
        #
        # plt.tight_layout(rect=(0., 0., 1., 0.97))
        # plt.show()



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


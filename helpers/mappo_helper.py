import torch as th
import numpy as np
import matplotlib.pyplot as plt
import sys


class RolloutBuffer:
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
                final_values = th.zeros_like(values[-1:])  # [N]
            else:
                final_values = final_values.unsqueeze(0)

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

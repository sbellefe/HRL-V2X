import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys


def compute_pi_hat(logits, betas, prev_option):
    #get master policy probabilities from logits
    pi_W = F.softmax(logits, dim=1) #[batch_size, num_options]

    # normalize prev_option to a 1-D tensor of length B
    if prev_option.dim() == 0:
        prev_option = prev_option.unsqueeze(0)  # [1]

    #address case of initial timestep (dummy option)
    if (prev_option < 0).all():
        return pi_W

    # get β_prev: shape [B, 1]
    beta_prev = betas.gather(1, prev_option.clamp(min=0).unsqueeze(1))

    # one-hot mask for the previous option (negatives become a dummy 0th row, but won't be used)
    mask = F.one_hot(prev_option.clamp(min=0), num_classes=pi_W.shape[1])  # [B, K]

    # full one-line formula:
    #   if prev<0:       π̂ = π_W
    #   otherwise:       π̂ = β_prev·π_W + (1−β_prev)·e_prev
    pi_hat = th.where(prev_option.unsqueeze(1) < 0,
                      pi_W,
                      beta_prev * pi_W + (1 - beta_prev) * mask
                      )
    return pi_hat  # tensor.shape[B, K]


def Xcompute_pi_hat(prediction, prev_option):
    """computes high-policy (option selection) based on previous option, master policy & beta outputs.
        in the initial state, prev_option is set to '-1', function returns the master policy without
        beta contribution. B = batch_size. K = num_options
            prediction: dict with
              - 'pi_W': Tensor [B, K]  master-policy probs
              - 'betas': Tensor [B, K] termination probs per option
            prev_option: LongTensor of shape [B] (or scalar) with values in {-1, 0..K-1}.
                         A value < 0 means “no previous option” (initial).
            Returns:
              pi_hat: Tensor [B, K]
        """

    #get high actor option probabilities
    pi_W = prediction['pi_W']       #[B, K]
    beta  = prediction['betas']   # [B, K]
    B, K  = pi_W.shape

    # normalize prev_option to a 1-D tensor of length B
    if prev_option.dim() == 0:
        prev_option = prev_option.unsqueeze(0)  # [1]

    # case: all “initial” → just return master-policy
    if (prev_option < 0).all():
        return pi_W

    # get β_prev: shape [B, 1]
    beta_prev = beta.gather(1, prev_option.clamp(min=0).unsqueeze(1))

    # one-hot mask for the previous option (negatives become a dummy 0th row, but won't be used)
    mask = F.one_hot(prev_option.clamp(min=0), num_classes=K).to(pi_W.dtype)  # [B, K]

    # full one-line formula:
    #   if prev<0:       π̂ = π_W
    #   otherwise:       π̂ = β_prev·π_W + (1−β_prev)·e_prev
    pi_hat = th.where(prev_option.unsqueeze(1) < 0,
        pi_W,
        beta_prev * pi_W + (1 - beta_prev) * mask
    )
    return pi_hat #tensor.shape[B, K]

def ZZcompute_pi_hat(prediction, prev_option):
    """computes high-policy (option selection) based on previous option, master policy & beta outputs.
        in the initial state, prev_option is set to None, function returns the master policy without
        beta contribution."""

    # get high actor option probabilities
    pi_W = prediction['pi_W']

    # Logic for initial state.
    if prev_option is None:
        return prediction['pi_W']

    # get option termination probabilities
    beta = prediction['betas']  # [batch_size, num_options]

    # create mask for the previous option(s)
    mask = th.zeros_like(pi_W)
    # mask[th.arange(pi_W.size(0)), prev_option] = 1
    mask.scatter_(1, prev_option.unsqueeze(1), 1)

    # Extract only the termination probability for the previously active option.
    beta_prev = beta.gather(1, prev_option.unsqueeze(1))  # Shape: [batch_size, 1]

    # compute pi_hat by factoring in beta contribution
    # pi_hat = (1 - beta) * mask + beta * pi_W
    pi_hat = (1 - beta_prev) * mask + beta_prev * pi_W

    return pi_hat  # tensor.shape[batch_size, 4]


class RolloutBuffer_FO:
    def __init__(self, params):
        self.device = params.device
        self.batch_size = params.buffer_episodes    #episodes per batch
        self.t_max = params.t_max                   #timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches   #for RNN
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_options = params.num_options

        #preallocate tensors, flat transitions for the entire batch (length = batch_size * t_max)
        BT, N = self.batch_size * self.t_max, self.N
        self.global_states = th.zeros((BT, self.state_dim), device=self.device)
        self.joint_actions = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.joint_options = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.global_rewards = th.zeros((BT,), device=self.device)
        self.prev_options = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.pi_hats = th.zeros((BT, N, self.num_options), device=self.device)
        self.values_h = th.zeros((BT, N), device=self.device)
        self.values_l = th.zeros((BT, N), device=self.device)
        self.logps_h = th.zeros((BT, N), device=self.device)
        self.logps_l = th.zeros((BT, N), device=self.device)


        self.advantages_h = None  # will be [BT,N]
        self.advantages_l = None  # will be [BT,N]
        self.returns = None     # will be [BT,]

        # internal write indices
        self.t_steps = 0  # 0..BT-1 within a batch
        self.n_eps = 0  # 0..B-1
        self.ep_tot, self.n_roll = 0, 0  # multiple rollout counters

        # reuse dataset / dataloader
        self.dataset = None
        self.dataloader = None

    def push(self, *args):
        s, a, o, r, prev_o, pi_hat, v_h, v_l, lp_h, lp_l = args
        idx = self.t_steps
        self.global_states[idx] = s
        self.joint_actions[idx] = a
        self.joint_options[idx] = o
        self.global_rewards[idx] = r
        self.prev_options[idx] = prev_o
        self.pi_hats[idx] = pi_hat
        self.values_h[idx] = v_h
        self.values_l[idx] = v_l
        self.logps_h[idx] = lp_h
        self.logps_l[idx] = lp_l

        #increment counters
        self.t_steps += 1
        if self.t_steps % self.t_max == 0:
            self.n_eps += 1
            self.ep_tot += 1

    def process_batch(self):
        self.compute_GAE()

        s = self.global_states
        a = self.joint_actions
        o = self.joint_options
        prev_o = self.prev_options
        pi_hat = self.pi_hats
        v_h = self.values_h
        v_l = self.values_l
        lp_h = self.logps_h
        lp_l = self.logps_l
        rtrn = self.returns
        adv_h = self.advantages_h
        adv_l = self.advantages_l
        tensors = s, a, o, prev_o, pi_hat, v_h, v_l, lp_h, lp_l, rtrn, adv_h, adv_l

        # prepare dataset and dataloader once
        if self.dataset is None:
            self.dataset = th.utils.data.TensorDataset(*tensors)
            mb_size = len(self.dataset) // self.num_mb
            self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=mb_size, shuffle=True)
        else:
            # overwrite tensors in-place
            self.dataset.tensors = tensors

        # for ten in tensors:
        #     print(ten.shape)

        # reset for next batch
        self.t_steps = 0
        self.n_eps = 0
        self.n_roll += 1

        return self.dataloader

    def compute_GAE(self):
        B, T, N = self.batch_size, self.t_max, self.N

        #reshape variables
        r = self.global_rewards.view(B, T)
        v_h = self.values_h.view(B, T, N)
        v_l = self.values_l.view(B, T, N)

        # pad values with zeros for last state [T]→[T+1]
        fv = th.zeros((B, 1, N), device=self.device)
        v_h_pad = th.cat([v_h, fv], dim=1)  # [B, T+1, N]
        v_l_pad = th.cat([v_l, fv], dim=1)  # [B, T+1, N]

        # Compute deltas
        delta_h = r.unsqueeze(-1) + self.gamma * v_h_pad[:, 1:] - v_h
        delta_l = r.unsqueeze(-1) + self.gamma * v_l_pad[:, 1:] - v_l

        # Allocate advantage & return arrays in episode form
        adv_h = th.zeros_like(delta_h)    # [B, T, N]
        adv_l = th.zeros_like(delta_l)    # [B, T, N]
        ret = th.zeros_like(r)          # [B, T]

        #Backward pass: for each t = T-1 … 0
        for t in reversed(range(T)):
            if t == T - 1:
                # terminal step: no future advantage or return
                adv_h[:, t] = delta_h[:, t]
                adv_l[:, t] = delta_l[:, t]
                ret[:, t] = r[:, t]
            else:
                # GAE recurrence and discounted return
                adv_h[:, t] = (delta_h[:, t] + self.gamma * self.gae_lambda * adv_h[:, t+1])
                adv_l[:, t] = (delta_l[:, t] + self.gamma * self.gae_lambda * adv_l[:, t+1])
                ret[:, t] = r[:, t] + self.gamma * ret[:, t+1]

        #Flatten back to [B*T, …] for DataLoader
        adv_h = adv_h.view(B*T, N)
        adv_l = adv_l.view(B*T, N)
        ret = ret.view(B*T)

        self.advantages_h = (adv_h - adv_h.mean()) / (adv_h.std() + 1e-8)
        self.advantages_l = (adv_l - adv_l.mean()) / (adv_l.std() + 1e-8)
        self.returns = (ret - ret.mean()) / (ret.std() + 1e-8)

    def plot_rollout(self):
        returns = self.returns.cpu().numpy()       # shape [BT,]
        adv_h = self.advantages_h.cpu().numpy()     # shape [BT, N]
        adv_l = self.advantages_l.cpu().numpy()     # shape [BT, N]
        v_h = self.values_h.cpu().numpy()     # shape [BT, N]
        v_l = self.values_l.cpu().numpy()     # shape [BT, N]
        pi_h = self.pi_hats.cpu().numpy()           # shape [BT, N, num_options]

        fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharex=True)

        axes[0].plot(returns)
        axes[0].set_title("Batch Returns")

        for agent_idx in range(self.N):
            axes[1].plot(v_h[:, agent_idx], label=f"Agent {agent_idx}")
        axes[1].set_title("High MDP Value")

        for agent_idx in range(self.N):
            axes[2].plot(v_l[:, agent_idx], label=f"Agent {agent_idx}")
        axes[2].set_title("Low MDP Value")

        for agent_idx in range(self.N):
            axes[3].plot(adv_h[:, agent_idx], label=f"Agent {agent_idx}")
        axes[3].set_title("High MDP Advantage")

        for agent_idx in range(self.N):
            axes[4].plot(adv_l[:, agent_idx], label=f"Agent {agent_idx}")
        axes[4].set_title("Low MDP Advantage")

        for agent_idx in range(self.N):
            axes[5].hist(v_l[:, agent_idx, :], bins=self.num_options, label=f"Agent {agent_idx}")
        axes[5].set_title("Option Probabilities")




    def zplot_buffer(self, processed_buffer):
        (batch_states, batch_actions, batch_pi_hat,
         batch_options, batch_prev_options,
         batch_v_h, batch_v_l, batch_logp_h, batch_logp_l,
         batch_rtrn, batch_adv_h, batch_adv_l, batch_pi_bar, batch_betas) = processed_buffer

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"Buffer {self.counter} Data Visualization")#, fontsize=16)

        # High-Level Policy Distribution (pi_hat) - Show average probability per option
        avg_pi_hat = batch_pi_hat.mean(dim=0).cpu().numpy()
        axes[0, 0].bar(range(len(avg_pi_hat)), avg_pi_hat, color='blue')
        axes[0, 0].set_title("High-Level Policy Distribution (pi_hat)")
        axes[0, 0].set_xlabel("Option Index")
        axes[0, 0].set_ylabel("Avg Probability")

        # Log Prob of Selected Options (logp_h), Actions (logp_l)
        axes[0, 1].hist(batch_logp_h.cpu().numpy(), bins=30, color='red', alpha=0.5, label="Selected Options (logp_h)")
        axes[0, 1].hist(batch_logp_l.cpu().numpy(), bins=30, color='blue', alpha=0.5, label="Selected Actions (logp_l)")
        axes[0, 1].set_title("Log Probability of")
        axes[0, 1].set_xlabel("Log Probability")
        axes[0, 1].set_ylabel("Frequency")
        axes[0,1].legend()

        #Plot option Terminations: batch_betas [batch_size, 4]
        for i in range(batch_betas.shape[1]):
            axes[0, 2].plot(range(len(batch_betas)), batch_betas[:, i].cpu().numpy(),label=f"Option {i}", alpha=0.7,  linestyle='dotted')
        axes[0, 2].set_title("Option Termination Probabilities (betas)")
        axes[0, 2].set_ylabel("Termination Probability")
        axes[0, 2].set_xlabel("Buffer Index")
        axes[0, 2].legend()

        # High-Level Advantage Estimates (Adv_H)
        axes[1, 0].plot(batch_adv_h.cpu().numpy(), color='blue')
        axes[1, 0].set_title("High MDP Advantage Estimates (Adv_H)")
        axes[1, 0].set_xlabel("Buffer index")
        axes[1, 0].set_ylabel("Advantage Value")

        # Low-Level Advantage Estimates (Adv_L)
        axes[1, 1].plot(batch_adv_l.cpu().numpy(), color='green')
        axes[1, 1].set_title("Low MDP Advantage Estimates (Adv_L)")
        axes[1, 1].set_xlabel("Buffer index")
        axes[1, 1].set_ylabel("Advantage Value")

        # Discounted Returns (batch_rtrn)
        axes[1, 2].plot(batch_rtrn.cpu().numpy(), color='orange')
        axes[1, 2].set_title("Discounted Returns")
        axes[1, 2].set_xlabel("Buffer index")
        axes[1, 2].set_ylabel("Return Value")

        # High-Level Value Function Estimates (V_H)
        axes[2, 0].plot(batch_v_h.cpu().numpy(), color='purple')
        axes[2, 0].set_title("High-Level Value Function Estimates (V_H)")
        axes[2, 0].set_xlabel("Buffer index")
        axes[2, 0].set_ylabel("Value")

        # Low-Level Value Function Estimates (V_L)
        axes[2, 1].plot(batch_v_l.cpu().numpy(), color='brown')
        axes[2, 1].set_title("Low-Level Value Function Estimates (V_L)")
        axes[2, 1].set_xlabel("Buffer index")
        axes[2, 1].set_ylabel("Value")

        axes[2, 2].scatter(range(len(batch_options)), batch_options.cpu().numpy(), label="Current Option", alpha=0.7, s=50)
        axes[2, 2].scatter(range(len(batch_prev_options)), batch_prev_options.cpu().numpy(),label="Previous Option", alpha=0.5, s=4)
        axes[2, 2].set_title("Option Switching Behavior")
        axes[2, 2].set_xlabel("Buffer Index")
        axes[2, 2].set_ylabel("Option Index")
        axes[2, 2].legend()

        plt.tight_layout(rect=(0., 0., 1., 0.97))
        plt.show()
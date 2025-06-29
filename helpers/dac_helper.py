import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys

def compute_pi_hat(prediction, prev_option):
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


        BT, N = self.batch_size * self.t_max, self.N
        # Flat lists of transitions for the entire batch (length = batch_size * t_max)
        self.global_state = th.zeros((BT, self.state_dim), device=self.device)
        self.action = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.global_reward = th.zeros((BT,), device=self.device)
        self.option = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.prev_option = th.zeros((BT, N), dtype=th.long, device=self.device)
        self.pi_hat = th.zeros((BT, N, self.num_options), device=self.device)
        self.value_h = th.zeros((BT, N), device=self.device)
        self.value_l = th.zeros((BT, N), device=self.device)
        self.logp_h = th.zeros((BT, N), device=self.device)
        self.logp_l = th.zeros((BT, N), device=self.device)

        # placeholders for computed advantages and returns
        self.adv_h = None  # will be [BT, N]
        self.adv_l = None
        self.returns = None

        # internal write indices
        self.t_steps = 0  # 0..BT-1 within a batch
        # self.ep_steps = 0  # 0..T-1
        self.n_eps = 0  # 0..B-1

        #multiple rollout counters
        self.ep_tot, self.n_roll = 0, 0

        # reuse dataset / dataloader
        self.dataset = None
        self.dataloader = None


    def push(self, *args):
        s, a, r, o, prev_o, pi_h, v_h, v_l, lp_h, lp_l = args

        # print(f"\n ****** Episode {self.n_eps}")
        # print('s.shape', s.shape)
        # print('a.shape', a.shape)
        # print('r.shape', r.shape)
        # print('o.shape', o.shape)
        # print('prev_o.shape', prev_o.shape)
        # print('pi_h.shape', pi_h.shape)
        # print('v_h.shape', v_h.shape)
        # print('v_l.shape', v_l.shape)
        # print('lp_h.shape', lp_h.shape)
        # print('lp_l.shape', lp_l.shape)

        #flat variables
        idx = self.t_steps
        self.global_state[idx] = s.squeeze(0)
        self.action[idx] = a
        self.global_reward[idx] = r
        self.option[idx] = o
        self.prev_option[idx] = prev_o
        self.value_h[idx] = v_h
        self.value_l[idx] = v_l
        self.logp_h[idx] = lp_h
        self.logp_l[idx] = lp_l

        # idx = (self.n_eps, self.ep_steps)



        #increment counters
        self.t_steps += 1
        # self.ep_steps += 1
        if self.t_steps % self.t_max == 0:
            self.n_eps += 1
            # self.ep_steps = 0
            self.ep_tot += 1



    def process_batch(self):
        # B, T, N = self.batch_size, self.t_max, self.N

        self.compute_GAE()

        #NOW prepare dataset/loader with flattened tensors
        s = self.global_state
        a = self.action
        o = self.option
        prev_o = self.prev_option
        pi_h = self.pi_hat
        v_h = self.value_h
        v_l = self.value_l
        lp_h = self.logp_h
        lp_l = self.logp_l
        rtrn = self.returns
        adv_h = self.adv_h
        adv_l = self.adv_l

        print('s.shape', s.shape)
        print('a.shape', a.shape)
        print('o.shape', o.shape)
        print('prev_o.shape', prev_o.shape)
        print('pi_h.shape', pi_h.shape)
        print('v_h.shape', v_h.shape)
        print('v_l.shape', v_l.shape)
        print('lp_h.shape', lp_h.shape)
        print('lp_l.shape', lp_l.shape)
        print('rtrn.shape', rtrn.shape)
        print('adv_h.shape', adv_h.shape)
        print('adv_l.shape', adv_l.shape)

        # prepare dataset and dataloader once
        if self.dataset is None:
            self.dataset = th.utils.data.TensorDataset(s, a, o, prev_o, pi_h,
                                                       v_h, v_l, lp_h, lp_l, rtrn, adv_h, adv_l)
            mb_size = len(self.dataset) // self.num_mb
            self.dataloader = th.utils.data.DataLoader(self.dataset, batch_size=mb_size, shuffle=True)
        else:
            # overwrite tensors in-place
            self.dataset.tensors = (s, a, o, prev_o, pi_h, v_h, v_l, lp_h, lp_l, rtrn, adv_h, adv_l)

        # reset for next batch
        self.t_steps = 0
        self.n_eps = 0
        self.n_roll += 1

        return self.dataloader

    def compute_GAE(self):
        """"""

        B, T, N = self.batch_size, self.t_max, self.N
        BT = B * T

        #reshape variables
        rewards = self.global_reward.view(B, T)
        vh = self.value_h.view(B, T, N)
        vl = self.value_l.view(B, T, N)

        # pad values with zeros for last state [T]→[T+1]
        final_values = th.zeros((B, 1, N), device=self.device)
        vh_pad = th.cat([vh, final_values], dim=1)  # [B, T+1, N]
        vl_pad = th.cat([vl, final_values], dim=1)  # [B, T+1, N]

        # Compute deltas, broadcast rewards → [B, T, 1] so shapes line up
        delta_h = (rewards.unsqueeze(-1)  # [B, T, 1]
                + self.gamma * vh_pad[:, 1:, :]  # [B, T, N]
                - vh  # [B, T, N]
        )
        delta_l = (rewards.unsqueeze(-1)
                + self.gamma * vl_pad[:, 1:, :]
                - vl
        )

        # Allocate advantage & return arrays in episode form
        adv_h = th.zeros_like(delta_h)  # [B, T, N]
        adv_l = th.zeros_like(delta_l)
        ret = th.zeros((B, T), device=self.device)

        #Backward pass: for each t = T-1 … 0
        for t in reversed(range(T)):
            if t == T - 1:
                # terminal step: no future advantage or return
                adv_h[:, t] = delta_h[:, t]
                adv_l[:, t] = delta_l[:, t]
                ret[:, t] = rewards[:, t]
            else:
                # GAE recurrence
                adv_h[:, t] = (delta_h[:, t] + self.gamma * self.gae_lambda * adv_h[:, t + 1])
                adv_l[:, t] = (delta_l[:, t] + self.gamma * self.gae_lambda * adv_l[:, t + 1])
                # discounted return
                ret[:, t] = rewards[:, t] + self.gamma * ret[:, t + 1]

        # 6) Flatten back to [B*T, …] for DataLoader
        self.adv_h = adv_h.view(BT, N)  # [B*T, N]
        self.adv_l = adv_l.view(BT, N)
        self.returns = ret.view(BT)  # [B*T]

        # normalize
        self.adv_h = (self.adv_h - self.adv_h.mean()) / (self.adv_h.std() + 1e-8)
        self.adv_l = (self.adv_l - self.adv_l.mean()) / (self.adv_l.std() + 1e-8)
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)


class RolloutBufferZ:
    def __init__(self, params):
        self.device = params.device
        self.PO = params.partial_observability
        self.batch_size = params.buffer_episodes  # episodes per batch
        self.t_max = params.t_max  # timesteps per episode
        self.N = params.num_agents
        self.num_mb = params.num_mini_batches  # for RNN
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.state_dim = params.state_dim
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim

        # Flat lists of transitions for the entire batch (length = batch_size * t_max)
        if not self.PO:
            self.global_state = []
            self.action = []
            self.global_reward = []
            self.option = []
            self.prev_option = []
            # self.beta = []
            self.pi_hat = []
            self.value_h = []
            self.value_l = []
            self.logp_h = []
            self.logp_l = []
        else:
            pass

        # per episode returns and advantages
        self.returns = []
        self.adv_h = []
        self.adv_l = []

        self.n_eps = 0
        self.t_steps = 0
        self.t_tot, self.ep_tot = 0, 0
        self.n_roll = 0

    def push(self, *args):
        if not self.PO:
            s, a, r, o, prev_o, pi_h, v_h, v_l, lp_h, lp_l = args
            self.global_state.append(s)  # tensor shape [1, state_dim]
            self.action.append(a)  # tensor shape [N]
            self.global_reward.append(r)  # float
            self.option.append(o)  # tensor shape [N]
            self.prev_option.append(prev_o)  # tensor shape [N]
            self.pi_hat.append(pi_h)  # tensor shape [N, num_options]
            self.value_h.append(v_h)  # tensor shape [N]
            self.value_l.append(v_l)  # tensor shape [N]
            self.logp_h.append(lp_h)  # tensor shape [N]
            self.logp_l.append(lp_l)  # tensor shape [N]
        else:
            pass

        self.t_steps += 1
        self.t_tot += 1

    def process_episode(self):
        """Call at end of each episode. Compute GAE and discounted returns
            for the last t_steps transitions."""

        # final_values = None
        T, N = self.t_steps, self.N

        # get the last T rewards as tensor
        rewards = th.tensor(self.global_reward[-T:], dtype=th.float32, device=self.device)  # [T]

        # final_values = None
        if not self.PO:
            value_h = th.cat(self.value_h[-T:], dim=0)  # [T, N]
            value_l = th.cat(self.value_l[-T:], dim=0)  # [T, N]

            final_value_h = th.zeros_like(value_h[-1:])  # [N]
            final_value_l = th.zeros_like(value_l[-1:])  # [N]

            adv_h = th.zeros_like(value_h)  # [T, N]
            adv_l = th.zeros_like(value_l)  # [T, N]
            gae_h = th.zeros_like(final_value_h)  # [N]
            gae_l = th.zeros_like(final_value_h)  # [N]
            ret = th.zeros_like(rewards)  # [T]

            value_h = th.cat([value_h, final_value_h], dim=0)  # [T+1, N]
            value_l = th.cat([value_l, final_value_l], dim=0)  # [T+1, N]

            R = 0.0
            for i in reversed(range(T)):
                delta_h = rewards[i] + self.gamma * value_h[i + 1] - value_h[i]
                delta_l = rewards[i] + self.gamma * value_l[i + 1] - value_l[i]

                gae_h = delta_h + self.gamma * self.gae_lambda * gae_h
                gae_l = delta_l + self.gamma * self.gae_lambda * gae_l

                adv_h[i] = gae_h
                adv_l[i] = gae_l

                R = rewards[i] + self.gamma * R
                ret[i] = R

            self.adv_h.append(adv_h)  # [T, N]
            self.adv_l.append(adv_l)  # [T, N]
            self.returns.append(ret)  # [T]

        else:
            pass
        self.n_eps += 1
        self.ep_tot += 1
        self.t_steps = 0

    def process_batch(self):
        B, T, N = self.batch_size, self.t_max, self.N

        if not self.PO:
            # SIG: flatten across all time-steps [B*T, ...]
            s = th.stack(self.global_state, dim=0).squeeze(1)
            a = th.stack(self.action, dim=0)  # [B*T, N]
            o = th.stack(self.option, dim=0)  # [B*T, N]
            prev_o = th.stack(self.prev_option, dim=0)  # [B*T, N]
            # b = th.stack(self.beta, dim=0)  # [B*T, N, num_options]
            pi_h = th.stack(self.pi_hat, dim=0)  # [B*T, N, num_options]
            v_h = th.stack(self.value_h, dim=0)  # [B*T, N]
            v_l = th.stack(self.value_l, dim=0)  # [B*T, N]
            lp_h = th.stack(self.logp_h, dim=0)  # [B*T, N]
            lp_l = th.stack(self.logp_l, dim=0)  # [B*T, N]
            rtrn = th.stack(self.returns, dim=0).view(-1)  # [B*T]
            adv_h = th.stack(self.adv_h, dim=0).view(B * T, N)  # [B*T, N]
            adv_l = th.stack(self.adv_l, dim=0).view(B * T, N)  # [B*T, N]

            # print('s.shape', s.shape)
            # print('a.shape', a.shape)
            # print('o.shape', o.shape)
            # print('prev_o.shape', prev_o.shape)
            # print('pi_h.shape', pi_h.shape)
            # print('v_h.shape', v_h.shape)
            # print('v_l.shape', v_l.shape)
            # print('lp_h.shape', lp_h.shape)
            # print('lp_l.shape', lp_l.shape)
            # print('rtrn.shape', rtrn.shape)
            # print('adv_h.shape', adv_h.shape)
            # print('adv_l.shape', adv_l.shape)

            # normalize
            adv_h = (adv_h - adv_h.mean()) / (adv_h.std() + 1e-8)
            adv_l = (adv_l - adv_l.mean()) / (adv_l.std() + 1e-8)
            rtrn = (rtrn - rtrn.mean()) / (rtrn.std() + 1e-8)

            dataset = th.utils.data.TensorDataset(s, a, o, prev_o, pi_h,
                                                  v_h, v_l, lp_h, lp_l, rtrn, adv_h, adv_l)
            mb_size = len(dataset) // self.num_mb
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mb_size, shuffle=True)
        else:
            pass

        self.reset_buffer()

        return dataloader

def reset_buffer(self):
    #loop through all attributes, if list, then clear
    for name, value in vars(self).items():
        if isinstance(value, list):
            value.clear()

    self.n_eps = 0
    self.n_roll += 1



def compute_GAE(rewards, v_h, v_l, gamma, gae_lambda, device):
    adv_h, adv_l, returns = [],[],[]
    R, gae_h, gae_l = 0,0,0 #set final next state advantages and return = 0

    for t in reversed(range(len(rewards))):
        #compute TD errors
        delta_h = rewards[t] + gamma * v_h[t + 1] - v_h[t]
        delta_l = rewards[t] + gamma * v_l[t + 1] - v_l[t]

        #compute GAE advantages
        gae_h = delta_h + gamma * gae_lambda * gae_h
        gae_l = delta_l + gamma * gae_lambda * gae_l

        #Compute discounted return. only immediate reward if t is terminal state
        R = rewards[t] + gamma * R

        #store advantage and return in list
        adv_h.insert(0, gae_h)
        adv_l.insert(0, gae_l)
        returns.insert(0, R)

    #convert lists to tensors
    returns = [th.tensor(agent_returns) for agent_returns in returns]
    returns = th.stack(returns).to(device)
    adv_h = th.stack(adv_h).to(device)
    adv_l = th.stack(adv_l).to(device)

    # remove final next state values from buffer
    del v_h[-1]; del v_l[-1]

    return returns, adv_h, adv_l

class BatchProcessing:
    def __init__(self):
        self.counter = 0
        pass

    def collate_batch(self, buffer, device):
        """process buffer into batch tensors once buffer is full"""
        batch_states, batch_actions, batch_pi_hat = [],[],[]
        batch_options, batch_prev_options = [],[]
        batch_v_h, batch_v_l, batch_logp_h, batch_logp_l = [],[],[],[]
        batch_rtrn, batch_adv_h, batch_adv_l = [],[],[]
        batch_pi_bar, batch_betas = [],[]

        for data in buffer:
            #unpack episode data
            (states_mb, actions_mb, pi_hat_mb,
             options_mb, prev_options_mb,
             v_h_mb, v_l_mb, logp_h_mb, logp_l_mb,
             rtrn_mb, adv_h_mb, adv_l_mb,
             pi_bar_mb, betas_mb) = data

            # print(f"state: {states_mb[0].shape}\n"
            #       f"action: {actions_mb[0].shape}\n"
            #       f"option: {options_mb[0].shape}\n"
            #       f"pi_hat: {pi_hat_mb[0].shape}\n"
            #       f"prev_option: {prev_options_mb[0].shape}\n"
            #       f"rtrn: {rtrn_mb[0].shape}\n"
            #       f"adv_h: {adv_h_mb[0].shape}\n"
            #       f"adv_l: {adv_l_mb[0].shape}\n"
            #       f"logp_h: {logp_h_mb[4]}\n"
            #       f"logp_l: {logp_l_mb[4]}\n"
            #       f"v_h: {v_h_mb[0].shape}\n"
            #       f"v_l: {v_l_mb[0].shape}\n")

            batch_states.append(th.stack(states_mb).to(device))
            batch_actions.append(th.stack(actions_mb).to(device))
            batch_pi_hat.append(th.stack(pi_hat_mb).to(device))
            batch_options.append(th.stack(options_mb).to(device))
            batch_prev_options.append(th.stack(prev_options_mb).to(device))
            batch_v_h.append(th.stack(v_h_mb).to(device))
            batch_v_l.append(th.stack(v_l_mb).to(device))
            batch_logp_h.append(th.stack(logp_h_mb).to(device))
            batch_logp_l.append(th.stack(logp_l_mb).to(device))
            batch_rtrn.append(rtrn_mb)
            batch_adv_h.append(adv_h_mb)
            batch_adv_l.append(adv_l_mb)

            #extras
            batch_pi_bar.append(th.stack(pi_bar_mb).to(device))
            batch_betas.append(th.stack(betas_mb).to(device))

        # convert to tensors
        batch_states = th.cat(batch_states, dim=0).squeeze(1)
        batch_actions = th.cat(batch_actions, dim=0)
        batch_pi_hat = th.cat(batch_pi_hat, dim=0).squeeze(1)
        batch_options = th.cat(batch_options, dim=0)
        batch_prev_options = th.cat(batch_prev_options, dim=0)
        batch_v_h = th.cat(batch_v_h, dim=0).squeeze(1)
        batch_v_l = th.cat(batch_v_l, dim=0).squeeze(1)
        batch_logp_h = th.cat(batch_logp_h, dim=0)
        batch_logp_l = th.cat(batch_logp_l, dim=0)
        batch_rtrn = th.cat(batch_rtrn, dim=0).unsqueeze(-1)
        batch_adv_h = th.cat(batch_adv_h, dim=0).squeeze(1)
        batch_adv_l = th.cat(batch_adv_l, dim=0).squeeze(1)

        #extras
        batch_pi_bar = th.cat(batch_pi_bar, dim=0).squeeze(1)
        batch_betas = th.cat(batch_betas, dim=0).squeeze(1)

        # normalize advantages
        batch_adv_h = (batch_adv_h - batch_adv_h.mean()) / batch_adv_h.std()
        batch_adv_l = (batch_adv_l - batch_adv_l.mean()) / batch_adv_l.std()

        # print(f"state: {batch_states.shape}\n"
        #       f"action: {batch_actions.shape}\n"
        #       f"option: {batch_options.shape}\n"
        #       f"pi_hat: {batch_pi_hat.shape}\n"
        #       f"prev_option: {batch_prev_options.shape}\n"
        #       f"rtrn: {batch_rtrn.shape}\n"
        #       f"adv_h: {batch_adv_h.shape}\n"
        #       f"adv_l: {batch_adv_l.shape}\n"
        #       f"logp_h: {batch_logp_h.shape}\n"
        #       f"logp_l: {batch_logp_l.shape}\n"
        #       f"v_h: {batch_v_h.shape}\n"
        #       f"v_l: {batch_v_l.shape}\n")

        processed_buffer = (batch_states, batch_actions, batch_pi_hat,
                            batch_options, batch_prev_options,
                            batch_v_h, batch_v_l, batch_logp_h, batch_logp_l,
                            batch_rtrn, batch_adv_h, batch_adv_l,
                            batch_betas, batch_pi_bar)

        if self.counter % 90 == 0 and False:  # *10 episodes per buffer
            self.plot_buffer(processed_buffer)
        self.counter += 1

        return processed_buffer

    def plot_buffer(self, processed_buffer):
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
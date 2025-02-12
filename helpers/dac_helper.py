import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import sys

def layer_init(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)
    return layer
def pre_process(obs):
    state = th.FloatTensor(obs).unsqueeze(0)
    return state

def compute_pi_hat(prediction, prev_option,t=None):
    """computes high-policy (option selection) based on previous option, master policy & beta outputs.
        in the initial state, prev_option is set to None, function returns the master policy without
        beta contribution."""

    #get high actor option probabilities
    pi_W = prediction['pi_W']

    #Logic for initial state.
    if prev_option is None:
        return prediction['pi_W']

    #get option termination probabilities
    beta = prediction['betas']  # [batch_size, num_options]

    #create mask for the previous option(s)
    mask = th.zeros_like(pi_W)
    mask[th.arange(pi_W.size(0)), prev_option] = 1

    #compute pi_hat by factoring in beta contribution
    pi_hat = (1 - beta) * mask + beta * pi_W

    return pi_hat  #tensor.shape[batch_size, 4]

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

        # if self.counter % 1 == 0:
        #     self.plot_buffer(processed_buffer)
        # self.counter += 1

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
        axes[0, 2].set_xlabel("Termination Probability")
        axes[0, 2].set_ylabel("Buffer Index")
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
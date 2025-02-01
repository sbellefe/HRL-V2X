import torch as th


def pre_process(obs):
    state = th.FloatTensor(obs).unsqueeze(0)
    return state

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
        pass

    def collate_batch(self, buffer, device):
        """process buffer into batch tensors once buffer is full"""
        batch_states_h, batch_states_l, batch_states_beta = [],[],[]
        batch_options, batch_prev_options, batch_betas = [],[],[]
        batch_returns, batch_adv_h, batch_adv_l = [],[],[]
        batch_logp_h, batch_logp_l, batch_v_h, batch_v_l = [],[],[],[]

        for data in buffer:
            #unpack episode data
            (state_h, state_l, state_beta,
             option, prev_option, beta,
             rtrn, adv_h, adv_l,
             logp_h, logp_l, v_h, v_l) = data

            # print(f"state_h: {state_h[0].shape}\n"
            #       f"state_l: {state_l[0].shape}\n"
            #       f"state_beta: {state_beta[0].shape}\n"
            #       f"option: {option[0].shape}\n"
            #       f"prev_option: {prev_option[0].shape}\n"
            #       f"beta: {beta[0].shape}\n"
            #       f"rtrn: {rtrn[0].shape}\n"
            #       f"adv_h: {adv_h[0].shape}\n"
            #       f"logp_h: {logp_h[0].shape}\n"
            #       f"v_h: {v_h[0].shape}\n")


            batch_states_h.append(th.stack(state_h).to(device))
            batch_states_l.append(th.stack(state_l).to(device))
            batch_states_beta.append(th.stack(state_beta).to(device))
            batch_options.append(th.stack(option).to(device))
            batch_prev_options.append(th.stack(prev_option).to(device))
            batch_betas.append(th.stack(beta).to(device))
            batch_returns.append(rtrn)
            batch_adv_h.append(adv_h)
            batch_adv_l.append(adv_l)
            batch_logp_h.append(th.stack(logp_h).to(device))
            batch_logp_l.append(th.stack(logp_l).to(device))
            batch_v_h.append(th.stack(v_h).to(device))
            batch_v_l.append(th.stack(v_l).to(device))

        #convert to tensors
        batch_states_h = th.cat(batch_states_h, dim=0).squeeze(1)
        batch_states_l = th.cat(batch_states_l, dim=0).squeeze(1)
        batch_states_beta = th.cat(batch_states_beta, dim=0).squeeze(1)
        batch_options = th.cat(batch_options, dim=0)
        batch_prev_options = th.cat(batch_prev_options, dim=0)
        batch_betas = th.cat(batch_betas, dim=0).squeeze(1)
        batch_returns = th.cat(batch_returns, dim=0).unsqueeze(-1)
        batch_adv_h = th.cat(batch_adv_h, dim=0).squeeze(1)
        batch_adv_l = th.cat(batch_adv_l, dim=0).squeeze(1)
        batch_logp_h = th.cat(batch_logp_h, dim=0)
        batch_logp_l = th.cat(batch_logp_l, dim=0)
        batch_v_h = th.cat(batch_v_h, dim=0).squeeze(1)
        batch_v_l = th.cat(batch_v_l, dim=0).squeeze(1)

        # normalize advantages
        batch_adv_h = (batch_adv_h - batch_adv_h.mean()) / batch_adv_h.std()
        batch_adv_l = (batch_adv_l - batch_adv_l.mean()) / batch_adv_l.std()

        # print(f"state_h: {batch_states_h.shape}\n"
        #       f"state_l: {batch_states_l.shape}\n"
        #       f"state_beta: {batch_states_beta.shape}\n"
        #       f"option: {batch_options.shape}\n"
        #       f"prev_option: {batch_prev_options.shape}\n"
        #       f"beta: {batch_betas.shape}\n"
        #       f"rtrn: {batch_returns.shape}\n"
        #       f"adv_h: {batch_adv_h.shape}\n"
        #       f"logp_h: {batch_logp_h.shape}\n"
        #       f"v_h: {batch_v_h.shape}\n")

        return batch_states_h, batch_states_l, batch_states_beta, batch_options, batch_prev_options, batch_betas, batch_returns, batch_adv_h, batch_adv_l, batch_logp_h, batch_logp_l, batch_v_h, batch_v_l
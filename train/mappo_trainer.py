import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F

from Agent.mappo import MAPPO_Actor, MAPPO_Critic
from helpers.mappo_helper import RolloutBuffer_FO, RolloutBuffer_PO


class MAPPOtrainer:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        if not params.partial_observability:
            trainer = MAPPOtrainer_FO()
            train_returns, test_returns = trainer.train(trial, env, params)
        else:
            trainer = MAPPOtrainer_PO()
            train_returns, test_returns = trainer.train(trial, env, params)

        return train_returns, test_returns

class MAPPOtrainer_FO:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actor_shared = MAPPO_Actor(params).to(device)
        central_critic = MAPPO_Critic(params).to(device)
        actor_opt = th.optim.Adam(actor_shared.parameters(), lr=params.lr_actor)
        critic_opt = th.optim.Adam(central_critic.parameters(), lr=params.lr_critic)
        buffer = RolloutBuffer_FO(params)

        train_returns = []
        test_returns = []
        n_ep = 0
        N = params.num_agents
        agent_ids = th.eye(N, device=device).unsqueeze(1)  # shape [N, 1, N]

        for it in range(params.train_iterations):
            for ep in range(params.buffer_episodes):
                total_reward = 0

                """Get initial state from environment, convert to tensor"""
                global_state, _, _ = env.reset()
                global_state = th.tensor(global_state, dtype=th.float32, device=device)

                # loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    # loop for timesteps in control interval
                    for t in range(params.t_max):
                        """loop through each agent, collect actions from actor"""
                        actions = th.empty((N,), dtype=th.long, device=device)
                        logps = th.empty((N,), device=device)
                        for a in range(N):
                            x = th.cat([global_state, agent_ids[a]], dim=-1)  # shape [1, input_dim]
                            with th.no_grad():
                                logits = actor_shared(x)
                            action, logp = actor_shared.select_action(logits)

                            actions[a] = action
                            logps[a] = logp

                        """get value - only pass through critic once"""
                        with th.no_grad():
                            value = central_critic(global_state)

                        """step environment"""
                        integer_action = actions.tolist() # list of integer actions
                        global_next_state, _, _, global_reward, _, done = env.step(integer_action, k, t)

                        """Add transition to buffer"""
                        transition = (
                            global_state.squeeze(0),    #tensor shape [state_dim]
                            actions,                    #tensor shape [N]
                            global_reward,              #float
                            value.view(-1),             #tensor shape [1]
                            logps,                      #tensor shape [N]
                        )
                        buffer.push(*transition)

                        total_reward += global_reward
                        global_state = th.tensor(global_next_state, dtype=th.float32, device=device)

                        if done:
                            # print(f"****episode {n_ep} complete. Reward = {total_reward}****")
                            break

                n_ep += 1
                train_returns.append(total_reward)

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return = self.test(actor_shared, params, env)
                    test_returns.append(test_return)
                    print(f'Trial {trial+1}. Test return at episode {n_ep}: {test_return:.3f}. Average train return: {np.mean(train_returns[-params.test_interval:]):.3f}')

            # process buffer once full
            dataloader = buffer.process_batch()

            loss_a, loss_c = [], []
            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    states_mb, actions_mb, logps_mb, values_mb, rtrns_mb, adv_mb = batch

                    # print(f"states_mb.shape = {states_mb.shape} \n"
                    #       f"actions_mb.shape = {actions_mb.shape} \n"
                    #       f"logps_mb.shape = {logps_mb.shape} \n"
                    #       f"values_mb.shape = {values_mb.shape} \n"
                    #       f"rtrns_mb.shape = {rtrns_mb.shape} \n"
                    #       f"adv_mb.shape = {adv_mb.shape} \n")
                    """states_mb.shape = torch.Size([MB, 54]) 
                        actions_mb.shape = torch.Size([MB, 4])
                        logps_mb.shape = torch.Size([MB, 4])
                        values_mb.shape = torch.Size([MB])
                        rtrns_mb.shape = torch.Size([MB])
                        adv_mb.shape = torch.Size([MB])
                        """

                    MB, N = actions_mb.shape

                    #critic update
                    critic_opt.zero_grad()
                    values_new = central_critic(states_mb)
                    total_critic_loss = central_critic.critic_loss(values_new, values_mb, rtrns_mb)
                    total_critic_loss.backward()
                    critic_opt.step()

                    # actor update
                    actor_opt.zero_grad()
                    total_actor_loss = 0
                    for a in range(params.num_agents):
                        agent_id = agent_ids[a].repeat(MB, 1)
                        old_logp = logps_mb[:, a]
                        action = actions_mb[:, a]

                        x = th.cat([states_mb, agent_id], dim=-1)
                        logits = actor_shared(x)
                        dist = Categorical(logits=logits)
                        new_logp = dist.log_prob(action)
                        entropy = dist.entropy().mean()

                        actor_loss = actor_shared.actor_loss(new_logp, old_logp, adv_mb)
                        total_actor_loss += actor_loss - params.entropy_coef * entropy
                    total_actor_loss /= params.num_agents
                    total_actor_loss.backward()
                    actor_opt.step()

                    loss_a.append(total_actor_loss)
                    loss_c.append(total_critic_loss)

            # av_loss_a, av_loss_c = sum(loss_a) / len(loss_a), sum(loss_c) / len(loss_c)
            # print(f"Optimization avg losses: Policy loss: {av_loss_a:.3f} | Critic loss: {av_loss_c:.3f}")

        return train_returns, test_returns

    @staticmethod
    def test(actor, params, env):
        device = params.device
        env.testing_mode = True

        test_returns = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):

            total_rewards = 0

            global_state, _, _ = env.reset(test_idx=i)
            # global_state, _, _ = env.reset()

            for k in range(1, params.k_max + 1):
                if k > 1:
                    env.reset_control(k)

                for t in range(params.t_max):

                    actions = []  #No need to store these?
                    global_state = th.tensor(global_state, dtype=th.float32, device=device)

                    for a in range(params.num_agents):
                        agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device)

                        x = th.cat([global_state, agent_id], dim=-1)
                        logits = actor(x)
                        action, _ = actor.select_action(logits)

                        actions.append(action.item())

                    global_next_state, _, _, global_reward, _, done = env.step(actions, k, t)

                    global_state = global_next_state
                    total_rewards += global_reward

                    if done:
                        break

            test_returns[i] += total_rewards

        average_return = np.mean(test_returns)
        env.testing_mode = False
        return average_return


class MAPPOtrainer_PO:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actor_shared = MAPPO_Actor(params).to(device)
        central_critic = MAPPO_Critic(params).to(device)
        actor_opt = th.optim.Adam(actor_shared.parameters(), lr=params.lr_actor)
        critic_opt = th.optim.Adam(central_critic.parameters(), lr=params.lr_critic)
        buffer = RolloutBuffer_PO(params)

        train_returns = []
        test_returns = []
        n_ep = 0
        N = params.num_agents
        agent_ids = th.eye(N, device=device).unsqueeze(1)  # shape [N, 1, N]

        for it in range(params.train_iterations):
            for ep in range(params.buffer_episodes):
                """Initialize first timestep RNN hidden states to zero"""
                #shape list(len(num_agents). for each agent, shape = (timestep, batch episode, hidden dim)
                hidden_states_actor = th.zeros(N, 1, 1, params.actor_hidden_dim[0]).to(device)
                hidden_states_critic = th.zeros(N, 1, 1, params.critic_hidden_dim[0]).to(device)

                total_reward = 0

                #Get initial states/observations from environment
                _, local_states, fp_global_states = env.reset()
                local_states = th.stack([th.tensor(local_states[a], dtype=th.float32, device=device)
                                         for a in range(params.num_agents)], dim=0)  # ➞ [N, 1, obs_dim]
                fp_global_states = th.stack([th.tensor(fp_global_states[a], dtype=th.float32, device=device)
                                             for a in range(params.num_agents)], dim=0)  # ➞ [N, 1, state_dim]

                #loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    #loop for timesteps in control interval
                    for t in range(params.t_max):
                        """loop through each agent, collect actions from actor"""
                        actions = th.empty((N,), dtype=th.long, device=device)
                        logps = th.empty((N,), device=device)
                        for a in range(N):
                            x = th.cat([local_states[a], agent_ids[a]], dim=-1).unsqueeze(0)   #shape [1, 1, input_dim]
                            with th.no_grad():
                                logits, h_a = actor_shared(x, hidden_states_actor[a])
                            hidden_states_actor[a] = h_a    #Overwrite previous hidden state

                            #next get action, logp  for agent
                            action, logp = actor_shared.select_action(logits)

                            actions[a] = action
                            logps[a] = logp


                        """loop through each agent, collect values from critic"""
                        values = th.empty((N,), device=device)
                        for a in range(N):
                            x = th.cat([fp_global_states[a], agent_ids[a]], dim=-1).unsqueeze(0)  # shape [1, 1, input_dim]
                            with th.no_grad():
                                value, h_c = central_critic(x, hidden_states_critic[a])
                            hidden_states_critic[a] = h_c  # Overwrite previous hidden state

                            values[a] = value

                        """step environment"""
                        integer_action = actions.tolist()  # list of integer actions
                        _, local_next_states, fp_global_next_states, global_reward, _, done = env.step(integer_action, k, t)

                        """store transition in buffer"""
                        # print(f"logps.transition = {logps}")
                        transition = (
                            local_states.squeeze(1),        #tensor [N, obs_dim]
                            fp_global_states.squeeze(1),    #tensor [N, state_dim]
                            actions,                        #tensor [N]
                            global_reward,                  # float
                            values,                         # tensor [N]
                            logps,                          # tensor [N]
                        )
                        buffer.push(*transition)

                        total_reward += global_reward
                        local_states = th.stack([th.tensor(local_next_states[a], dtype=th.float32, device=device)
                                                 for a in range(N)], dim=0)  # ➞ [N, 1, obs_dim]
                        fp_global_states = th.stack([th.tensor(fp_global_next_states[a], dtype=th.float32, device=device)
                                                     for a in range(N)], dim=0)  # ➞ [N, 1, state_dim]

                        if done:
                            # print(f"****episode {n_ep} complete. Reward = {total_reward}****")
                            break

                n_ep += 1
                train_returns.append(total_reward)

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return = self.test(actor_shared, params, env)
                    test_returns.append(test_return)
                    print(f'Trial {trial+1}. Test return at episode {n_ep}: {test_return:.3f}. Average train return: {np.mean(train_returns[-params.test_interval:]):.3f}')

            dataloader = buffer.process_batch()

            for epoch in range(params.opt_epochs):
                for minibatch in dataloader:
                    # unpack minibatch sequences
                    obs_mb, fpgs_mb, actions_mb, values_mb, logps_mb, rtrns_mb, adv_mb = minibatch

                    # print(f"obs_mb.shape = {obs_mb.shape} \n"
                    #       f"fpgs_mb.shape = {fpgs_mb.shape} \n"
                    #       f"actions_mb.shape = {actions_mb.shape} \n"
                    #       f"logps_mb.shape = {logps_mb.shape} \n"
                    #       f"values_mb.shape = {values_mb.shape} \n"
                    #       f"rtrns_mb.shape = {rtrns_mb.shape} \n"
                    #       f"adv_mb.shape = {adv_mb.shape} \n")
                    """ obs_mb.shape = torch.Size([256, 10, 4, 24])
                        fpgs_mb.shape = torch.Size([256, 10, 4, 54])
                        actions_mb.shape = torch.Size([256, 10, 4])
                        logps_mb.shape = torch.Size([256, 10, 4])
                        values_mb.shape = torch.Size([256, 10, 4])
                        prev_action_mb.shape = torch.Size([256, 10, 4, 13])
                        rtrns_mb.shape = torch.Size([256, 10])
                        adv_mb.shape = torch.Size([256, 10, 4])
                         """
                    # sys.exit()

                    MB, T, N, _  = obs_mb.shape
                    #initialize first hidden states for each sequence (full episode) to zero
                    hidden_states_actor = th.zeros((N, 1, MB, params.actor_hidden_dim[0]), device=device)
                    hidden_states_critic = th.zeros((N, 1, MB, params.actor_hidden_dim[0]), device=device)

                    # critic update
                    critic_opt.zero_grad()
                    total_critic_loss = 0
                    for a in range(N):
                        agent_id = agent_ids[a].repeat(MB, T, 1)

                        x = th.cat([fpgs_mb[:,:,a,:], agent_id], dim=-1)
                        values_new, _ = central_critic(x, hidden_states_critic[a])
                        values_new = values_new.squeeze(-1)

                        critic_loss = central_critic.critic_loss(values_new, values_mb[:,:, a], rtrns_mb)
                        total_critic_loss += critic_loss
                    total_critic_loss /= N
                    total_critic_loss.backward()
                    critic_opt.step()

                    # actor update
                    actor_opt.zero_grad()
                    total_actor_loss = 0
                    for a in range(N):
                        agent_id = agent_ids[a].repeat(MB,T, 1)

                        obs = obs_mb[:,:, a, :]
                        actions = actions_mb[:,:, a]
                        logps_old = logps_mb[:,:, a]
                        adv = adv_mb[:,:, a]

                        x = th.cat([obs, agent_id], dim=-1)
                        logits, _ = actor_shared(x, hidden_states_actor[a])
                        dist = Categorical(logits=logits)
                        logps_new = dist.log_prob(actions)
                        entropy = dist.entropy().mean()

                        actor_loss = actor_shared.actor_loss(logps_new, logps_old, adv)
                        total_actor_loss += actor_loss - params.entropy_coef * entropy

                    total_actor_loss /= N
                    total_actor_loss.backward()
                    actor_opt.step()

        return train_returns, test_returns

    @staticmethod
    def test(actor, params, env):
        device = params.device
        env.testing_mode = True
        # if params.partial_observability: actor.gru.flatten_parameters()

        test_returns = np.zeros(params.test_episodes)
        N = params.num_agents
        agent_ids = th.eye(N, device=device).unsqueeze(1)  # shape [N, 1, N]

        for i in range(params.test_episodes):
            hidden_states_actor = th.zeros(N, 1, 1, params.actor_hidden_dim[0]).to(device)
            total_rewards = 0

            _, local_states, _ = env.reset(test_idx=i)
            # _, local_states, _ = env.reset()

            for k in range(1, params.k_max + 1):

                if k > 1:
                    env.reset_control(k)

                for t in range(params.t_max):
                    local_states = th.stack([th.tensor(local_states[a], dtype=th.float32, device=device)
                                             for a in range(N)], dim=0)  # ➞ [N, obs_dim]

                    actions = th.empty((N,), dtype=th.long, device=device)
                    for a in range(N):
                        x = th.cat([local_states[a], agent_ids[a]], dim=-1).unsqueeze(0)
                        with th.no_grad():
                            logits, h_a = actor(x, hidden_states_actor[a])
                        hidden_states_actor[a] = h_a
                        action, _ = actor.select_action(logits)

                        actions[a] = action

                    _, local_next_states, _, global_reward, _, done = env.step(actions.tolist(), k, t)

                    local_states = local_next_states
                    total_rewards += global_reward

                    if done:
                        break

            test_returns[i] += total_rewards

        average_return = np.mean(test_returns)
        env.testing_mode = False
        return average_return

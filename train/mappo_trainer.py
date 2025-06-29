import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F

from Agent.mappo import MAPPO_Actor, MAPPO_Critic
from helpers.mappo_helper import RolloutBuffer_FO, RolloutBuffer_PO, NEWRolloutBuffer_PO


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

    def xtrain(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actor_shared = MAPPO_Actor(params).to(device)
        central_critic = MAPPO_Critic(params).to(device)
        actor_opt = th.optim.Adam(actor_shared.parameters(), lr=params.lr_actor)
        critic_opt = th.optim.Adam(central_critic.parameters(), lr=params.lr_critic)
        # buffer = RolloutBuffer_PO(params)
        buffer = NEWRolloutBuffer_PO(params) #maybe not broken
        # buffer1 = NEWRolloutBuffer_PO(params) #broken

        train_returns = []
        test_returns = []
        n_ep = 0
        N = params.num_agents
        agent_ids = th.eye(N, device=device).unsqueeze(1)  # shape [N, 1, N]

        for it in range(params.train_iterations):
            for ep in range(params.buffer_episodes):
                """Initialize first timestep RNN hidden states to zero"""
                #shape list(len(num_agents). for each agent, shape = (timestep, batch episode, hidden dim)
                hidden_states_actor = [th.zeros(1, 1, params.actor_hidden_dim[0]).to(device) for _ in range(N)]
                hidden_states_critic = [th.zeros(1, 1, params.critic_hidden_dim[0]).to(device) for _ in range(N)]
                # hidden_states_actor = th.zeros((N, 1, 1, params.actor_hidden_dim[0]), device=device)
                # hidden_states_critic = th.zeros((N, 1, 1, params.actor_hidden_dim[0]), device=device)

                # prev_actions = [th.zeros(1, params.action_dim).to(device) for _ in range(params.num_agents)]
                prev_actions = th.zeros(N, params.action_dim).to(device)

                total_reward = 0

                #Get initial states/observations from environment
                _, local_states, fp_global_states = env.reset()
                local_states = th.stack([th.tensor(local_states[a], dtype=th.float32, device=device)
                                         for a in range(params.num_agents)], dim=0)  # ➞ [N, 1, obs_dim]
                fp_global_states = th.stack([th.tensor(fp_global_states[a], dtype=th.float32, device=device)
                                             for a in range(params.num_agents)], dim=0)  # ➞ [N, 1, state_dim]
                # print(f"obs: {local_states.shape}")
                # print(f"fpgs: {fp_global_states.shape}")


                #loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    #loop for timesteps in control interval
                    for t in range(params.t_max):
                        """loop through each agent, collect actions from actor"""
                        # actions, logps = [], []
                        actions = th.empty((N,), dtype=th.long, device=device)
                        logps = th.empty((N,), device=device)
                        for a in range(N):
                            # print(f"\n t= {t}. agent {a}. Local_states = \n {local_states[a]}")
                            prev_action = prev_actions[a].unsqueeze(0)  #[1, action_dim]
                            # x = th.cat([local_states[a], agent_ids[a], prev_action], dim=-1).unsqueeze(0)   #shape [1, 1, input_dim]
                            x = th.cat([local_states[a], agent_ids[a]], dim=-1).unsqueeze(0)   #shape [1, 1, input_dim]
                            with th.no_grad():
                                logits, h_a = actor_shared(x, hidden_states_actor[a])
                            hidden_states_actor[a] = h_a    #Overwrite previous hidden state

                            #next get action, logp  for agent
                            action, logp = actor_shared.select_action(logits)

                            actions[a] = action
                            logps[a] = logp

                            # actions.append(action.view(-1))  #shape [1]
                            # logps.append(logp.view(-1))      #shape [1]

                            #prepare agent prev_action for next timestep by converting action to one-hot
                            # prev_actions[a] = F.one_hot(action.view(-1), num_classes=params.action_dim).float().to(device)  #[1, action_dim]
                            prev_actions[a] = F.one_hot(action.view(-1), num_classes=params.action_dim).float()  #[1, action_dim]

                        """loop through each agent, collect values from critic"""
                        # values = []
                        values = th.empty((N,), device=device)
                        prev_joint_action = prev_actions.view(1, -1)  # shape [1, N * action_dim]
                        # print(f"prev_actions.shape: {prev_actions.shape}")
                        # print(f"prev_joint_action.shape: {prev_joint_action.shape}")
                        for a in range(N):
                            # x = th.cat([fp_global_states[a], agent_ids[a], prev_joint_action], dim=-1).unsqueeze(0)  # shape [1, 1, input_dim]
                            x = th.cat([fp_global_states[a], agent_ids[a]], dim=-1).unsqueeze(0)  # shape [1, 1, input_dim]
                            with th.no_grad():
                                value, h_c = central_critic(x, hidden_states_critic[a])
                            hidden_states_critic[a] = h_c  # Overwrite previous hidden state

                            values[a] = value.item()
                            # values.append(value.view(-1))

                        """step environment"""
                        integer_action = actions.tolist()  # list of integer actions
                        # integer_action = [int(a.item()) for a in th.stack(actions, dim=0)]  # list of integer actions
                        _, local_next_states, fp_global_next_states, global_reward, _, done = env.step(integer_action, k, t)

                        """store transition in buffer"""
                        # print(f"logps.transition = {logps}")
                        transition = (
                            local_states.squeeze(1),        #tensor [N, obs_dim]
                            fp_global_states.squeeze(1),    #tensor [N, state_dim]
                            actions,                        #tensor [N]
                            global_reward,  # float
                            logps,  # tensor [N]
                            values,  # tensor [N]
                            prev_actions,                   #tensor [N, action_dim]
                        )
                        buffer.push(*transition)
                        # buffer1.push(*transition)
                        # transition = (
                        #     local_states.squeeze(1),        #tensor [N, obs_dim]
                        #     fp_global_states.squeeze(1),    #tensor [N, state_dim]
                        #     th.cat(actions, dim=0),                        #tensor [N]
                        #     prev_actions,                   #tensor [N, action_dim]
                        #     global_reward,                  #float
                        #     th.cat(values, dim=0),                         #tensor [N]
                        #     th.cat(logps, dim=0),                          #tensor [N]
                        # )
                        # print(f"\nlocal_states: {local_states.shape}")
                        # print(f"fpgs: {fp_global_states.shape}")
                        # print(f"actions: {actions}")
                        # print(f"prev_actions: {prev_actions}")
                        # print(f"values: {values}")
                        # print(f"logps: {logps}")
                        # print(f"integer_action: {integer_action}")

                        # transition = (
                        #     local_states,   #tensor [N, obs_dim]
                        #     fp_global_states,   #tensor [N, state_dim]
                        #     th.cat(actions, dim=0),  #tensor [N]
                        #     global_reward,  # float
                        #     th.cat(values, dim=0),  # shape [N]
                        #     th.cat(logps, dim=0),  # shape[N]
                        #     prev_joint_action.view(-1),   #tensor [N * action_dim]
                        # )



                        total_reward += global_reward
                        local_states = th.stack([th.tensor(local_next_states[a], dtype=th.float32, device=device)
                                                 for a in range(N)], dim=0)  # ➞ [N, 1, obs_dim]
                        fp_global_states = th.stack([th.tensor(fp_global_next_states[a], dtype=th.float32, device=device)
                                                     for a in range(N)], dim=0)  # ➞ [N, 1, state_dim]

                        if done:
                            break

                buffer.process_episode()
                # buffer1.process_episode()
                n_ep += 1
                train_returns.append(total_reward)

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return = self.test(actor_shared, params, env)
                    test_returns.append(test_return)
                    print(f'Trial {trial+1}. Test return at episode {n_ep}: {test_return:.3f}. Average train return: {np.mean(train_returns[-params.test_interval:]):.3f}')

            # process buffer once full
            # print("— OLD BUFFER DUMP —")
            dataloader = buffer.process_batch()
            # print("— NEW BUFFER DUMP —")
            # dataloader1 = buffer1.process_batch()
            # Easy-to-change indices:
            ep_idx = 0  # which episode to spot-check
            ag = 0  # which agent to spot-check

            debuggg = False
            if debuggg:
                # Fetch a single batch from each DataLoader
                batch_old = next(iter(dataloader))
                # batch_new = next(iter(dataloader1))

                labels = [
                    'obs', 'fpgs', 'actions', 'logps',
                    'values', 'prev_actions', 'returns', 'advantages'
                ]

                # 1) Print overall shape & dtype for each tensor
                print("— OLD BUFFER BATCH SHAPES & DTYPES —")
                for name, tensor in zip(labels, batch_old):
                    print(f"{name:12s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

                print("\n— NEW BUFFER BATCH SHAPES & DTYPES —")
                for name, tensor in zip(labels, batch_new):
                    print(f"{name:12s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

                # 2) Print a spot-check for the chosen episode & agent
                print(f"\n— SAMPLE EP={ep_idx}, AG={ag}")
                # print("OLD obs:          ", batch_old[0][ep_idx, :, ag, :])
                # print("NEW obs:          ", batch_new[0][ep_idx, :, ag, :])
                # print("OLD fpgs:         ", batch_old[1][ep_idx, :, ag, :])
                # print("NEW fpgs:         ", batch_new[1][ep_idx, :, ag, :])
                print("OLD actions:      ", batch_old[2][ep_idx, :, ag])
                print("NEW actions:      ", batch_new[2][ep_idx, :, ag])
                print("OLD logps:        ", batch_old[3][ep_idx, :, ag])
                print("NEW logps:        ", batch_new[3][ep_idx, :, ag])
                print("OLD values:       ", batch_old[4][ep_idx, :, ag])
                print("NEW values:       ", batch_new[4][ep_idx, :, ag])
                print("OLD prev_actions: ", batch_old[5][ep_idx, :, ag, :])
                print("NEW prev_actions: ", batch_new[5][ep_idx, :, ag, :])
                print("OLD returns:      ", batch_old[6][ep_idx, :])
                print("NEW returns:      ", batch_new[6][ep_idx, :])
                print("OLD advantages:   ", batch_old[7][ep_idx, :, ag])
                print("NEW advantages:   ", batch_new[7][ep_idx, :, ag])




            for epoch in range(params.opt_epochs):
                for minibatch in dataloader:
                    # unpack minibatch sequences
                    obs_mb, fpgs_mb, actions_mb, logps_mb, values_mb, prev_action_mb, rtrns_mb, adv_mb = minibatch

                    # print(f"obs_mb.shape = {obs_mb.shape} \n"
                    #       f"fpgs_mb.shape = {fpgs_mb.shape} \n"
                    #       f"actions_mb.shape = {actions_mb.shape} \n"
                    #       f"logps_mb.shape = {logps_mb.shape} \n"
                    #       f"values_mb.shape = {values_mb.shape} \n"
                    #       f"prev_action_mb.shape = {prev_action_mb.shape} \n"
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

                    MB, T, N, action_dim  = prev_action_mb.shape
                    #initialize first hidden states for each sequence (full episode) to zero
                    hidden_states_actor = [th.zeros(1, MB, params.actor_hidden_dim[0]).to(device) for _ in range(N)]
                    # hidden_states_actor = th.zeros((N, 1, MB, params.actor_hidden_dim[0]), device=device)
                    hidden_states_critic = [th.zeros(1, MB, params.critic_hidden_dim[0]).to(device) for _ in range(N)]
                    # hidden_states_critic = th.zeros((N, 1, MB, params.actor_hidden_dim[0]), device=device)

                    #Reshape prev action one-hot to create joint action one-hot
                    prev_joint_actions_mb = prev_action_mb.view(MB, T, N * action_dim)

                    # if epoch == 4:
                    #     print(f'\n obs_mb:\n {obs_mb[0, 4]}')
                    #     print(f'\n fpgs_mb:\n {fpgs_mb[0, 4]}')
                    #     print(f'\n actions_mb:\n {actions_mb[0, 4]}')
                    #     print(f'\n logps_mb:\n {logps_mb[0, 4]}')
                    #     print(f'\n values_mb:\n {values_mb[0, 4]}')
                    #     print(f'\n prev_action_mb:\n {prev_action_mb[0, 4]}')
                    #     print(f'\n rtrns_mb:\n {rtrns_mb[0, 4]}')
                    #     print(f'\n adv_mb:\n {adv_mb[0], 4}')
                    #     sys.exit()


                    # critic update
                    critic_opt.zero_grad()
                    total_critic_loss = 0
                    for a in range(N):
                        agent_id = agent_ids[a].repeat(MB, T, 1)
                        # print(f"\nprev_joint_actions_mb.shape: {prev_joint_actions_mb.shape}")
                        # print(f"agent_id.shape: {agent_id[0,0,:]}")
                        # print(f"fpgs_mb[:,:,a,:].shape: {fpgs_mb[:,:,a,:].dtype}")

                        # x = th.cat([fpgs_mb[:,:,a,:], agent_id, prev_joint_actions_mb], dim=-1)
                        x = th.cat([fpgs_mb[:,:,a,:], agent_id], dim=-1)
                        values_new, _ = central_critic(x, hidden_states_critic[a])
                        values_new = values_new.squeeze(-1)

                        # print(f"values_mb.shape: {values_mb[:,:, a].dtype}")
                        # print(f"values_new.shape: {values_new.dtype}")
                        # print(f"rtrns_mb.shape: {rtrns_mb.dtype}")


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
                        prev_actions = prev_action_mb[:,:, a, :]
                        logps_old = logps_mb[:,:, a]
                        adv = adv_mb[:,:, a]

                        # print(f"\nobs.shape: {obs.dtype}")
                        # print(f"actions.shape: {actions.dtype}")
                        # print(f"prev_actions.shape: {prev_actions.dtype}")
                        # print(f"logps_old.shape: {logps_old.dtype}")
                        # print(f"adv.shape: {adv.dtype}")



                        # x = th.cat([obs, agent_id, prev_actions], dim=-1)
                        x = th.cat([obs, agent_id], dim=-1)
                        logits, _ = actor_shared(x, hidden_states_actor[a])
                        dist = Categorical(logits=logits)
                        logps_new = dist.log_prob(actions)
                        entropy = dist.entropy().mean()

                        # print(f"logps_new.shape: {logps_new.dtype}")


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

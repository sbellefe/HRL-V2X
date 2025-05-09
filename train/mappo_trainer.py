import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F

from Agent.mappo import MAPPO_Actor, MAPPO_Critic
from helpers.mappo_helper import RolloutBuffer


class MAPPOtrainer:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actor_shared = MAPPO_Actor(params).to(device)
        central_critic = MAPPO_Critic(params).to(device)
        actor_opt = th.optim.Adam(actor_shared.parameters(), lr=params.lr_actor)
        critic_opt = th.optim.Adam(central_critic.parameters(), lr=params.lr_critic)
        buffer = RolloutBuffer(params)

        train_rewards = []
        test_rewards = []
        n_ep = 0

        for it in range(params.train_iterations):
            for ep in range(params.buffer_episodes):

                if params.partial_observability:    #Initialize first timestep RNN hidden states to zero
                    #shape list(len(num_agents). for each agent, shape = (timestep, batch episode, hidden dim)
                    hidden_states_actor = [th.zeros(1, 1, params.actor_hidden_dim[0]).to(device) for _ in range(params.num_agents)]
                    hidden_states_critic = [th.zeros(1, 1, params.critic_hidden_dim[0]).to(device) for _ in range(params.num_agents)]
                    prev_actions = [th.zeros(1, params.action_dim).to(device) for _ in range(params.num_agents)]
                else:
                    hidden_states_actor, hidden_states_critic, prev_actions = None, None, None

                # TODO: Initialize agents_id once here instead of several times later

                total_reward = 0

                #Get initial states/observations from environment
                global_state, local_states, fp_global_states = env.reset()


                # print(f"global state: {global_state.shape}\n\n")
                      # f"local states: {local_states}")

                #loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    #loop for timesteps in control interval
                    for t in range(params.t_max):
                        # convert states to tensors
                        global_state = th.tensor(global_state, dtype=th.float32, device=device)
                        if local_states is not None:  # POSIG ONLY
                            local_states = {agent_id: th.tensor(state, dtype=th.float32, device=device)
                                            for agent_id, state in local_states.items()}
                            fp_global_states = {agent_id: th.tensor(state, dtype=th.float32, device=device)
                                                for agent_id, state in fp_global_states.items()}

                        """loop through each agent, collect actions from actor"""
                        actions, logps = [], []

                        for a in range(params.num_agents):
                            agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device)  #shape [1, num_agents]

                            #first get logits from actor
                            if params.partial_observability:
                                # print(f"ACTOR: obs: {local_states[a].shape} | prev a: {prev_actions[a].shape} | id = {agent_id.shape}")
                                x = th.cat([local_states[a], agent_id, prev_actions[a]], dim=-1).unsqueeze(0)   #shape [1, 1, input_dim]
                                with th.no_grad():
                                    logits, h_a = actor_shared(x, hidden_states_actor[a])
                                hidden_states_actor[a] = h_a    #Overwrite previous hidden state
                            else:
                                x = th.cat([global_state, agent_id], dim=-1)        #shape [1, input_dim]
                                with th.no_grad():
                                    logits = actor_shared(x)

                            #next get action, logp  for agent
                            action, logp = actor_shared.select_action(logits)
                            action = action.squeeze(-1)
                            logp = logp.squeeze(-1)
                            # print(f'agent {a}, action: {action}\n'
                            #       f'action.item(): {action.item()}\n'
                            #       f'logp.shape: {logp.shape}\n')
                            actions.append(action)  #shape [1, 1]
                            logps.append(logp)      #shape [1, 1]

                            #for POSIG only: prepare agent prev_action for next timestep by converting action to one-hot
                            if params.partial_observability:
                                prev_actions[a] = F.one_hot(action, num_classes=params.action_dim).float().to(device)  #[1, action_dim]
                                # prev_actions[a] = F.one_hot(action.squeeze(-1), num_classes=params.action_dim).float().to(device)   #[1, action_dim]
                                # prev_actions[a] = F.one_hot(th.tensor([action.item()]), num_classes=params.action_dim).float().to(device)




                        """loop through each agent, collect values from critic"""
                        if params.partial_observability:
                            prev_joint_action = th.cat(prev_actions, dim=-1) # shape [1, action_dim * num_agents]
                            # print(prev_joint_action.shape, len(prev_actions))
                            # sys.exit()
                            values = []
                            for a in range(params.num_agents):
                                agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device) #shape [1, num_agents]
                                # print(f"CRITIC: fpgs: {fp_global_states[a].shape} | prev j a: {prev_joint_action.shape} | id = {agent_id.shape}")

                                x = th.cat([fp_global_states[a], prev_joint_action, agent_id], dim=-1).unsqueeze(0) #shape [1, 1, input_dim]
                                with th.no_grad():
                                    value, h_c = central_critic(x, hidden_states_critic[a])
                                hidden_states_critic[a] = h_c  # Overwrite previous hidden state
                                values.append(value)

                            # values = th.stack(values).squeeze(-1)   # shape [N, 1]

                        else:       #For non-POSIG, we only pass through critic once
                            x = global_state
                            with th.no_grad():
                                values = central_critic(x)  #shape [1, 1]
                            # values = values.squeeze(-1) #shape [1]
                            # values = value.expand(params.num_agents, 1).squeeze() # expand value to all agents. shape [N,]


                        """step environment"""
                        integer_action = th.stack(actions).flatten().int().tolist()  # list of integer actions
                        global_next_state, local_next_states, fp_global_next_states, global_reward, individual_rewards, done = env.step(integer_action, k, t)


                        """store transition in buffer"""
                        if params.partial_observability:
                            transition = {
                                "observations": local_states,   #dict of tensors [1, obs_dim]
                                "fp_global_states": fp_global_states,   #dict of tensors  [1, state_dim]
                                "joint_action": th.stack(actions),  # shape[N]
                                "prev_actions": prev_actions,   #list of N one-hot tensors shape [1, action_dim]
                                "global_reward": global_reward, #float
                                "logps": th.stack(logps).squeeze(),   # shape[N]
                                "values": th.stack(values).squeeze()   # shape [N]
                            }
                        else:
                            transition = {
                                "global_state": global_state, #tensor shape [1, state_dim]
                                "joint_action": th.stack(actions).squeeze(),  # shape[N]
                                "global_reward": global_reward,     #float
                                "logps": th.stack(logps).squeeze(),   # shape[N]
                                "values": values.squeeze(-1), #shape [1]
                            }

                        buffer.push(**transition)

                        total_reward += global_reward
                        global_state = global_next_state
                        local_states = local_next_states
                        fp_global_states = fp_global_next_states

                        # print(f"global state: {global_state.shape}\n"
                        #       f"local_states: {type(local_states)}\n"
                        #       f"actions: {len(actions)}\n"
                        #       f"prev_actions: {prev_actions}\n"
                        #       f"individual_rewards: {individual_rewards.shape}\n"
                        #       f"logps: {th.stack(logps, dim=0).squeeze()}\n"
                        #       f"values: {values.shape}\n"
                        #       )
                        if done:
                            #get final value
                            if params.partial_observability:
                                prev_joint_action = th.cat(prev_actions, dim=-1)  # shape [1, action_dim * num_agents]
                                fp_global_states = {agent_id: th.tensor(state, dtype=th.float32, device=device)
                                                    for agent_id, state in fp_global_states.items()}
                                values = []
                                for a in range(params.num_agents):
                                    agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device)
                                    x = th.cat([fp_global_states[a], prev_joint_action, agent_id], dim=-1).unsqueeze(0)
                                    with th.no_grad():
                                        value, _ = central_critic(x, hidden_states_critic[a])
                                    values.append(value)
                                final_values = th.stack(values).squeeze()  # shape [N]
                                # final_values = th.stack(values).squeeze(-1)  # shape [N, 1]
                            else:
                                global_state = th.tensor(global_state, dtype=th.float32, device=device)
                                with th.no_grad():
                                    final_values = central_critic(global_state)
                                final_values = final_values.squeeze(-1) #[1]
                                # final_values = final_value.expand(params.num_agents, 1).squeeze()
                            buffer.process_episode(final_values)

                            # t0 = env.current_veh_positions['time'].unique()[0]
                            # print(f"Training Episode {n_ep+1} complete. t0 = {t0}. R = {total_reward:.2f}")
                            break


                n_ep += 1
                train_rewards.append(total_reward)


                p = 100
                if n_ep % p == 0:
                    print(f"n_ep={n_ep}. Training Return Running average over last {p} eps: {np.mean(train_rewards[-p:]):.3f}")

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    # show_testing = False if n_ep < params.render_delay and params.show_testing else True
                    test_reward = self.test(deepcopy(actor_shared), params, env)
                    test_rewards.append(test_reward)

                    # plt.figure(1)
                    # plt.clf()
                    # plt.plot(test_rewards, label="Test Reward")
                    # plt.xlabel("Test Interval")
                    # plt.ylabel("Return")
                    # plt.title("Test Return Over Time MAPPO")
                    # plt.legend()
                    # plt.grid(True)
                    # plt.draw()
                    # plt.pause(1)

                    print(f'Trial {trial}. Test return at episode {n_ep}: {test_reward:.3f}. Average train return: {np.mean(train_rewards[-params.test_interval:]):.3f}')

            # process buffer once full
            dataloader = buffer.process_batch()

            #initiate learning
            if params.partial_observability:
                self.learn_gru(actor_shared, central_critic, actor_opt, critic_opt, dataloader, params)
            else:
                self.learn_ffn(actor_shared, central_critic, actor_opt, critic_opt, dataloader, params)

        print(f"Trial {trial} Complete. Max test reward {max(test_rewards):.3f}.")
        return train_rewards, test_rewards

    def learn_gru(self, actor, critic, actor_opt, critic_opt, dataloader, params):
        device = params.device
        # print(f"Dataloader {len(dataloader)}.")
              # f"batch[0].shape = {batch[0].shape} \n"
              # f"batch.len = {len(batch)} \n")

        for epoch in range(params.opt_epochs):
            for minibatch in dataloader:

                # unpack minibatch sequences
                actions_mb, obs_mb, fpgs_mb, prev_actions_mb, logps_mb, values_mb, rtrns_mb, adv_mb = minibatch

                # print(f"actions_mb.shape = {actions_mb.shape} \n"
                #       f"obs_mb.shape = {obs_mb.shape} \n"
                #       f"fpgs_mb.shape = {fpgs_mb.shape} \n"
                #       f"prev_actions_mb.shape = {prev_actions_mb.shape} \n"
                #       f"logps_mb.shape = {logps_mb.shape} \n"
                #       f"values_mb.shape = {values_mb.shape} \n"
                #       f"rtrns_mb.shape = {rtrns_mb.shape} \n"
                #       f"adv_mb.shape = {adv_mb.shape} \n")
                """
                actions_mb.shape = torch.Size([16, 10, 4]) 
                obs_mb.shape = torch.Size([16, 10, 4, 24])
                fpgs_mb.shape = torch.Size([16, 10, 4, 54])
                prev_actions_mb.shape = torch.Size([16, 10, 4, 13])
                logps_mb.shape = torch.Size([16, 10, 4])
                values_mb.shape = torch.Size([16, 10, 4])
                rtrns_mb.shape = torch.Size([16, 10])
                adv_mb.shape = torch.Size([16, 10, 4])               """


                MB, T, N, action_dim  = prev_actions_mb.shape
                #initialize first hidden state TODO why is MB 2nd index when we have batch_first=True?
                hidden_states_actor = [th.zeros(1, MB, params.actor_hidden_dim[0]).to(device) for _ in range(params.num_agents)]
                hidden_states_critic = [th.zeros(1, MB, params.critic_hidden_dim[0]).to(device) for _ in range(params.num_agents)]

                #Reshape prev action one-hot to create joint action one-hot
                prev_joint_actions_mb = prev_actions_mb.view(MB, T, N * action_dim)

                # critic update
                critic_opt.zero_grad()
                total_critic_loss = 0
                for a in range(params.num_agents):
                    agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().repeat(MB,T, 1).to(device)
                    x = th.cat([fpgs_mb[:,:,a,:], prev_joint_actions_mb, agent_id], dim=-1)
                    values_new, _ = critic(x, hidden_states_critic[a])
                    values_new = values_new.squeeze(-1)

                    critic_loss = critic.critic_loss(values_new, values_mb[:,:, a], rtrns_mb, params.eps_clip)
                    total_critic_loss += critic_loss
                total_critic_loss /= params.num_agents
                total_critic_loss.backward()
                critic_opt.step()

                # actor update
                actor_opt.zero_grad()
                total_actor_loss = 0
                for a in range(params.num_agents):
                    agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().repeat(MB, T, 1).to(device)

                    obs = obs_mb[:,:, a, :]
                    actions = actions_mb[:,:, a]
                    prev_actions = prev_actions_mb[:,:, a, :]
                    logps_old = logps_mb[:,:, a]
                    adv = adv_mb[:,:, a]

                    x = th.cat([obs, agent_id, prev_actions], dim=-1)
                    logits, _ = actor(x, hidden_states_actor[a])
                    dist = Categorical(logits=logits)
                    logps_new = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    actor_loss = actor.actor_loss(logps_new, logps_old, adv)
                    total_actor_loss += actor_loss - params.entropy_coef * entropy

                total_actor_loss /= params.num_agents
                total_actor_loss.backward()
                actor_opt.step()


    def learn_ffn(self, actor, critic, actor_opt, critic_opt, dataloader, params):
        #TODO: This code is 100% broken at the moment
        for _ in range(params.opt_epochs):
            for batch in dataloader:
                state_mb, action_mb, logp_mb, value_mb, rtrn_mb, adv_mb = batch

                # critic update
                critic_opt.zero_grad()
                total_critic_loss = 0
                value_new = central_critic(state_mb)
                total_critic_loss = central_critic.critic_loss(value_new, value_mb, rtrn_mb, params.eps_clip)
                total_critic_loss.backward()
                critic_opt.step()

                # actor update
                actor_opt.zero_grad()
                total_actor_loss = 0
                for a in range(params.num_agents):
                    agent_id = F.one_hot(th.tensor(a), num_classes=params.num_agents).float().unsqueeze(0).repeat(
                        mb_size, 1).to(device)
                    old_logp = logp_mb[:, a]
                    action = action_mb[:, a]

                    x = th.cat([state_mb, agent_id], dim=-1)
                    logits = actor_shared(x)
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(action)
                    entropy = dist.entropy().mean()

                    actor_loss = actor_shared.actor_loss(new_logp, old_logp, adv_mb, params.eps_clip)
                    total_actor_loss += actor_loss - params.entropy_coef * entropy
                total_actor_loss /= params.num_agents
                total_actor_loss.backward()
                actor_opt.step()





    @staticmethod
    def test(actor, params, env):
        device = params.device
        env.testing_mode = True
        actor.flatten_parameters()
        actor.train(mode=True)

        test_rewards = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            if params.partial_observability:
                hidden_states_actor = [th.zeros(1, 1, params.actor_hidden_dim[0]).to(device) for _ in
                                       range(params.num_agents)]
                prev_actions = [th.zeros(1, params.action_dim).to(device) for _ in range(params.num_agents)]
            else:
                hidden_states_actor, prev_actions = None, None

            total_rewards = 0

            global_state, local_states, fp_global_states = env.reset()


            for k in range(1, params.k_max + 1):

                if k > 1:
                    env.reset_control(k)

                for t in range(params.t_max):

                    actions = []  #No need to store these?
                    global_state = th.tensor(global_state, dtype=th.float32, device=device)
                    if local_states is not None:
                        local_states = {agent_id: th.tensor(state, dtype=th.float32, device=device)
                                        for agent_id, state in local_states.items()}

                    for a in range(params.num_agents):
                        agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device)
                        if params.partial_observability:
                            x = th.cat([local_states[a], agent_id, prev_actions[a]], dim=-1).unsqueeze(0)
                            logits, h_a = actor(x, hidden_states_actor[a])
                            hidden_states_actor[a] = h_a
                            action = actor.select_action(logits)[0]
                            prev_actions[a] = F.one_hot(th.tensor([action.item()]), num_classes=params.action_dim).float().to(device)
                        else:
                            x = th.cat([global_state, agent_id], dim=-1)
                            logits = actor(x)
                            action = actor.select_action(logits)[0]

                        actions.append(action.item())

                    global_next_state, local_next_states, _, global_reward, individual_rewards, done = env.step(actions, k, t)

                    global_state = global_next_state
                    local_states = local_next_states
                    total_rewards += global_reward

                    if done:
                        break

            test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        env.testing_mode = False
        return average_reward

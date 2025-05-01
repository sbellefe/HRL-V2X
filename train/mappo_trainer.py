import sys
from copy import deepcopy
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F

from Agent.mappo import MAPPO_Actor, MAPPO_Critic
from helpers.mappo_helper import BatchProcessing, compute_GAE, pre_process

# from Envs.UtilityCommunication.veh_position_helper import *
# from Envs.zenv_helper import EnvironHelper
# from Envs.zzenv_params import V2Xparams
# from Envs.Environment import Environ


class MAPPOtrainer:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        device = params.device

        """Initialize network(s) and optimizers"""
        actor = MAPPO_Actor(params).to(device)

        if not params.partial_observability:
            actor_shared = MAPPO_Actor(params).to(device)
            central_critic = MAPPO_Critic(params).to(device)
            actor_opt = th.optim.Adam(actor_shared.parameters(), lr=params.lr_actor)
        else:
            actor_list = actors = [MAPPO_Actor(params.state_dim, params.actor_hidden_dim, params.action_dim).to(device) for _ in range(params.num_agents)]
        #

        critic_opt = th.optim.Adam(central_critic.parameters(), lr=params.lr_critic)

        # actors = [MAPPO_Actor(params.state_dim, params.actor_hidden_dim, params.action_dim).to(device) for _ in range(params.num_agents)]
        # critics = [MAPPO_Critic(params.state_dim, params.critic_hidden_dim).to(device) for _ in range(params.num_agents)]



        train_rewards = []
        test_rewards = []
        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []

            for ep in range(params.buffer_episodes):
                state_history = []  #global state
                action_history = [] #joint actions
                logp_history = []
                reward_history = []
                value_history = []
                obs_history = []   #dont think I need this?

                total_reward = 0

                #gather episode positional data and load first sample
                global_state, local_states = env.reset()
                global_state = th.tensor(global_state, dtype=th.float32).squeeze()

                #loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):

                    if k > 1:
                        env.reset_control(k)


                    #loop for timesteps in control interval
                    for t in range(params.t_max):

                        actions, logps, values = [],[],[]
                        RRA_all_agents = np.zeros([params.num_agents, env.n_neighbor, 2], dtype='int32')

                        # Collect actions/logp/values for each agent, store in histories
                        # global_state = env.get_state([0, 0], t)
                        # global_state = th.tensor(global_state, dtype=th.float32).squeeze().to(device)

                        #loop through each agent
                        for a in range(params.num_agents):
                            agent_id = F.one_hot(th.tensor(a), num_classes=params.num_agents).float().to(device)
                            # print(f"Agent {a} is {agent_id}")

                            with th.no_grad():
                                logits = actor_shared(global_state, agent_id)
                                action, logp = actor_shared.select_action(logits)
                                # print(f"{env.action_dim} - Agent {a} action: {action}")
                                logps.append(logp)
                                actions.append(action.item())
                                # value = central_critic(global_state, agent_id)
                                # values.append(value.item())
                                # RRA_all_agents[a, 0, 0], RRA_all_agents[a, 0, 1] = env.mapping_action2RRA(action)


                        # print(f"ep={n_ep}; t={t}. a={actions}   |\n    RRA_all_agents={RRA_all_agents[:, 0, :] }\n\n")
                        with th.no_grad():
                            value = central_critic(global_state)


                        #Take step with joint actions
                        # joint_actions = actions
                        # global_next_state, local_next_states, global_reward, done = env.step(RRA_all_agents.copy(), k, t)
                        global_next_state, local_next_states, global_reward, done = env.step(actions, k, t)


                        # print(f"\n*************Timestep {t}: \n"
                        #       f"global_state: {global_state.shape}\n"
                        #       f"joint_actions: {joint_actions}\n"
                        #       f"RRA_all_agents: {RRA_all_agents.shape}\n"
                        #       f"logp: {logps}\n"
                        #       f"global_reward: {global_reward}\n"
                        #       f"value: {value}\n"
                        #       )

                        state_history.append(global_state)
                        action_history.append(actions)
                        logp_history.append(logps)
                        # reward_history.append(individual_rewards)
                        reward_history.append(global_reward)
                        value_history.append(value)
                        # value_history.append(values)

                        total_reward += global_reward
                        global_state = th.tensor(global_next_state, dtype=th.float32).squeeze()

                        if params.partial_observability:
                            obs_history.append(local_next_states)
                            local_states = th.tensor(local_next_states, dtype=th.float32).squeeze()

                        if done:
                            print(f'****Training Episode {n_ep + 1} complete. pos_idx = {env.current_veh_positions.iloc[0]["time"]} | '
                                  f'Global Reward: {total_reward:.3f}')

                            break

                n_ep += 1
                train_rewards.append(total_reward)

                # compute returns and advantages for episode, add episode to buffer
                returns, advantages = compute_GAE(reward_history, value_history, params.gamma, params.gae_lambda, device)
                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                # print(f"Single Episode Data: {n_ep}\n"
                #       f"State History: {state_history[2].shape}\n"
                #       f"Action History: {action_history}\n"
                #       f"LogP History: {logp_history}\n"
                #       f"Value History: {value_history}\n"
                #       f"Returns: {returns.shape}\n"
                #       f"Advantages: {advantages.shape}")

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

                    print(f'Test return at episode {n_ep}: {test_reward:.3f}. Average train return: {np.mean(train_rewards[-params.test_interval]):.3f}')

            #process buffer once full
            batch_process = BatchProcessing()
            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages \
                = batch_process.collate_batch(buffer, params.device)

            #convert to dataset and initialize dataloader for mini_batch sampling
            dataset = th.utils.data.TensorDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns,
                                                  batch_advantages)
            mini_batch_size = len(dataset) // params.num_mini_batches
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
            #optimization loop
            loss_p, loss_c = [], []
            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    states_mb, actions_mb, logp_mb, values_mb, returns_mb, advantages_mb = batch

                    states_mb = states_mb.to(params.device)
                    actions_mb = actions_mb.to(params.device)
                    logp_mb = logp_mb.to(params.device)
                    values_mb = values_mb.to(params.device)
                    returns_mb = returns_mb.to(params.device)
                    advantages_mb = advantages_mb.to(params.device)

                    # Critic Update
                    critic_opt.zero_grad()
                    total_critic_loss = 0
                    values_new = central_critic(states_mb)
                    critic_loss = central_critic.critic_loss(values_new, values_mb, returns_mb, params.eps_clip)

                    # for a in range(params.num_agents):
                    #     agent_id = F.one_hot(th.tensor(a), num_classes=params.num_agents).float().unsqueeze(0).repeat(states_mb.size(0), 1).to(device)
                    #     values = central_critic(states_mb, agent_id)
                    #     critic_loss = central_critic.critic_loss(values, values_mb, returns_mb, params.eps_clip)
                    #     total_critic_loss += critic_loss

                    # total_critic_loss /= params.num_agents
                    # total_critic_loss.backward()
                    critic_loss.backward()
                    critic_opt.step()

                    # Actor Update
                    actor_opt.zero_grad()
                    total_actor_loss = 0
                    for a in range(params.num_agents):
                        observation = states_mb.to(device)
                        action = actions_mb[:, a]
                        old_log_probs = logp_mb[:, a]
                        agent_id = F.one_hot(th.tensor(a), num_classes=params.num_agents).float().unsqueeze(0).repeat(states_mb.size(0), 1).to(device)

                        logits = actor_shared(observation, agent_id)
                        dist = Categorical(logits=logits)
                        new_log_probs = dist.log_prob(action)

                        actor_loss = actor_shared.actor_loss(new_log_probs, old_log_probs, advantages_mb, params.eps_clip)

                        entropy = dist.entropy().mean()
                        actor_loss = actor_loss - params.entropy_coef * entropy

                        total_actor_loss += actor_loss

                    total_actor_loss /= params.num_agents
                    total_actor_loss.backward()
                    actor_opt.step()

                    loss_p.append(total_actor_loss)
                    loss_c.append(total_critic_loss)




        # Optional print average optimization losses
        # av_loss_p, av_loss_c = sum(loss_p)/len(loss_p), sum(loss_c)/len(loss_c)
        # print(f"Optimization avg losses: Policy loss: {av_loss_p:.3f} | Critic loss: {av_loss_c:.3f}")

        print(f"Trial Complete. Max test reward {max(test_rewards):.3f}.")
        return train_rewards, test_rewards

    @staticmethod
    def test(actor, params, env):

        env.testing_mode = True

        test_rewards = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):

            total_rewards = 0

            # test_data = test_data_list[i % len(test_data_list)]
            # params.loaded_veh_data = test_data
            # env.loaded_veh_data = params.loaded_veh_data

            # env.new_random_game()
            #
            # if params.game_mode == 1 or params.game_mode == 2:
            #     num_control_interval = 1
            # else:
            #     num_control_interval = params.t_max_control

            global_state, local_states = env.reset()
            global_state = th.tensor(global_state, dtype=th.float32).squeeze()

            for k in range(1, params.k_max + 1):

                if k > 1:
                    env.reset_control()

                for t in range(params.t_max):

                    actions = []  #No need to store these?
                    # RRA_all_agents = np.zeros([env.num_veh_V2V - 1, env.n_neighbor, 2], dtype='int32')


                    for a in range(params.num_agents):
                        agent_id = F.one_hot(th.tensor(a), num_classes=params.num_agents).float()
                        logits = actor(global_state, agent_id)
                        action, _ = actor.select_action(logits)
                        actions.append(action.item())
                        # RRA_all_agents[a, 0, 0], RRA_all_agents[a, 0, 1] = env.mapping_action2RRA(action)

                    joint_action = actions
                    global_next_state, _, global_reward, done = env.step(actions, k, t)
                    global_state = th.tensor(global_next_state, dtype=th.float32).squeeze()

                    total_rewards += global_reward


            test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        env.testing_mode = False
        return average_reward


    # @staticmethod
    # def test(actor, params, n_ep, goal):
    #     """tests Agent and averages result, configure whether to show (render)
    #         testing and how long to delay in ParametersPPO class"""
    #     render_testing = params.show_testing and n_ep > params.render_delay
    #
    #     if params.env_name == 'FourRooms':
    #         test_env = FourRooms(render_mode="human") if render_testing else FourRooms()
    #         test_env.choose_goal(goal)
    #     elif params.env_name == 'FourRooms_m':
    #         test_env = FourRooms_m(render_mode="human") if render_testing else FourRooms_m()
    #         test_env.choose_goal(goal)
    #     else:
    #         test_env = gym.make(params.env_name)  # , render_mode="human")
    #
    #     test_rewards = np.zeros(params.test_episodes)
    #     episode_lengths = np.zeros(params.test_episodes)
    #     test_returns = np.zeros(params.test_episodes)
    #
    #     for i in range(params.test_episodes):
    #         obs, _ = test_env.reset()
    #         rewards = []
    #
    #         for t in range(params.t_max):
    #             state = pre_process(obs)
    #             logits = actor(state)
    #             action, _, _ = actor.select_action(logits)
    #             next_obs, reward, done, trunc, _ = test_env.step(action.item())
    #
    #             test_env.render(i)
    #
    #             rewards.append(reward)
    #             # total_reward += reward
    #             obs = next_obs
    #             if done or trunc:
    #                 episode_lengths[i] = t + 1
    #                 break
    #         test_rewards[i] = sum(rewards) #Check if this works?
    #
    #         #compute discounted return
    #         gt = 0
    #         for k in reversed(range(len(rewards))):
    #             gt = rewards[k] + params.gamma * gt
    #         test_returns[i] = gt
    #
    #     test_env.close()
    #
    #     # average_reward = np.mean(test_rewards)
    #     average_length = np.mean(episode_lengths)
    #     average_return = np.mean(test_returns)
    #
    #     return average_return, average_length
import sys
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch as th
from torch.distributions import Categorical

from Agent.ippo import IPPO_Actor, IPPO_Critic
from helpers.ppo_helper import BatchProcessing, compute_GAE, pre_process

from Env.UtilityCommunication.veh_position_helper import *

class IPPOtrainer:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        device = params.device
        # actor = PPO_Actor(params.state_dim, params.actor_hidden_dim, params.action_dim).to(device)
        # critic = PPO_Critic(params.state_dim, params.critic_hidden_dim).to(device)

        actors = [IPPO_Actor(params.state_dim, params.actor_hidden_dim, params.action_dim).to(device) for _ in range(params.num_agents)]
        critics = [IPPO_Critic(params.state_dim, params.critic_hidden_dim).to(device) for _ in range(params.num_agents)]


        opt = th.optim.Adam([
                    {'params': actor.parameters(), 'lr': params.lr_actor} for actor in actors
                ] + [
                    {'params': critic.parameters(), 'lr': params.lr_critic} for critic in critics
                ])


        training_rewards = []
        testing_rewards = []
        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []
            # episode_rewards = [0 for _ in range(params.num_agents)]

            for ep in range(params.buffer_episodes):
                state_history = []
                action_history = []
                logp_history = []
                reward_history = []
                value_history = []
                done_history = []

                # state_history = [[] for _ in range(params.num_agents)]
                # action_history = [[] for _ in range(params.num_agents)]
                # logp_history = [[] for _ in range(params.num_agents)]
                # reward_history = [[] for _ in range(params.num_agents)]
                # value_history = [[] for _ in range(params.num_agents)]


                # for "game modes 1 or 2" ?
                num_control_interval = 1

                if params.env_name == 'NFIG':
                    sampled_data = sample_veh_position_from_timestep(env.veh_pos_data, params.env_setup)  # [25.0, 30.0, 35.0, 65.0]
                elif params.env_name == 'SIG':
                    sampled_data = sample_veh_positions(num_control_interval, env.veh_pos_data)
                else:
                    raise NotImplementedError

                env.loaded_veh_data = sampled_data

                env.new_random_game()

                for interval in range(1, num_control_interval + 1):

                    if interval > 1:
                        env.renew_positions_by_file(interval)
                        env.renew_channel()
                        env.renew_queue()

                    for t in range(params.n_step_per_episode_communication):

                        env.renew_fast_fading()


                        RRA_all_agents = np.zeros([params.n_veh - 1, params.n_neighbor, 2], dtype='int32')

                        # Collect actions/logp/values for each agent
                        for a in range(params.num_agents):
                            with th.no_grad():
                                obs = env.get_state([a, 0], 0, t)
                                state = th.tensor(obs, dtype=th.float32).squeeze().to(device)
                                action, logp = actors[a](state)
                                value = critics[a](state)

                            state_history[a].append(state)
                            action_history[a].append(action)
                            logp_history[a].append(logp)
                            value_history[a].append(value)

                            RRA_all_agents[a, 0, 0], RRA_all_agents[a, 0, 1] = actors[a].mapping_action2RRA(action)

                        global_reward, individual_rewards, V2I_throughput = env.reward_step(RRA_all_agents.copy())

                        if params.game_mode == 1:
                            global_reward = global_reward + sum(V2I_throughput)
                        else:
                            global_reward = global_reward

                        for i in range(params.num_agents):
                            reward_history[i].append(individual_rewards[i])
                            episode_rewards[i] += individual_rewards[i]

                        if params.game_mode == 1 or params.game_mode == 2:
                            if t == params.n_step_per_episode_communication - 1:
                                done = True
                            else:
                                done = False
                        else:
                                if t == params.n_step_per_episode_communication - 1 and interval == params.t_max:
                                    done = True
                                else:
                                    done = False

                        done_history.append(done)

                        if done:
                            break
                n_ep += 1

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    # show_testing = False if n_ep < params.render_delay and params.show_testing else True
                    test_reward = self.test(deepcopy(actors), params, env.test_data_list, n_ep)
                    testing_rewards.append(test_reward)
                    print(f'Test return at episode {n_ep}: {test_reward:.3f}.')










                obs, _ = env.reset()
                total_reward = 0

                for t in range(params.t_max):
                    state = pre_process(obs).to(device)

                    #select action, compute value estimate
                    with th.no_grad():
                        logits = actor(state)
                        action, logp, _ = actor.select_action(logits)
                        value = critic(state)

                    #take a step
                    next_obs, reward, terminated, truncated, _ = env.step(action.item())

                    #store transition
                    state_history.append(state)
                    action_history.append(action)
                    logp_history.append(logp)
                    reward_history.append(reward)
                    value_history.append(value)

                    total_reward += reward
                    obs = next_obs

                    if terminated or truncated:     #Optional for printing train episode lengths
                        # print(f"****training episode {n_ep+1}: {t+1} steps ****")
                        pass

                    #logic for episode termination/truncation
                    if truncated:   #Compute next value if episode Env timelimit is reached
                        next_state = pre_process(obs).to(device)
                        with th.no_grad():
                            next_value = critic(next_state)
                        value_history.append(next_value)
                        break
                    if terminated: #Compute next value = 0 if terminal state reached
                        next_value = th.zeros_like(value)
                        value_history.append(next_value)
                        break

                episode_rewards.append(total_reward)
                n_ep += 1

                #compute returns and advantages for episode, add episode to buffer
                returns, advantages = compute_GAE(reward_history, value_history, params.gamma, params.gae_lambda, device)
                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                #test at interval and print result
                if n_ep % params.test_interval == 0:
                    # show_testing = False if n_ep < params.render_delay and params.show_testing else True
                    test_return, episode_length = self.test(deepcopy(actor), params, n_ep, env.goal)
                    test_returns.append(test_return)
                    test_episode_lengths.append(episode_length)
                    print(f'Test return at episode {n_ep}: {test_return:.3f} | '
                          f'Average test episode length: {episode_length}')

                #Switch Goal location
                if params.switch_goal and n_ep == params.total_train_episodes // 2:
                    env.switch_goal(goal=params.new_goal)
                    print(f"New goal {env.goal}. Max return so far: {max(test_returns):.3f}")

            #process buffer once full
            batch_process = BatchProcessing()
            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages \
                = batch_process.collate_batch(buffer, params.device)

            #convert to dataset and initialize dataloader for mini_batch sampling
            dataset = th.utils.data.TensorDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns,
                                                  batch_advantages)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True)

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
                    values_new = critic(states_mb)
                    critic_loss = critic.critic_loss(values_new, values_mb, returns_mb, params.eps_clip)
                    critic_loss.backward()
                    critic_opt.step()

                    # Actor Update
                    actor_opt.zero_grad()
                    dist = Categorical(logits=actor(states_mb))
                    logp_new = dist.log_prob(actions_mb)
                    entropy = dist.entropy().mean()
                    actor_loss = actor.actor_loss(logp_new, logp_mb, advantages_mb, params.eps_clip)
                    actor_loss = actor_loss - params.entropy_coef * entropy
                    actor_loss.backward()
                    actor_opt.step()

                    loss_p.append(actor_loss)
                    loss_c.append(critic_loss)

            # Optional print average optimization losses
            # av_loss_p, av_loss_c = sum(loss_p)/len(loss_p), sum(loss_c)/len(loss_c)
            # print(f"Optimization avg losses: Policy loss: {av_loss_p:.3f} | Critic loss: {av_loss_c:.3f}")

        print(f"Trial Complete. Max test returns for: "
              f"G1 = {max(test_returns[:len(test_returns) // 2]):.3f}, "
              f"G2 = {max(test_returns[-len(test_returns) // 2:]):.3f}")
        return episode_rewards, test_returns, test_episode_lengths

    @staticmethod
    def test(actor, params, n_ep, goal):
        """tests Agent and averages result, configure whether to show (render)
            testing and how long to delay in ParametersPPO class"""
        render_testing = params.show_testing and n_ep > params.render_delay

        if params.env_name == 'FourRooms':
            test_env = FourRooms(render_mode="human") if render_testing else FourRooms()
            test_env.choose_goal(goal)
        elif params.env_name == 'FourRooms_m':
            test_env = FourRooms_m(render_mode="human") if render_testing else FourRooms_m()
            test_env.choose_goal(goal)
        else:
            test_env = gym.make(params.env_name)  # , render_mode="human")

        test_rewards = np.zeros(params.test_episodes)
        episode_lengths = np.zeros(params.test_episodes)
        test_returns = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            obs, _ = test_env.reset()
            rewards = []

            for t in range(params.t_max):
                state = pre_process(obs)
                logits = actor(state)
                action, _, _ = actor.select_action(logits)
                next_obs, reward, done, trunc, _ = test_env.step(action.item())

                test_env.render(i)

                rewards.append(reward)
                # total_reward += reward
                obs = next_obs
                if done or trunc:
                    episode_lengths[i] = t + 1
                    break
            test_rewards[i] = sum(rewards) #Check if this works?

            #compute discounted return
            gt = 0
            for k in reversed(range(len(rewards))):
                gt = rewards[k] + params.gamma * gt
            test_returns[i] = gt

        test_env.close()

        # average_reward = np.mean(test_rewards)
        average_length = np.mean(episode_lengths)
        average_return = np.mean(test_returns)

        return average_return, average_length
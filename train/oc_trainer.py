import sys
from copy import deepcopy
import gymnasium as gym
from itertools import count
import numpy as np
import torch as th

from Env.fourrooms import FourRooms, FourRooms_m
from Agent.oc import OC_Network
from helpers.oc_helper import ReplayBuffer, pre_process


class OCtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device

        agent = OC_Network(params).to(device)
        agent_prime = deepcopy(agent)
        # opt = th.optim.RMSprop(Agent.parameters(), lr=params.lr)
        # for name, param in Agent.named_parameters():
        #     print(name, param.shape)
        # sys.exit()
        opt = th.optim.Adam([
            {'params': [p for n, p in agent.named_parameters() if 'Q' in n], 'lr': params.lr_q},  # critic (master policy)
            {'params': [p for n, p in agent.named_parameters() if 'beta' in n], 'lr': params.lr_beta},  # beta
            {'params': [p for n, p in agent.named_parameters() if 'options' in n], 'lr': params.lr_la}, #sub -policies
            {'params': [p for n, p in agent.named_parameters() if 'features' in n], 'lr': params.lr_phi}, #features
        ])


        buffer = ReplayBuffer(params.buffer_size)

        episode_rewards = []
        test_returns = []
        test_episode_lengths = []
        episode_lengths = []

        params.t_tot = 0
        n_ep = 0

        if params.switch_goal: print(f"Current goal {env.goal}")
        running_av_length = 0

        for _ in count():

            episode_reward = 0
            option_lengths = {op: [] for op in range(params.num_options)}

            obs, _ = env.reset()

            state = agent.get_state(pre_process(obs))
            greedy_option = agent.greedy_option(state)
            current_option = 0

            done = False
            ep_steps = 0
            beta = True
            curr_op_len = 0

            #Episode loop
            while not done and ep_steps < params.t_max:
                epsilon = params.epsilon #get current epsilon

                if beta:    #if current option terminates (beta = True)
                    #record option length
                    option_lengths[current_option].append(curr_op_len)

                    #epsilon-greedy option selection
                    if np.random.rand() < epsilon:
                        current_option = np.random.choice(params.num_options)
                    else:
                        current_option = greedy_option
                    curr_op_len = 0

                #get action according to current option
                action, logp, entropy = agent.get_action(state, current_option, params.temp)

                #take step and store in buffer
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(obs, current_option, reward, next_obs, done)
                episode_reward += reward

                if done:  # Optional for printing train episode lengths
                    # print(f"****training episode {n_ep+1}: {ep_steps+1} steps | epsilon = {epsilon:.3f} ****")
                    episode_lengths.append(ep_steps+1)
                    running_av_length += ep_steps+1

                actor_loss, critic_loss = None, None
                if len(buffer) > params.batch_size:
                    #compute actor loss every timestep
                    actor_loss = agent.actor_loss(agent_prime, obs, current_option, logp, entropy, reward, done, next_obs, params)
                    loss = actor_loss

                    #compute critic loss at interval
                    if params.t_tot % params.critic_optim_freq == 0:
                        batch = buffer.sample(params.batch_size)
                        critic_loss = agent.critic_loss(agent_prime, batch, params)
                        loss += critic_loss

                    opt.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(agent.parameters(), params.gradient_clip)
                    opt.step()

                    #hard update target network at interval
                    if params.t_tot % params.target_update_freq == 0:
                        agent_prime.load_state_dict(agent.state_dict())

                #get next state
                state = agent.get_state(pre_process(next_obs))

                #get next state beta and greedy option TODO: rename this method?
                beta, greedy_option = agent.get_beta(state, current_option)

                #update counters
                params.t_tot += 1
                ep_steps += 1
                curr_op_len +=1
                obs = next_obs

                #TODO: log data here?

            episode_rewards.append(episode_reward)
            n_ep += 1

            # test at interval and print result
            if n_ep % params.test_interval == 0:
                test_return, episode_length = self.test(deepcopy(agent), params, n_ep, env.goal)
                test_returns.append(test_return)
                test_episode_lengths.append(episode_length)
                running_av_len = sum(episode_lengths[-10:]) / 10
                print(f'Test return at episode {n_ep}: {test_return:.3f} | '
                      f'Average test (train) episode length: {episode_length} ({running_av_len:.1f}) | '
                      f'Total steps: {params.t_tot} | '
                      f'Epsilon: {params.epsilon:.3f}')


            # Switch Goal location and experiment termination
            if params.switch_goal and n_ep == params.total_train_episodes // 2:
                env.switch_goal(goal=params.new_goal)
                print(f"New goal {env.goal}. Max return so far: {max(test_returns):.3f}")
            if n_ep >= params.total_train_episodes:
                break

        print(f"Trial Complete. Max test returns for: "
              f"G1 = {max(test_returns[:len(test_returns) // 2]):.3f}, "
              f"G2 = {max(test_returns[-len(test_returns) // 2:]):.3f}")
        return episode_rewards, test_returns, test_episode_lengths

    @staticmethod
    def test(agent, params, n_ep, goal):
        """tests Agent and averages result, configure whether to show (render)
                    testing and how long to delay in ParametersPPO class"""
        render_testing = params.show_testing and n_ep > params.render_delay
        agent.train(mode=False)

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
            rewards = []

            obs, _ = test_env.reset()
            state = agent.get_state(pre_process(obs))
            greedy_option = agent.greedy_option(state)

            current_option = 0
            beta = True

            for t in range(params.t_max):

                if beta: # epsilon-greedy option selection with eps_test
                   if np.random.rand() < params.eps_test:
                       current_option = np.random.choice(params.num_options)
                   else:
                       current_option = greedy_option

                action, _, _ = agent.get_action(state, current_option, params.temp)

                next_obs, reward, done, trunc, _ = test_env.step(action)

                # render environment, including metrics
                test_env.render(i, text_top=f"Current option = {current_option}",
                                text_bot=f"beta = {beta:.1f}")

                next_state = agent.get_state(pre_process(next_obs))
                beta, greedy_option = agent.get_beta(next_state, current_option)

                state = next_state
                rewards.append(reward)

                if done or trunc:
                    episode_lengths[i] = t + 1
                    break
            test_rewards[i] = sum(rewards) #Check if this works?

            # compute discounted return
            gt = 0
            for k in reversed(range(len(rewards))):
                gt = rewards[k] + params.gamma * gt
            test_returns[i] = gt

        test_env.close()

        average_length = np.mean(episode_lengths)
        average_return = np.mean(test_returns)

        return average_return, average_length

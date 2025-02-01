import sys
from copy import deepcopy
import numpy as np
import torch as th
import gymnasium as gym
from torch.distributions import Categorical

from env.fourrooms import FourRooms, FourRooms_m
from agent.dac import DAC_Actor, DAC_Critic, DAC_Beta
from helpers.dac_helper import BatchProcessing, compute_GAE, pre_process

class DACtrainer():
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device

        actor_h = DAC_Actor(params, mdp='high')
        actor_l = DAC_Actor(params, mdp='low')
        critic = DAC_Critic(params)
        beta_net = DAC_Beta(params)

        opt_actor_h = th.optim.Adam(actor_h.parameters(), lr=params.lr_ha)
        opt_actor_l = th.optim.Adam(actor_l.parameters(), lr=params.lr_la)
        opt_critic = th.optim.Adam(critic.parameters(), lr=params.lr_critic)
        opt_beta = th.optim.Adam(beta_net.parameters(), lr=params.lr_beta)

        # opt = th.optim.Adam([
        #     {'params': actor_h.parameters(), 'lr': params.lr_ha},
        #     {'params': actor_l.parameters(), 'lr': params.lr_la},
        #     {'params': critic.parameters(), 'lr': params.lr_critic},
        #     {'params': beta_net.parameters(), 'lr': params.lr_beta}],
        #     weight_decay=1e-4)

        episode_rewards = []
        test_returns = []
        test_episode_lengths = []

        if params.switch_goal: print(f"Current goal {env.goal}")

        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []

            for ep in range(params.buffer_episodes):
                state_h_history, state_l_history, state_beta_history = [],[],[]
                beta_history, option_history, prev_option_history, reward_history = [],[],[],[]
                logp_h_history, logp_l_history, v_h_history, v_l_history = [],[],[],[]


                obs, _ = env.reset()
                beta = th.Tensor([[1.0]])   #set initial beta to 1 to reselect option
                prev_option = th.LongTensor([0]) #set initial prev_option to 0
                total_reward = 0

                for t in range(params.t_max):
                    #select option in high MDP
                    state_h = actor_h.get_state(obs)
                    with th.no_grad():
                        option, logp_h, pi_hat = actor_h.get_policy(state_h, beta, prev_option)

                    #select action in low MDP
                    state_l = actor_l.get_state(obs, option)
                    with th.no_grad():
                        action, logp_l, pi_bar = actor_l.get_policy(state_l)

                    # print(f"Obs:{obs.shape}\n"
                    #       f"state_h:{state_h.shape}\n"
                    #       f"state_l:{state_l.shape}\n")
                    # sys.exit()
                    #compute values
                    with th.no_grad():
                        v_h, v_l = critic.get_values(state_l, pi_hat)


                    #take a step
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    #compute beta for next state-option TODO: final state condition?
                    state_beta = beta_net.get_state(next_obs, option)
                    with th.no_grad():
                        beta = beta_net(state_beta)

                    #store transition data
                    state_h_history.append(state_h)
                    state_l_history.append(state_l)
                    state_beta_history.append(state_beta)
                    reward_history.append(reward)
                    beta_history.append(beta)
                    option_history.append(option)
                    prev_option_history.append(prev_option)
                    v_h_history.append(v_h)
                    v_l_history.append(v_l)
                    logp_h_history.append(logp_h)
                    logp_l_history.append(logp_l)

                    obs = next_obs
                    prev_option = option
                    total_reward += reward

                    if terminated or truncated:  # Optional for printing train episode lengths
                        print(f"****training episode {n_ep+1}: {t+1} steps ****")

                    # logic for episode termination/truncation
                    if truncated:  # Compute next value if episode env timelimit is reached
                        # compute values
                        next_state_l = actor_l.get_state(next_obs, option)
                        with th.no_grad():
                            next_v_h, next_v_l = critic.get_values(next_state_l, pi_hat)
                        v_h_history.append(next_v_h)
                        v_l_history.append(next_v_l)
                        break
                    if terminated:  # Compute next value = 0 if terminal state reached
                        next_value = th.zeros_like(v_h)
                        v_h_history.append(next_value)
                        v_l_history.append(next_value)
                        break

                n_ep += 1
                episode_rewards.append(episode_rewards)

                #compute advantages and returns for episode
                returns, adv_h, adv_l = compute_GAE(reward_history, v_h_history, v_l_history,
                                                    params.gamma, params.gae_lambda, params.device)

                #store episode in buffer
                buffer.append((state_h_history, state_l_history, state_beta_history,
                               option_history, prev_option_history, beta_history,
                               returns, adv_h, adv_l, logp_h_history, logp_l_history,
                               v_h_history, v_l_history))

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    # show_testing = False if n_ep < params.render_delay and params.show_testing else True
                    test_return, episode_length = self.test(deepcopy(actor_h),deepcopy(actor_l),deepcopy(beta_net), params, n_ep, env.goal)
                    test_returns.append(test_return)
                    test_episode_lengths.append(episode_length)
                    print(f'Test return at episode {n_ep}: {test_return:.3f} | '
                          f'Average test episode length: {episode_length}')

                # Switch Goal location
                if params.switch_goal and n_ep == params.total_train_episodes / 2:
                    env.switch_goal()
                    print(f"New goal {env.goal}")

            # process buffer once full
            batch_process = BatchProcessing()
            (states_h_mb, states_l_mb, states_beta_mb,
             options_mb, prev_options_mb, betas_mb,
             returns_mb, adv_h_mb, adv_l_mb,
             logp_h_mb, logp_l_mb, v_h_mb, v_l_mb) = batch_process.collate_batch(buffer, params.device)

            # convert to dataset and initialize dataloader for mini_batch sampling
            dataset = th.utils.data.TensorDataset(states_h_mb, states_l_mb, states_beta_mb,
                                                  options_mb, prev_options_mb, betas_mb,
                                                  returns_mb, adv_h_mb, adv_l_mb,
                                                  logp_h_mb, logp_l_mb, v_h_mb, v_l_mb)

            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True)

            #Optimize model
            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    #unpack mini batch
                    (states_h_mb, states_l_mb, states_beta_mb,
                     options_mb, prev_options_mb, betas_mb,
                     returns_mb,adv_h_mb, adv_l_mb,
                     logp_h_mb, logp_l_mb, v_h_mb, v_l_mb) = batch

                    #high actor loss
                    opt_actor_h.zero_grad()
                    _, new_logp_h, pi_hat = actor_h.get_policy(states_h_mb, betas_mb, prev_options_mb)
                    entropy_h = Categorical(probs=pi_hat).entropy().mean()
                    loss_actor_h = actor_h.actor_loss(new_logp_h, logp_h_mb, adv_h_mb)
                    loss_actor_h += -params.entropy_coef * entropy_h
                    loss_actor_h.backward()
                    opt_actor_h.step()

                    #low actor loss
                    opt_actor_l.zero_grad()
                    _, new_logp_l, pi_bar = actor_l.get_policy(states_l_mb)
                    entropy_l = pi_bar.entropy().mean()
                    loss_actor_l = actor_l.actor_loss(new_logp_l, logp_l_mb, adv_l_mb)
                    loss_actor_l += -params.entropy_coef * entropy_l
                    loss_actor_l.backward()
                    opt_actor_l.step()

                    #beta network loss TODO: beta_reg??
                    opt_beta.zero_grad()
                    loss_beta = beta_net.beta_loss(states_beta_mb, _,adv_l_mb)
                    loss_beta.backward()
                    opt_beta.step()

                    #critic loss
                    opt_critic.zero_grad()
                    new_v_h, new_v_l = critic.get_values(states_l_mb, pi_hat.detach())
                    loss_critic_h = critic.critic_loss(new_v_h, v_h_mb, returns_mb)
                    loss_critic_l = critic.critic_loss(new_v_l, v_l_mb, returns_mb)
                    loss_critic = loss_critic_h + loss_critic_l
                    loss_critic.backward()
                    opt_critic.step()

                    # loss = loss_actor_h + loss_actor_l + loss_beta + loss_critic
                    # print(f"Optimization epoch {_}:\n"
                    #       f"Actor loss h: {loss_actor_h}\n"
                    #       f"Actor loss l: {loss_actor_l}\n"
                    #       f"critic loss h: {loss_critic_h}\n"
                    #       f"critic loss l: {loss_critic_l}\n"
                    #       f"beta loss: {loss_beta}\n")

                    # loss.backward()
                    # opt.step()



    @staticmethod
    def test(actor_h, actor_l, beta_net, params, n_ep, goal):
        """tests agent and averages result, configure whether to show (render)
            testing and how long to delay in ParametersPPO class"""
        # print("TESTING"); return None, None
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

            beta = th.Tensor([[1.0]])
            prev_option = np.array([0])

            for t in range(params.t_max):
                #get option
                state_h = actor_h.get_state(obs)
                option, _,_ = actor_h.get_policy(state_h, beta, prev_option)

                #get action
                state_l = actor_l.get_state(obs, option)
                action, _,_ = actor_l.get_policy(state_l)

                #take step
                next_obs, reward, done, trunc, _ = test_env.step(action.item())

                #get option termination beta
                state_beta = beta_net.get_state(next_obs, option)
                beta = beta_net(state_beta)

                test_env.render()

                rewards.append(reward)
                # total_reward += reward
                obs = next_obs
                if done or trunc:
                    episode_lengths[i] = t + 1
                    break
            test_rewards[i] = sum(rewards)  # Check if this works?

            # compute discounted return
            gt = 0
            for k in reversed(range(len(rewards))):
                gt = rewards[k] + params.gamma * gt
            test_returns[i] = gt

        test_env.close()

        # average_reward = np.mean(test_rewards)
        average_length = np.mean(episode_lengths)
        average_return = np.mean(test_returns)

        return average_return, average_length


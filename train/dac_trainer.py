import sys
from copy import deepcopy
import numpy as np
import torch as th
import gymnasium as gym
from torch.distributions import Categorical

from Env.fourrooms import FourRooms, FourRooms_m
from Agent.dac import DAC_Network
from helpers.dac_helper import BatchProcessing, compute_GAE, pre_process, compute_pi_hat

class DACtrainer():
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device

        #initialize network and optimizer
        network = DAC_Network(params).to(device)
        # for name, param in network.named_parameters():
        #     print(name, param.shape)
        # sys.exit()
        opt = th.optim.Adam([
            {'params': [p for n, p in network.named_parameters() if 'pi_' in n], 'lr': params.lr_la},    #sub policy
            {'params': [p for n, p in network.named_parameters() if 'actor' in n], 'lr': params.lr_ha},     #master policy
            {'params': [p for n, p in network.named_parameters() if 'critic' in n], 'lr': params.lr_critic},
            {'params': [p for n, p in network.named_parameters() if 'beta' in n], 'lr': params.lr_beta},
            {'params': [p for n, p in network.named_parameters() if 'phi' in n], 'lr': params.lr_phi},
        ])

        episode_rewards = []
        test_returns = []
        test_episode_lengths = []

        #initialize batch processing class
        batch_process = BatchProcessing()

        if params.switch_goal: print(f"Current goal {env.goal}")
        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []
            for ep in range(params.buffer_episodes):
                #initialize episode histories
                state_history, action_history, reward_history, done_history = [],[],[],[]
                option_history, prev_option_history, beta_history = [],[],[]
                v_h_history, v_l_history, logp_h_history, logp_l_history  = [],[],[],[]
                prediction_history, pi_hat_history = [],[]

                #for troubleshooting only
                pi_bar_history, beta_history = [],[]

                #reset environment and convert to tensor
                obs, _ = env.reset()
                prev_option = None
                total_reward = 0

                for t in range(params.t_max):
                    state = pre_process(obs)
                    #forward pass through network
                    with th.no_grad():
                        prediction = network(state)

                    #compute high MDP policy, logp, and sample option
                    pi_hat = compute_pi_hat(prediction, prev_option)
                    dist = Categorical(probs=pi_hat)
                    option = dist.sample()
                    logp_h = dist.log_prob(option)

                    #compute low MDP policy for current option, logp, and sample action
                    pi_bar = prediction['pi_w'][0, option,:]
                    dist = Categorical(probs=pi_bar)
                    action = dist.sample()
                    logp_l = dist.log_prob(action)

                    #compute high and low MDP value functions
                    v_bar = prediction['q_W'][:,option]  # q value for current option
                    v_hat = (prediction['q_W'] * pi_hat).sum(-1).unsqueeze(-1)  # weighted sum of q for each option

                    #take a step
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    #store transition in episode history
                    state_history.append(state)
                    action_history.append(action)
                    reward_history.append(reward)
                    option_history.append(option)
                    prev_option_history.append(prev_option if prev_option is not None else th.LongTensor([0]))
                    v_h_history.append(v_hat)
                    v_l_history.append(v_bar)
                    logp_h_history.append(logp_h)
                    logp_l_history.append(logp_l)
                    pi_hat_history.append(pi_hat)

                    #For testing only
                    pi_bar_history.append(pi_bar)
                    beta_history.append(prediction['betas'])

                    obs = next_obs
                    prev_option = option
                    total_reward += reward

                    if terminated or truncated:  # Optional for printing train episode lengths
                        # print(f"****training episode {n_ep + 1}: {t + 1} steps ****")
                        pass

                    # logic for episode termination/truncation
                    if truncated:  # Compute next value if episode Env timelimit is reached
                        with th.no_grad():
                            prediction = network(pre_process(obs))
                        pi_hat = compute_pi_hat(prediction, prev_option)
                        next_v_bar = prediction['q_W'].gather(1, option.unsqueeze(-1))
                        next_v_hat = (prediction['q_W'] * pi_hat).sum(-1).unsqueeze(-1)
                        v_h_history.append(next_v_hat)
                        v_l_history.append(next_v_bar)
                        break
                    if terminated:  # Compute next value = 0 if terminal state reached
                        next_value = th.zeros_like(v_hat)
                        v_h_history.append(next_value)
                        v_l_history.append(next_value)
                        break
                n_ep += 1
                episode_rewards.append(total_reward)

                # compute advantages and returns for episode
                returns, adv_h, adv_l = compute_GAE(reward_history, v_h_history, v_l_history,
                                                    params.gamma, params.gae_lambda, params.device)

                # store episode in buffer
                buffer.append((state_history, action_history, pi_hat_history,
                               option_history, prev_option_history,
                               v_h_history, v_l_history,
                               logp_h_history, logp_l_history,
                               returns, adv_h, adv_l, pi_bar_history, beta_history))

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return, episode_length = self.test(deepcopy(network), params, n_ep, env.goal)
                    test_returns.append(test_return)
                    test_episode_lengths.append(episode_length)
                    print(f'Test return at episode {n_ep}: {test_return:.3f} | '
                          f'Average test episode length: {episode_length}')

                # Switch Goal location
                if params.switch_goal and n_ep == params.total_train_episodes // 2:
                    env.switch_goal(goal=params.new_goal)
                    print(f"New goal {env.goal}. Max return so far: {max(test_returns):.3f}")

            # process buffer once full
            (states_mb, actions_mb, pi_hat_mb,
             options_mb, prev_options_mb,
             v_h_mb, v_l_mb, logp_h_mb, logp_l_mb,
             returns_mb, adv_h_mb, adv_l_mb, pi_bar_mb, betas_mb) = batch_process.collate_batch(buffer, params.device)

            # convert to dataset and initialize dataloader for mini_batch sampling
            dataset = th.utils.data.TensorDataset(states_mb, actions_mb, pi_hat_mb,
                                                         options_mb, prev_options_mb,
                                                         v_h_mb, v_l_mb, logp_h_mb, logp_l_mb,
                                                         returns_mb, adv_h_mb, adv_l_mb, pi_bar_mb)

            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True)

            #initiate learning
            mdps = ['hat', 'bar']
            # np.random.shuffle(mdps)
            self.learn(network, dataloader, opt, params, mdps[1])
            self.learn(network, dataloader, opt, params, mdps[0])

        print(f"Trial Complete. Max test returns for: "
              f"G1 = {max(test_returns[:len(test_returns)//2]):.3f}, "
              f"G2 = {max(test_returns[-len(test_returns)//2:]):.3f}")
        return episode_rewards, test_returns, test_episode_lengths

    def learn(self, network, dataloader, opt, params, mdp):
        loss_p, loss_c = [], []
        for epoch in range(params.opt_epochs):
            for batch in dataloader:
                # unpack mini batch
                (states_mb, actions_mb, pi_hat_mb,
                 options_mb, prev_options_mb,
                 v_h_mb, v_l_mb,
                 old_logp_h, old_logp_l,
                 returns_mb, adv_h_mb, adv_l_mb, pi_bar_mb)  = batch

                #forward pass minibatch states
                prediction = network(states_mb)

                #Get new policies and values for each MDPNew calculations
                if mdp == 'hat':
                    #High policy
                    new_pi_hat = compute_pi_hat(prediction, prev_options_mb.view(-1))
                    dist = Categorical(probs=new_pi_hat)
                    new_logp = dist.log_prob(options_mb.view(-1)).unsqueeze(-1)
                    entropy = dist.entropy().mean()

                    #High value
                    new_v = (prediction['q_W'] * pi_hat_mb).sum(-1).unsqueeze(-1)

                elif mdp == 'bar':
                    #Low policy
                    new_pi_bar = prediction['pi_w'][th.arange(states_mb.size(0)), options_mb.view(-1),:]
                    dist = Categorical(probs=new_pi_bar)
                    new_logp = dist.log_prob(actions_mb.view(-1)).unsqueeze(-1)
                    entropy = dist.entropy().mean()

                    #Low value
                    new_v = prediction['q_W'].gather(1, options_mb)
                else:
                    raise NotImplementedError

                #PPO Actor loss with entropy
                tau = params.entropy_coef_h if mdp == 'hat' else params.entropy_coef_l
                old_logp = old_logp_h if mdp == 'hat' else old_logp_l
                advantages = adv_h_mb if mdp == 'hat' else adv_l_mb
                policy_loss = network.actor_loss(new_logp, old_logp, advantages, params.eps_clip)
                policy_loss -= entropy * tau

                #critic loss
                old_v = v_h_mb if mdp == 'hat' else v_l_mb
                critic_loss = network.critic_loss(new_v, old_v, returns_mb, params.eps_clip)

                #backpropegate
                opt.zero_grad()
                (policy_loss + critic_loss).backward()
                opt.step()

                loss_p.append(policy_loss)
                loss_c.append(critic_loss)

        #Optional print average optimization losses
        # av_loss_p, av_loss_c = sum(loss_p) / len(loss_p), sum(loss_c) / len(loss_c)
        # print(f"MDP-{mdp}, optimization complete avg losses: Policy loss: {av_loss_p:.3f} | Critic loss: {av_loss_c:.3f}")

    @staticmethod
    def test(network, params, n_ep, goal):
        """tests Agent and averages result, configure whether to show (render)
            testing and how long to delay in ParametersPPO class"""
        network.train(mode=False)
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
            state = pre_process(obs)
            rewards = []

            prev_option = None

            for t in range(params.t_max):
                #pass through network
                prediction = network(state)

                # compute high MDP policy and sample option
                pi_hat = compute_pi_hat(prediction, prev_option).squeeze(0)
                dist = Categorical(probs=pi_hat)
                option = dist.sample()

                # compute low MDP policy for current option and sample action
                pi_w = prediction['pi_w']
                pi_bar = pi_w[0, option, :].squeeze(0)
                dist = Categorical(probs=pi_bar)
                action = dist.sample()

                #take a step
                next_obs, reward, terminated, trunc, _ = test_env.step(action)
                done = terminated or trunc

                if params.show_testing:
                    q_w = prediction['q_W'].detach().squeeze(0)
                    betas = prediction['betas'].squeeze(0)
                    pi_bar = pi_bar.squeeze(0)
                    # render environment, including metrics
                    test_env.render(i, text_top=f"Option={option.item()}, Prev={prev_option} | q=[{q_w[0]:.1f},{q_w[1]:.1f},{q_w[2]:.1f},{q_w[3]:.1f}]",
                                    text_bot=f"pi_bar,hat,beta = [{pi_bar[0]:.1f},{pi_bar[1]:.1f},{pi_bar[2]:.1f},{pi_bar[3]:.1f}], "
                                             f"[{pi_hat[0]:.1f},{pi_hat[1]:.1f},{pi_hat[2]:.1f},{pi_hat[3]:.1f}], "
                                             f"[{betas[0]:.1f},{betas[1]:.1f},{betas[2]:.1f},{betas[3]:.1f}]"
                                    )

                rewards.append(reward)
                state = pre_process(next_obs)
                prev_option = option.unsqueeze(0)

                if done or trunc:
                    episode_lengths[i] = t + 1
                    break
            test_rewards[i] = sum(rewards)

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
import sys
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch as th
from torch.distributions import Categorical

from env.fourrooms import FourRooms
from agent.ppo import PPOActor, PPOCritic
from helpers.ppo_helper import BatchProcessing, compute_GAE, pre_process

class PPOtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device
        actor = PPOActor(params.state_dim, params.actor_hidden_dim, params.action_dim).to(device)
        critic = PPOCritic(params.state_dim, params.critic_hidden_dim).to(device)
        actor_opt = th.optim.Adam(actor.parameters(), lr=params.actor_lr)
        critic_opt = th.optim.Adam(critic.parameters(), lr=params.critic_lr)

        episode_rewards = []
        test_rewards = []
        test_episode_lengths = []

        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []

            for ep in range(params.buffer_episodes):
                state_history = []
                action_history = []
                logp_history = []
                reward_history = []
                value_history = []

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
                    if truncated:   #Compute next value if episode env timelimit is reached
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
                    test_reward, episode_length = self.test(deepcopy(actor), params, n_ep)
                    test_rewards.append(test_reward)
                    test_episode_lengths.append(episode_length)
                    print(f'Test reward at episode {n_ep}: {test_reward:.2f} | '
                          f'Average test episode length: {episode_length}')

            #process buffer once full
            batch_process = BatchProcessing()
            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages \
                = batch_process.collate_batch(buffer, params.device)

            #convert to dataset and initialize dataloader for mini_batch sampling
            dataset = th.utils.data.TensorDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns,
                                                  batch_advantages)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.mini_batch_size, shuffle=True)

            #optimization loop
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

        print("Trial complete")
        return episode_rewards, test_rewards, test_episode_lengths

    @staticmethod
    def test(actor, params, n_ep):
        """tests agent and averages result, configure whether to show (render)
            testing and how long to delay in ParametersPPO class"""
        render_testing = params.show_testing and n_ep > params.render_delay

        if params.env_name is None:
            test_env = FourRooms(render_mode="human") if render_testing else FourRooms()
        else:
            test_env = gym.make(params.env_name)  # , render_mode="human")

        test_rewards = np.zeros(params.test_episodes)
        episode_lengths = np.zeros(params.test_episodes)
        test_returns = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            total_reward = 0
            obs, _ = test_env.reset()
            rewards = []

            for t in range(params.t_max):
                state = pre_process(obs)
                logits = actor(state)
                action, _, _ = actor.select_action(logits)
                next_obs, reward, done, trunc, _ = test_env.step(action.item())

                test_env.render()

                rewards.append(reward)
                total_reward += reward
                obs = next_obs
                if done or trunc:
                    episode_lengths[i] = t + 1
                    break
            test_rewards[i] = total_reward

            #compute discounted return
            gt = 0
            for k in reversed(range(len(rewards))):
                gt = rewards[k] + params.gamma * gt
            test_returns[i] = gt

        test_env.close()

        average_reward = np.mean(test_rewards)
        average_length = np.mean(episode_lengths)
        average_return = np.mean(test_returns)

        return average_return, average_length
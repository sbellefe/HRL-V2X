import sys
from copy import deepcopy
import numpy as np
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F

# from Agent.dac import DAC_Actors, DAC_Critic
from Agent.dac import DAC_HighActor, DAC_LowActor, DAC_Critic
# from Agent.dac import DAC_Network
from helpers.dac_helper import RolloutBuffer_FO, compute_pi_hat#, get_flat_action
# from helpers.dac_helper import BatchProcessing, compute_GAE, pre_process

class DACtrainer:
    def __init__(self):
        pass

    def train(self, trial, env, params):
        if not params.partial_observability:
            trainer = DACtrainer_FO()
            train_returns, test_returns = trainer.train(trial, env, params)
        else:
            trainer = DACtrainer_PO()
            train_returns, test_returns = trainer.train(trial, env, params)
        return train_returns, test_returns

class DACtrainer_FO:
    def __init__(self):
        pass
        # self.actor_opt = None
        # self.critic_opt = None

    def train(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actor_h = DAC_HighActor(params).to(device)
        actor_l = DAC_LowActor(params).to(device)
        critic = DAC_Critic(params).to(device)

        opt = th.optim.Adam([
            {'params': [p for n, p in actor_h.named_parameters() if 'actor' in n], 'lr': params.lr_ha},
            {'params': [p for n, p in actor_h.named_parameters() if 'beta' in n], 'lr': params.lr_beta},
            {'params': actor_l.parameters(), 'lr': params.lr_la},
            {'params': critic.parameters(), 'lr': params.lr_critic}])
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

                """Set initial previous options - dummy option"""
                prev_options = th.full((N,), -1, dtype=th.long, device=device)  # shape [N], all –1

               # loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    # loop for timesteps in control interval
                    for t in range(params.t_max):
                        """get option-values"""
                        with th.no_grad():
                            option_values = critic(global_state)  # [1, num_options]


                        """Loop for each agent, get options, actions, logps"""
                        # 1) Allocate once (on GPU) for this time‐step:
                        options = th.empty((N,), dtype=th.long, device=device)
                        actions = th.empty((N,), dtype=th.long, device=device)
                        logps_h = th.empty((N,), dtype=th.float32, device=device)
                        logps_l = th.empty((N,), dtype=th.float32, device=device)
                        v_hats = th.empty((N,), dtype=th.float32, device=device)
                        v_bars = th.empty((N,), dtype=th.float32, device=device)
                        pi_hats = th.empty((N, params.num_options), dtype=th.float32, device=device)
                        # print(f"\n*****step {t}:")
                        for a in range(N):
                            #High actor network
                            x = th.cat([global_state, agent_ids[a]], dim=-1)  # shape [1, input_dim]
                            with th.no_grad():
                                logits, betas = actor_h(x)
                            pi_hat = compute_pi_hat(logits, betas, prev_options[a])
                            option_dist = Categorical(probs=pi_hat)
                            option = option_dist.sample()
                            lp_h = option_dist.log_prob(option)


                            #Low actor network
                            option_onehot = F.one_hot(option, num_classes=params.num_options).to(device)
                            x = th.cat([global_state, agent_ids[a], option_onehot], dim=-1)  # shape [1, input_dim]
                            with th.no_grad():
                                logits = actor_l(x)
                            #mask invalid actions
                            Z = -1e8  # Large negative number
                            invalid_mask = th.zeros_like(logits, dtype=th.bool)
                            if option != params.num_options - 1: #Tramsit
                                invalid_mask[0, params.action_dim - 1] = True
                            else:   #NO Transmit
                                invalid_mask[0, :params.action_dim - 1] = True
                            logits.masked_fill_(invalid_mask, Z)
                            action_dist = Categorical(logits=logits)
                            action = action_dist.sample()
                            lp_l = action_dist.log_prob(action)

                            # print(f"agent {a}: option: {option}. action: {action} ")

                            #Compute High and low value functions
                            v_hat = (option_values * pi_hat).sum(-1).unsqueeze(-1)  # weighted sum of q for each option
                            v_bar = option_values[:, option] # q value for current option

                            # write into pre‐allocated tensors
                            options[a] = option
                            actions[a] = action
                            logps_h[a] = lp_h
                            logps_l[a] = lp_l
                            v_hats[a] = v_hat
                            v_bars[a] = v_bar
                            pi_hats[a] = pi_hat.squeeze(0)  # [num_options]

                        """Convert action and step environment"""
                        integer_action = [env.RRA_idx2int[(o.item(), a.item())] for o, a in zip(options, actions)]
                        # print(f"joint_action: {integer_action}")
                        global_next_state, _, _, global_reward, _, done = env.step(integer_action, k, t)

                        """Add transition to buffer"""
                        transition = (
                            global_state,                   # tensor shape [1, state_dim]
                            actions,         # tensor shape [N]
                            options,  # tensor shape [N]
                            global_reward,                  # float
                            prev_options.clone(),    # tensor shape [N]
                            pi_hats,       # tensor shape [N, num_options]
                            v_hats,          # tensor shape [N]
                            v_bars,          # tensor shape [N]
                            logps_h,          # tensor shape [N]
                            logps_l,          # tensor shape [N]
                        )
                        buffer.push(*transition)

                        total_reward += global_reward
                        global_state = th.tensor(global_next_state, dtype=th.float32, device=device)
                        # global_state = th.from_numpy(global_next_state).to(device)
                        prev_options[:] = options

                        if done:
                            # print(f"****episode {n_ep} complete. Reward = {total_reward}****")
                            break
                n_ep += 1
                train_returns.append(total_reward)

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return = self.test(actor_h, actor_l, params, env)
                    test_returns.append(test_return)
                    print(
                        f'Trial {trial + 1}. Test return at episode {n_ep}: {test_return:.3f}. Average train return: {np.mean(train_returns[-params.test_interval:]):.3f}')

            # process buffer once full
            dataloader = buffer.process_batch()

            """Begin optmization"""
            loss_p, loss_c = [], []
            for epoch in range(params.opt_epochs):
                for minibatch in dataloader:
                    # unpack mini batch
                    (s_mb, a_mb, o_mb, prev_o_mb, pi_hat_mb, v_h_mb, v_l_mb,
                     lp_h_mb, lp_l_mb, rtrn_mb, adv_h_mb, adv_l_mb) = minibatch

                    MB, _ = a_mb.shape

                    opt.zero_grad()
                    option_values = critic(s_mb)
                    loss_ah, loss_al, loss_ch, loss_cl = 0,0,0,0
                    for a in range(N):
                        agent_id = agent_ids[a].repeat(MB, 1)
                        """"HIGH MDP ACTOR"""
                        old_logp = lp_h_mb[:, a]
                        advantages = adv_h_mb[:, a]
                        options = o_mb[:, a]
                        prev_options = prev_o_mb[:, a]

                        x = th.cat([s_mb, agent_id], dim=-1)
                        logits, betas = actor_h(x)

                        new_pi_hats = compute_pi_hat(logits, betas, prev_options)
                        option_dist = Categorical(probs=new_pi_hats)
                        new_logp = option_dist.log_prob(options)
                        entropy = option_dist.entropy().mean()

                        loss = actor_h.actor_loss(new_logp, old_logp, advantages)
                        loss_ah += loss - entropy * params.entropy_coef_h

                        """"LOW MDP ACTOR"""
                        old_logp = lp_l_mb[:, a]
                        advantages = adv_l_mb[:, a]
                        actions = a_mb[:, a]
                        options_onehot = F.one_hot(options.squeeze(-1), num_classes=params.num_options).to(device)

                        x = th.cat([s_mb, agent_id, options_onehot], dim=-1)
                        logits = actor_l(x)
                        action_dist = Categorical(logits=logits)
                        new_logp = action_dist.log_prob(actions)
                        entropy = action_dist.entropy().mean()

                        loss = actor_h.actor_loss(new_logp, old_logp, advantages)
                        loss_al += loss - entropy * params.entropy_coef_l

                        """HIGH MDP CRITIC"""
                        old_values = v_h_mb[:, a]
                        new_values = option_values[:, options]
                        loss = critic.critic_loss(new_values, old_values, rtrn_mb)
                        loss_ch += loss

                        """LOW MDP CRITIC"""
                        old_values = v_l_mb[:, a]
                        new_values = (option_values * pi_hat_mb[:, a, :]).sum(-1).unsqueeze(-1)
                        loss = critic.critic_loss(new_values, old_values, rtrn_mb)
                        loss_cl += loss

                    total_loss = (loss_ah + loss_al + loss_ch + loss_cl) / N
                    total_loss.backward()
                    opt.step()

    @staticmethod
    def test(actor_h, actor_l, params, env):
        device = params.device
        env.testing_mode = True

        test_returns = np.zeros(params.test_episodes)
        N = params.num_agents
        agent_ids = th.eye(N, device=device).unsqueeze(1)  # shape [N, 1, N]


        for i in range(params.test_episodes):
            total_rewards = 0

            global_state, _, _ = env.reset(test_idx=i)
            # global_state, _, _ = env.reset()
            # prev_options = [None for _ in range(params.num_agents)]
            prev_options = th.full((N,), -1, dtype=th.long, device=device)  # shape [N], all –1

            for k in range(1, params.k_max + 1):
                if k > 1:
                    env.reset_control(k)

                for t in range(params.t_max):

                    # actions, options = [], []
                    options = th.empty((N,), dtype=th.long, device=device)
                    actions = th.empty((N,), dtype=th.long, device=device)
                    global_state = th.tensor(global_state, dtype=th.float32, device=device)

                    for a in range(N):
                        x = th.cat([global_state, agent_ids[a]], dim=-1)
                        with th.no_grad():
                            logits, betas = actor_h(x)
                        pi_hat = compute_pi_hat(logits, betas, prev_options[a])
                        option_dist = Categorical(probs=pi_hat)
                        option = option_dist.sample()

                        # Low actor network
                        option_onehot = F.one_hot(option, num_classes=params.num_options).to(device)
                        x = th.cat([global_state, agent_ids[a], option_onehot], dim=-1)  # shape [1, input_dim]
                        with th.no_grad():
                            logits = actor_l(x)
                        # mask invalid actions
                        Z = -1e8  # Large negative number
                        invalid_mask = th.zeros_like(logits, dtype=th.bool)
                        if option != params.num_options - 1:  # Tramsit
                            invalid_mask[0, params.action_dim - 1] = True
                        else:  # NO Transmit
                            invalid_mask[0, :params.action_dim - 1] = True
                        logits.masked_fill_(invalid_mask, Z)
                        action_dist = Categorical(logits=logits)
                        action = action_dist.sample()

                        options[a] = option
                        actions[a] = action

                    integer_action = [env.RRA_idx2int[(o.item(), a.item())] for o, a in zip(options, actions)]
                    global_next_state, _, _, global_reward, _, done = env.step(integer_action, k, t)

                    global_state = global_next_state
                    prev_options[:] = options
                    total_rewards += global_reward

                    if done:
                        break

            test_returns[i] += total_rewards

        average_return = np.mean(test_returns)
        env.testing_mode = False
        return average_return


class zDACtrainer_FO:
    def __init__(self):
        self.actor_opt = None
        self.critic_opt = None

    def train(self, trial, env, params):
        device = params.device

        """Initialize networks, optimizers, buffer"""
        actors = DAC_Actors(params).to(device)
        critic = DAC_Critic(params).to(device)

        # network = DAC_Network(params).to(device)
        # for name, param in network.named_parameters():
        #     print(name, param.shape)
        # sys.exit()
        self.actor_opt = th.optim.Adam([
            {'params': [p for n, p in actors.named_parameters() if 'pi_' in n], 'lr': params.lr_la},    #sub policy
            {'params': [p for n, p in actors.named_parameters() if 'actor' in n], 'lr': params.lr_ha},     #master policy
            {'params': [p for n, p in actors.named_parameters() if 'beta' in n], 'lr': params.lr_beta},
        ])
        self.critic_opt = th.optim.Adam(critic.parameters(), lr=params.lr_critic)
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
                # global_state = th.from_numpy(global_state).to(device)

                """Set initial previous options - dummy option"""
                # prev_options = [None for _ in range(N)]
                prev_options = th.full((N,), -1, dtype=th.long, device=device)  # shape [N], all –1

                # loop for k_max control intervals (1 for without AoI)
                for k in range(1, params.k_max + 1):
                    if k > 1:
                        env.reset_control(k)

                    # loop for timesteps in control interval
                    for t in range(params.t_max):
                        """get option-values"""
                        with th.no_grad():
                            q_W = critic(global_state)  # [1, num_options]


                            """Loop for each agent, get options, actions, logps"""
                            # 1) Allocate once (on GPU) for this time‐step:
                            options = th.empty((N,), dtype=th.long, device=device)
                            actions = th.empty((N,), dtype=th.long, device=device)
                            logps_h = th.empty((N,), dtype=th.float32, device=device)
                            logps_l = th.empty((N,), dtype=th.float32, device=device)
                            v_hats = th.empty((N,), dtype=th.float32, device=device)
                            v_bars = th.empty((N,), dtype=th.float32, device=device)
                            pi_hats = th.empty((N, params.num_options), dtype=th.float32, device=device)

                            # options, actions, logps_h, logps_l = [], [], [], []
                            # options, actions, prev_options, logps_h, logps_l = [], [], [], [], []
                            # pi_hats, v_hats, v_bars = [], [], []
                            for a in range(N):
                                # forward pass through network
                                # agent_id = F.one_hot(th.tensor([a]), num_classes=N).float().to(device)  # shape [1, num_agents]
                                x = th.cat([global_state, agent_ids[a]], dim=-1)  # shape [1, input_dim]
                                # with th.no_grad():
                                prediction = actors(x)

                                # compute high MDP policy, logp, and sample option
                                pi_hat = compute_pi_hat(prediction, prev_options[a])
                                dist = Categorical(probs=pi_hat)
                                option = dist.sample()
                                logp_h = dist.log_prob(option)

                                # compute low MDP policy for current option, logp, and sample action
                                pi_bar = prediction['pi_w'][0, option, :]
                                dist = Categorical(probs=pi_bar)
                                action = dist.sample()
                                logp_l = dist.log_prob(action)

                                # compute high and low MDP value functions
                                v_bar = q_W[:, option]  # q value for current option
                                v_hat = (q_W * pi_hat).sum(-1).unsqueeze(-1)  # weighted sum of q for each option

                                # write into pre‐allocated tensors
                                options[a] = option
                                actions[a] = action
                                logps_h[a] = logp_h
                                logps_l[a] = logp_l
                                v_hats[a] = v_hat
                                v_bars[a] = v_bar
                                pi_hats[a] = pi_hat.squeeze(0)  # [num_options]

                                #Add agent variables to lists
                                # options.append(option.view(-1))      # shape [1]
                                # # prev_options.append(prev_option[a] if prev_option[a] is not None else th.LongTensor([0]))
                                # actions.append(action.view(-1))     # shape [1]
                                # logps_h.append(logp_h.view(-1))     # shape [1]
                                # logps_l.append(logp_l.view(-1))     # shape [1]
                                # v_hats.append(v_hat.view(-1))     # shape [1]
                                # v_bars.append(v_bar.view(-1))     # shape [1]
                                # pi_hats.append(pi_hat.view(-1))     # shape [num_options]

                        """Convert action and step environment"""
                        # integer_action = get_flat_action(options, actions, env.action2RRA_reverse, env.V2V_power_dB_list)
                        integer_action = [env.RRA_idx2int[(o.item(), a.item())] for o, a in zip(options, actions)]
                        global_next_state, _, _, global_reward, _, done = env.step(integer_action, k, t)


                        """Add transition to buffer (override dummy options)"""
                        # prev_options = [o if o is not None else th.zeros(1, dtype=th.int64, device=device)
                        #                 for o in prev_options]
                        prev_options[:] = options
                        transition = (
                            global_state,                   # tensor shape [1, state_dim]
                            actions,         # tensor shape [N]
                            global_reward,                  # float
                            options,         # tensor shape [N]
                            prev_options.clone(),    # tensor shape [N]
                            pi_hats,       # tensor shape [N, num_options]
                            v_hats,          # tensor shape [N]
                            v_bars,          # tensor shape [N]
                            logps_h,          # tensor shape [N]
                            logps_l,          # tensor shape [N]
                        )
                        # transition = (
                        #     global_state,                   # tensor shape [1, state_dim]
                        #     th.cat(actions, dim=0),         # tensor shape [N]
                        #     global_reward,                  # float
                        #     th.cat(options, dim=0),         # tensor shape [N]
                        #     th.cat(prev_options, dim=0),    # tensor shape [N]
                        #     th.stack(pi_hats, dim=0),       # tensor shape [N, num_options]
                        #     th.cat(v_hats, dim=0),          # tensor shape [N]
                        #     th.cat(v_bars, dim=0),          # tensor shape [N]
                        #     th.cat(logps_h, dim=0),          # tensor shape [N]
                        #     th.cat(logps_l, dim=0),          # tensor shape [N]
                        # )

                        # print(f'\n****Episode {n_ep + 1}, t = {t + 1}')
                        # print(f"actions = {th.cat(actions, dim=0)}\n"
                        #       f"options = {th.cat(options, dim=0)}\n"
                        #       f"integer_action = {integer_action}\n"
                        #       f"prev_options = {th.cat(prev_options, dim=0)}\n"
                        #       f"betas = {prediction['betas'].squeeze()}\n"
                        #       f"pi_hats = {th.stack(pi_hats, dim=0)}"
                        #       )

                        # names = ["global_state", "actions", "global_reward","options",
                        #     "prev_options","pi_hats","v_hats","v_bars","logps_h","logps_l"]
                        #
                        # print(f'\nEpisode {n_ep+1}, t = {t+1}')
                        # for name, item in zip(names, transition):
                        #     if isinstance(item, th.Tensor):
                        #         print(f"{name}.shape = {tuple(item.shape)}")
                        #     else:
                        #         print(f"{name} = {item}")


                        buffer.push(*transition)

                        total_reward += global_reward
                        global_state = th.tensor(global_next_state, dtype=th.float32, device=device)
                        # global_state = th.from_numpy(global_next_state).to(device)
                        prev_options = options

                        if done:
                            break

                # buffer.process_episode()
                n_ep += 1
                train_returns.append(total_reward)

                # test at interval and print result
                if n_ep % params.test_interval == 0:
                    test_return = self.test(actors, params, env)
                    test_returns.append(test_return)
                    print(f'Trial {trial+1}. Test return at episode {n_ep}: {test_return:.3f}. Average train return: {np.mean(train_returns[-params.test_interval:]):.3f}')

            # process buffer once full
            dataloader = buffer.process_batch()

            #initiate learning
            mdps = ['hat', 'bar']
            # np.random.shuffle(mdps)
            self.learn(actors, critic, dataloader, params, mdps[1])
            self.learn(actors, critic, dataloader, params, mdps[0])

        return train_returns, test_returns

    def learn(self, actors, critic, dataloader, params, mdp):
        device = params.device
        loss_p, loss_c = [], []
        for epoch in range(params.opt_epochs):
            for minibatch in dataloader:
                # unpack mini batch
                (s_mb, a_mb, o_mb, prev_o_mb, pi_h_mb, v_h_mb, v_l_mb,
                 lp_h_mb, lp_l_mb, rtrn_mb, adv_h_mb, adv_l_mb) = minibatch

                MB, N = a_mb.shape

                #Actor loss
                self.actor_opt.zero_grad()
                total_actor_loss = 0
                for a in range(N):
                    agent_id = F.one_hot(th.tensor(a), num_classes=N).float().unsqueeze(0).repeat(MB, 1).to(device)
                    x = th.cat([s_mb, agent_id], dim=-1)
                    prediction = actors(x)

                    if mdp == 'bar':
                        old_logp = lp_l_mb[:, a]
                        advantages = adv_l_mb[:, a]
                        actions = a_mb[:, a]
                        new_policy = prediction['pi_w'][:, o_mb[:,a], :]
                        tau = params.entropy_coef_l
                    elif mdp == 'hat':
                        old_logp = lp_h_mb[:, a]
                        advantages = adv_h_mb[:, a]
                        actions = o_mb[:, a]
                        new_policy = compute_pi_hat(prediction, prev_o_mb[:, a])
                        tau = params.entropy_coef_h
                    else:
                        raise NotImplementedError

                    dist = Categorical(probs=new_policy)
                    new_logp = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    actor_loss = actors.actor_loss(new_logp, old_logp, advantages)
                    total_actor_loss += actor_loss - tau * entropy
                total_actor_loss /= params.num_agents
                total_actor_loss.backward()
                self.actor_opt.step()

                #critic loss
                self.critic_opt.zero_grad()
                new_q_W = critic(s_mb)
                total_critic_loss = 0
                for a in range(N):
                    if mdp == 'bar':
                        old_values = v_l_mb[:, a]
                        new_values = new_q_W[:, o_mb[:,a]]
                    elif mdp == 'hat':
                        old_values = v_h_mb[:, a]
                        new_values = (new_q_W * pi_h_mb[:,a,:]).sum(-1).unsqueeze(-1)
                    else:
                        raise NotImplementedError
                    critic_loss = critic.critic_loss(new_values, old_values, rtrn_mb)
                    total_critic_loss += critic_loss
                total_critic_loss /= params.num_agents
                total_critic_loss.backward()
                self.critic_opt.step()


                loss_p.append(total_actor_loss)
                loss_c.append(total_critic_loss)

        # Optional print average optimization losses
        # av_loss_p, av_loss_c = sum(loss_p) / len(loss_p), sum(loss_c) / len(loss_c)
        # print(f"MDP-{mdp}, optimization complete avg losses: Policy loss: {av_loss_p:.3f} | Critic loss: {av_loss_c:.3f}")


    @staticmethod
    def test(actors, params, env):
        device = params.device
        env.testing_mode = True

        test_returns = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            total_rewards = 0

            global_state, _, _ = env.reset(test_idx=i)
            # global_state, _, _ = env.reset()
            # prev_options = [None for _ in range(params.num_agents)]
            prev_options = th.full((params.num_agents,), -1, dtype=th.long, device=device)  # shape [N], all –1

            for k in range(1, params.k_max + 1):
                if k > 1:
                    env.reset_control(k)

                for t in range(params.t_max):

                    # actions, options = [], []
                    options = th.empty((params.num_agents,), dtype=th.long, device=device)
                    actions = th.empty((params.num_agents,), dtype=th.long, device=device)
                    global_state = th.tensor(global_state, dtype=th.float32, device=device)

                    for a in range(params.num_agents):
                        agent_id = F.one_hot(th.tensor([a]), num_classes=params.num_agents).float().to(device)
                        x = th.cat([global_state, agent_id], dim=-1)
                        with th.no_grad():
                            prediction = actors(x)

                        pi_hat = compute_pi_hat(prediction, prev_options[a])
                        dist = Categorical(probs=pi_hat)
                        option = dist.sample()

                        pi_bar = prediction['pi_w'][0, option, :]
                        dist = Categorical(probs=pi_bar)
                        action = dist.sample()

                        options[a] = option
                        actions[a] = action
                        # options.append(option.view(-1))
                        # actions.append(action.view(-1))

                    integer_action = [env.RRA_idx2int[(o.item(), a.item())] for o, a in zip(options, actions)]
                    global_next_state, _, _, global_reward, _, done = env.step(integer_action, k, t)

                    global_state = global_next_state
                    prev_options[:] = options
                    total_rewards += global_reward

                    if done:
                        break

            test_returns[i] += total_rewards

        average_return = np.mean(test_returns)
        env.testing_mode = False
        return average_return

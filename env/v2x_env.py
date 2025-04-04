import math, random, sys
import numpy as np
import matplotlib.pyplot as plt

from env.env_helper import sample_veh_positions, remove_test_data_from_veh_pos
from env.render import HighwayVisualizer




class Vehicle:
    """Vehicle class to encode positional data for V2V and V2I cars """
    def __init__(self, veh_id, start_position, velocity):
        self.id = veh_id
        self.position = start_position
        self.velocity = velocity
        self.destinations = []
        self.neighbors = []


class V2XEnvironment:

    def __init__(self, env_setup, veh_pos_data):
        """configuration hyperparameters"""
        self.game_mode = env_setup['game_mode']
        self.k_max = env_setup['k_max']
        self.t_max = env_setup['t_max']
        self.num_agents = env_setup['num_agents']
        self.fast_fading = env_setup['fast_fading']
        self.multi_location = env_setup['multi_location']
        self.single_loc_idx = env_setup['single_loc_idx']
        self.multi_loc_test_idx = env_setup['multi_loc_test_idx']
        self.partial_observability = env_setup['partial_observability']

        self.render_control_interval = True

        if self.render_control_interval:
            self.visualizer = HighwayVisualizer()

        """local env parameters"""
        self.n_neighbor = 1  # number of neighboring vehicles need to receive CAM
        #Radio transmission related
        self.fc = 2                     # carrier frequency [GHz]
        self.num_SC = 4                 # number of sub-channels and V2I links
        self.V2V_power_dB_list = [23, 15, 5]  # power levels at the PM vehicle side
        self.V2I_power_dB = 23          # dBm
        self.sig2 = 10 ** (-114 / 10)   #-114dB power level of AWGN noise
        self.h_bs = 1.5                 # height of base station antenna [m]
        self.h_ms = 1.5                 # height of mobile station (vehicle) antenna [m]
        self.eNB_xy = [500, -43]        # position of base station
        self.veh_ant_gain = 3           # Vehicle antenna gain
        self.veh_noise_figure = 9       # Vehicle noise figure
        self.bs_ant_gain = 8            # Base station antenna gain
        self.bs_noise_figure = 5        # Base station noise figure
        self.norm_V2V_channel_factor = 120  # Normalization factor for V2V channel

        #Reward weights
        self.lambda1 = 0.001  # reward weight corresponding to TODO V2I ??? used for V2V???
        self.lambda2 = 0.1  # reward weight corresponding to V2V TODO used * V2V_SE
        self.lambda3 = 1  # reward weight corresponding to AoI
        self.reward_G = 5.0  # reward weight corresponding to V2V data rate

        # Queue related
        self.NQ = 1  # number of buffer size at each agent
        self.time_fast = 0.001  #queue parameter
        self.bandwidth_per_SC = int(1000000)  # bandwidth of each sub-channel in Hz
        self.CAM_size = 25600  # number of bits per cooperative awareness message (''CAM), in bits'


        """dynamic parameters"""
        self.testing_mode = False
        self.current_veh_positions = None   #single positional timestep
        self.episode_veh_positions = None   #single or multi positional timestep
        self.active_links = None
        self.queue = None
        self.AoI = None
        #below are kept for debugging/rendering
        self.individual_rewards = None  #dim = [num_agents]
        self.pathloss_V2I_V2V = None    #dim = [num_veh_v2i, num_veh_v2v] = [num_SC, num_agents*n_neighbor]
        self.pathloss_V2I_eNB = None    #dim = [num_veh_v2i]
        self.pathloss_V2V_eNB = None       #dim = [num_veh_v2v]
        self.pathloss_V2V_V2V = None    #dim = [num_veh_v2v, num_veh_v2v]
        self.distance_V2I_V2V = None    #dim = [num_veh_v2i, num_veh_v2v]
        self.distance_V2V_V2V = None    #dim = [num_veh_v2v, num_veh_v2v]
        self.SC_RSRP_dB = None      #Received Signal Recieved Power
        self.interference_V2V_tot = None
        self.interference_V2V_Rx = None
        self.interference_V2V_other = None
        self.interference_V2I_Rx = None
        self.signal_V2V = None
        self.V2V_SE = None      #Spectral efficiency


        """calculated parameters"""
        self.platoon_V2V = [2 for _ in range(self.num_agents)]
        self.num_veh_V2V = sum(self.platoon_V2V)
        self.num_veh_V2I = self.num_SC #num V2I=num sub-channel
        self.num_power_levels = len(self.V2V_power_dB_list)
        self.action_dim = self.num_power_levels * self.num_SC + 1

        #---Define null_ID: set of V2V cars that are not agents (receiver)---
        """Assumes 1 non-transmitting vehicle per platoon
            Typical format: set(0, 2, 4, 6) """
        self.null_ID = set()
        non_ag_offset = 0
        for p in range(len(self.platoon_V2V)):
            n_veh = self.platoon_V2V[p] #num vehicles in platoon p
            for v in range(n_veh):  #loop for vehicles in p
                if v == 0:  #first vehicle is non-transmitting
                    self.null_ID.add(v + non_ag_offset) #add global index to null_ID
            non_ag_offset += n_veh

        #---Agent->Vehicle ID mapping dictionary ---
        """Dictionary format (length num_agents):
            keys: agent index (i.e. 0, 1, 2, 3)
            values: global vehicle index (i.e. 1, 3, 5, 7) """
        self.agent2veh = {}
        agent_index = 0
        for veh_idx in range(self.num_veh_V2V):
            if veh_idx not in self.null_ID:
                self.agent2veh[agent_index] = veh_idx
                agent_index += 1
            else: continue

        #---compute state dimension---
        dim_t = self.t_max
        dim_G_ji = self.num_agents * (self.num_agents - 1)
        dim_G_m = self.num_SC
        dim_G_Bi = self.num_SC * self.num_agents
        dim_G_iB = dim_q = dim_AoI = dim_G_i = self.num_agents

        if self.game_mode == 1: #State = [G_i, G_ji, G_m, G_Bi, G_iB]
            self.state_dim = dim_G_i + dim_G_ji + dim_G_m + dim_G_Bi + dim_G_iB
        elif self.game_mode in [2, 4]: #State = [t, G_i, G_ji, G_m, G_Bi, G_iB, q]
            self.state_dim = dim_t + dim_G_i + dim_G_ji + dim_G_m + dim_G_Bi + dim_G_iB + dim_q
        elif self.game_mode in [3, 5]: #State = [t, G_i, G_ji, G_m, G_Bi, G_iB, q, AoI]
            self.state_dim = dim_t + dim_G_i + dim_G_ji + dim_G_m + dim_G_Bi + dim_G_Bi + dim_q + dim_AoI
        else: raise ValueError

        #  and observation (local state) dimension---
        if self.game_mode == 4:
            self.obs_dim = dim_t + (dim_G_i + dim_G_ji + dim_q) // self.num_agents
        elif self.game_mode == 5:
            self.obs_dim = dim_t + (dim_G_i + dim_G_ji + dim_q + dim_AoI) // self.num_agents
        else:
            self.obs_dim = None



        """extract and format train/test data samples"""
        if not self.multi_location:  # NFIG & SIG-sloc only
            self.veh_pos_data, _ = sample_veh_positions(veh_pos_data, t0=self.single_loc_idx)
            self.test_data_list = self.veh_pos_data
        else:  # SIG-mloc & POSIG
            self.test_data_list = [sample_veh_positions(veh_pos_data, t0=step, k_max=self.k_max)[0] for step in
                                   self.multi_loc_test_idx]
            self.veh_pos_data = remove_test_data_from_veh_pos(veh_pos_data, self.multi_loc_test_idx)


        """Initialize vehicles once by creating vehicle objects and computing neighbors/destinations."""
        self.vehicles_V2V = []
        self.vehicles_V2I = []
        sampled_veh_pos, _ = sample_veh_positions(veh_pos_data, t0=self.single_loc_idx)  # use sloc sample for vehicle init
        self.initialize_vehicles(sampled_veh_pos)


    def initialize_vehicles(self, sampled_data):
        """Initialize vehicles given position data sample. Called in
            __init__ to create all vehicle objects w/ static params."""

        # Store positional data (for sloc this is the only time we store data, for mloc this is overwritten)
        self.current_veh_positions = self.episode_veh_positions = sampled_data

        def add_vehicle(veh_id, position, velocity, type):
            """vehicle adder function"""
            if type == "V2V":
                self.vehicles_V2V.append(Vehicle(veh_id, position, velocity))
            elif type == "V2I":
                self.vehicles_V2I.append(Vehicle(veh_id, position, velocity))

        # Create vehicles from DataFrame
        for _, row in sampled_data.iterrows():
            veh_id = row['id']
            position = [row['x'], row['y']]
            velocity = row['speed']
            if veh_id.startswith("carflowV2I"):
                add_vehicle(veh_id, position, velocity, "V2I")
            elif veh_id.startswith("carflow"):
                add_vehicle(veh_id, position, velocity, "V2V")



        # Compute neighbors and destinations for V2V vehicles TODO Check!!
        for i in range(self.num_agents):
            veh_index = self.agent2veh[i]
            v2v_vehicle = self.vehicles_V2V[veh_index]
            # For example: assign the previous vehicle as the neighbor (static assignment)
            v2v_vehicle.neighbors = [veh_index - 1] * self.n_neighbor
            v2v_vehicle.destinations = v2v_vehicle.neighbors.copy()

            veh = self.agent2veh[i]
            v2v_vehicle = self.vehicles_V2V[veh]
            v2v_vehicle.neighbors = [veh - 1] * self.n_neighbor  # Assuming `veh - 1` is valid
            v2v_vehicle.destinations = v2v_vehicle.neighbors  # Reusing the list


    def update_vehicle_positions(self, sampled_veh_pos_data):
        """
        Updates the positions of V2V and V2I vehicles based on sampled CSV data.

        Parameters:
        - sampled_veh_pos_data: DataFrame containing columns ['id', 'x', 'y', 'speed']
        - plot_enabled (bool): If True, visualize the updated highway.
        """
        for _, row in sampled_veh_pos_data.iterrows():
            veh_id = row['id']
            new_position = [row['x'], row['y']]
            new_velocity = row['speed']

            if veh_id.startswith("carflowV2I"):
                for veh in self.vehicles_V2I:
                    if veh.id == veh_id:
                        veh.position = new_position
                        veh.velocity = new_velocity
                        break  # Avoid unnecessary iterations
            elif veh_id.startswith("carflow"):
                for veh in self.vehicles_V2V:
                    if veh.id == veh_id:
                        veh.position = new_position
                        veh.velocity = new_velocity
                        break




    # def update_vehicle_positions(self, sampled_veh_pos_data):
    #     """Update positions for already initialized vehicles using new positional data."""
    #     v2v_idx, v2i_idx = 0, 0 #TODO Optimize?
    #     for _, row in sampled_veh_pos_data.iterrows():
    #         veh_id = row['id']
    #         new_position = [row['x'], row['y']]
    #         new_velocity = row['speed']
    #         if veh_id.startswith("carflowV2I"):
    #             self.vehicles_V2I[v2i_idx].position = new_position
    #             self.vehicles_V2I[v2i_idx].velocity = new_velocity
    #             v2i_idx += 1
    #         elif veh_id.startswith("carflow"):
    #             self.vehicles_V2V[v2v_idx].position = new_position
    #             self.vehicles_V2V[v2v_idx].velocity = new_velocity
    #             v2v_idx += 1

    def reset(self):
        """Create a new random game by reloading vehicle positions, reinitializing channels,
            queues, active links, and AoI values.
            Re-sample vehicle data based on the game mode:
                - For game_mode 1, 2, 4: No AoI; a single control interval is used (1 positional data sample).
                - For game_mode 3, 5: AoI; k_max control intervals (positional sample)
            Previous name was 'new_random_game' """

        """Sample and store new vehicle positions (mloc only)"""
        if self.multi_location is True:     #Multi-Location
            #SAMPLE
            if self.testing_mode is False:  # Training mode
                sampled_data, t0 = sample_veh_positions(self.veh_pos_data, k_max=self.k_max)
            else:                           #Testing Mode
                t0 = np.random.choice(len(self.test_data_list))
                sampled_data = self.test_data_list[t0]

            #STORE episode and current positional data samples
            self.episode_veh_positions = sampled_data
            first_timestep = sampled_data['time'].unique()[0]  # Get the first unique time index
            self.current_veh_positions = sampled_data[sampled_data['time'] == first_timestep]

            #UPDATE vehicle positions
            self.update_vehicle_positions(self.current_veh_positions)
        else: t0 = self.single_loc_idx

        """Update channels and other environment parameters"""
        #update channels
        self.renew_channel()

        # Reset environment parameters
        self.active_links = np.ones((self.num_agents, self.n_neighbor), dtype='bool')  # TODO:??
        if self.game_mode != 1:
            self.queue = np.ones((self.num_agents, self.n_neighbor))
        if self.game_mode in [3,5]:
            self.AoI = np.ones((self.num_agents, self.n_neighbor))

        global_state = self.get_state(idx=(0, 0), i_step=0)


        """Initialize rendering of positional data and pathloss info"""
        if self.render_control_interval:
            # get stored data
            position_info = {
                'pl_v2i_v2v': self.pathloss_V2I_V2V,    #[m, i*2]
                'pl_v2i_bs': self.pathloss_V2I_eNB,     #[m]
                'pl_v2v_bs': self.pathloss_V2V_eNB,     #[i*2]
                'pl_v2v_v2v': self.pathloss_V2V_V2V,    #[i*2, i*2]
                'd_v2i_v2v': self.distance_V2I_V2V,  # [m, i*2]
                'd_v2v_v2v': self.distance_V2V_V2V,  # [i*2, i*2]
            }
            self.visualizer.draw_frame(self.current_veh_positions, position_info, t0, self.testing_mode)




        return global_state


    def step(self, actions, t, k):
        """Executes one communication interval step in the environment.
            Args:
                actions: array for all agent RRA action [num_agents, action_dim]
                t (int): Current time step (communication interval) in the control interval (episode)
                k (int): Index of control interval (will always be 1 unless we include AoI i.e. game_mode in [3,5]).
            Returns:
                global_next_state:
                global_reward
                done

                individual_rewards (removed)
                V2I_SE (removed)
                done        """

        # --- Compute Performance Metrics ---
        V2V_SE, V2I_SE, queue, reward_V2V = self.compute_reward(actions)

        # --- Update Environment State ---
        self.compute_SC_received_power(actions)
        self.active_links[:, 0] = queue[:, 0] > 0

        # --- Check Episode Completion ---
        done = (t == self.t_max - 1) and (k == self.k_max)

        # --- Reward Calculation Based on Game Mode --- #TODO I added V2I_SE here since it was in the training loops
        if self.game_mode == 1: # Reward: Sum of all V2V+V2I spectral efficiencies
            global_reward = np.array([[np.sum(V2V_SE) + np.sum(V2I_SE)]])
            individual_rewards = []

        elif self.game_mode in [2, 4]: # Reward: Sum of V2V SEs +  bonus for empty queues
            individual_rewards = np.where(queue <= 0, self.reward_G, V2V_SE * 0.01)
            global_reward = np.array([[np.sum(individual_rewards)]])

        elif self.game_mode in [3, 5]: # Reward: V2V SE weighted by lambda2 - AoI weighted by lambda3
            individual_rewards = -self.AoI * self.lambda3 + V2V_SE * self.lambda2
            global_reward = np.array([[np.sum(individual_rewards)]])

        else:
            raise ValueError('Invalid gamemode')

        #---- Get next state ----
        global_next_state = self.get_state(idx=(0, 0), i_step=t)
        if self.render_control_interval:
            pass
            # self.visualizer.update_text(f"")

        return global_next_state, global_reward, done

        # if self.game_mode in [4, 5]: #POSIG
        #     global_state, local_state = self.get_state(idx=(0, 0), i_step=t)
        #     observation =
        #     return global_state, local_state, global_reward, done
        # else:
        #     global_state = self.get_state(idx=(0, 0), i_step=t)
        #     return global_state, global_reward, done




        # return global_reward, done  #, individual_rewards, V2I_SE

    def step_control(self, interval):
        """Used only for gamemodes 3,4 with multiple control intervals. At the
            start of non-initial control intervals: update positions,channels,queues, AoI,active links
            previous name 'renew_positions' """
        # Update current positions based on the episode data at the given interval
        episode_data = self.episode_veh_positions
        new_timestep = episode_data['time'].unique()[interval - 1]  #get unique time index
        self.current_veh_positions = episode_data[episode_data['time'] == new_timestep]
        self.update_vehicle_positions(self.current_veh_positions)
        self.renew_channel()

        # Reset and update queues and AoI as per control episode requirements.
        if self.game_mode in [2, 3, 4]:
            self.queue = np.ones((self.num_agents, self.n_neighbor))
        if self.game_mode in [3, 4]:
            self.AoI = np.ones((self.num_agents, self.n_neighbor))

        # Integrated renew_queue functionality:
        self.AoI[self.queue > 0] += 1
        self.AoI[self.queue <= 0] = 1
        self.queue = np.ones((self.num_agents, self.n_neighbor))
        self.queue[:, 0] = np.minimum(self.queue[:, 0], self.NQ)

        # Integrated renew_active_links functionality:
        self.active_links[:, 0] = self.queue[:, 0] > 0



    def get_state(self, idx, i_step):
        """Generate the state representation based on the current game mode.
           Args:
               idx (tuple): Tuple of indices; typically, (agent_index, destination_index).
               i_step (int): Current step within a control interval. **Communication?
           Returns:
               np.ndarray: A state vector of shape (1, stateDim) or (1, local_stateDim) depending on the mode.
           """

        def norm_gain(pathloss):
            """Helper function to normalize channel gain between two positions."""
            return pathloss / self.norm_V2V_channel_factor


        """G_i: Common V2V channel gain from its V2V sender to its destination (all gamemodes)"""
        G_i = []    #dim= [num_agents]
        for i in range(self.num_agents):
            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
            G_i.append(norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver]))

        """G_ji: V2V interference channel gain for each pair of different agents (all gamemodes)"""
        G_ji = []   #dim= [num_agents * (num_agents - 1)]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                veh_sender = self.agent2veh[i]
                veh_receiver = self.vehicles_V2V[self.agent2veh[j]].destinations[idx[1]]
                G_ji.append(norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver]))

        """G_m: V2I channel gain for each subchannel (all gamemodes)"""
        #dim = [num_sub_channels]
        G_m = [norm_gain(self.pathloss_V2I_eNB[m]) for m in range(self.num_SC)]

        """G_Bi: Interference channel gain V2I (sender) to V2V (receiver) (all gamemodes)"""
        G_Bi = []  #dim = [num_sub_channels * num_agents]
        for i in range(self.num_agents):
            veh_receiver = self.agent2veh[i]
            for m in range(self.num_SC):
                veh_sender = m
                G_Bi.append(norm_gain(self.pathloss_V2I_V2V[veh_sender, veh_receiver]))

        """G_iB: Interference channel gain V2V (sender) to BaseStation (receiver) (all gamemodes)"""
        G_iB = []   #dim = [num_agents]
        for i in range(self.num_agents):
            veh_sender = self.agent2veh[i]
            G_iB.append(norm_gain(self.pathloss_V2V_eNB[veh_sender]))


        if self.game_mode != 1:
            """t:one-hot encoded communication interval time vector"""
            t = np.zeros(self.t_max)
            t[min(i_step, self.t_max - 1)] = 1

            """Queue: dim = [num_agents]""" #TODO: FIX THIS!?
            q_i = (self.queue / self.NQ).flatten()
        else:
            q_i, t = [], []
            # queue_length = self.queue[idx[0], idx[1]] / self.NQ #FROM get_observation

        """AgeofInformation: Not currently implemented."""
        if self.game_mode in [3, 5]:
            AoI = self.AoI.flatten() #dim = [num_agents]
        else:
            AoI = []
            # AoI = self.AoI[idx[0], idx[1]]  #FROM get_observation

        """G_i_po: V2V Channel gain for 1 agent in partial observability (mode 4)"""
        veh_sender = self.agent2veh[idx[0]]
        veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
        G_i_po = norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver])


        """Concatonnate state based on game mode"""#TODO POSIG we need both global and non global states
        if self.game_mode == 1: #NFIG
            global_state = np.hstack((G_i, G_ji, G_m, G_Bi, G_iB))
        elif self.game_mode == 2: #SIG_noAoI
            global_state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i))
        elif self.game_mode == 3: #SIG_AoI
            global_state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i, AoI))
        elif self.game_mode == 4:  #POSIG_noAoI
            global_state =  np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i))
            local_state = np.hstack((t, G_i_po, G_ji, G_m, G_Bi, G_iB, q_i))
        elif self.game_mode == 5:   #POSIG_AoI
            global_state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i, AoI))
            local_state = np.hstack((t, G_i_po, G_ji, G_m, G_Bi, G_iB, q_i, AoI))
        else: raise ValueError("Invalid game mode")

        global_state = global_state.reshape((1, self.state_dim))
        if self.game_mode in [4, 5]:
            local_state = local_state.reshape((1, self.local_state_dim))
            return global_state, local_state

        return global_state

        # print(f"t:{t.shape}\n"
        #       f"Gi: {len(G_i)}\n"
        #       f"Gij: {len(G_ji)}\n"
        #       f"Gm: {len(G_m)}\n"
        #       f"Gbi: {len(G_Bi)}\n"
        #       f"GiB: {len(G_iB)}\n"
        #       f"q_i: {len(q_i)}\n")



        return state

    def renew_channel(self):
        """Renew channels by recalculating all path loss values.
            calls methods compute_V2V_pathloss and compute_V2I_pathloss"""
        # ---- V2I → V2V interference channel ---- Create a matrix of shape (num_veh_V2I, num_veh_V2V)
        self.pathloss_V2I_V2V = np.zeros((self.num_veh_V2I, self.num_veh_V2V,))
        self.distance_V2I_V2V = np.zeros((self.num_veh_V2I, self.num_veh_V2V,))
        for m in range(self.num_veh_V2I):
            for i in range(self.num_veh_V2V):
                idx = (m, i)
                self.pathloss_V2I_V2V[idx], self.distance_V2I_V2V[idx] = self.compute_V2V_pathloss(self.vehicles_V2I[m].position, self.vehicles_V2V[i].position)
        #
        # self.pathloss_V2I_V2V = np.array([
        #     [self.compute_V2V_pathloss(v2v.position, v2i.position)[0]
        #      for v2v in self.vehicles_V2V]
        #     for v2i in self.vehicles_V2I ])

        # ---- V2I → eNB Channel ---- Create a matrix of shape (num_veh_V2I)
        self.pathloss_V2I_eNB = np.array([
            self.compute_V2I_pathloss(v2i.position) for v2i in self.vehicles_V2I])

        # ---- V2V → eNB Channel ---- Create a matrix of shape (num_veh_V2V)
        self.pathloss_V2V_eNB = np.array([
            self.compute_V2I_pathloss(v2v.position) for v2v in self.vehicles_V2V])

        # ---- V2V → V2V Channel ---- Create a matrix of shape (num_veh_V2V, num_veh_V2V)
        n = self.num_veh_V2V
        self.pathloss_V2V_V2V = np.zeros((n,n,))
        self.distance_V2V_V2V = np.zeros((n,n,))
        for i in range(n):
            for j in range(i+1, n):
                pl, d = self.compute_V2V_pathloss(self.vehicles_V2V[i].position, self.vehicles_V2V[j].position)
                self.pathloss_V2V_V2V[i, j] = self.pathloss_V2V_V2V[j, i] = pl
                self.distance_V2V_V2V[i, j] = self.distance_V2V_V2V[j, i] = d


    def mapping_action2RRA(self, action, use_simplified=None):
        """maps joint action indices to radio resource allocation (RRA)
            Converts action into:
                - Subchannel index
                - Power level index
            Returns:
                - SC_index (int): SC index (-1 if idle).
                - Power_level_index (int): The power level index (-1 if idle).
            """


        if use_simplified is None:  # Standard version
            if action < self.action_dim - 1:
                SC_idx = action // self.num_power_levels  # Floor division for SC index
                power_lvl_idx = action % self.num_power_levels
            else:
                SC_idx = -1
                power_lvl_idx = -1
        else:   #simplified version
            if action < self.action_dim - 1:
                SC_idx = self.ag_idx  # TODO Ensure `self.ag_idx` is correctly initialized
                power_lvl_idx = action % self.num_power_levels
            else:
                SC_idx = -1
                power_lvl_idx = -1

        return SC_idx, power_lvl_idx


    def compute_SC_received_power(self, actions):
        """Computes the received power (including noise and interference) at each subchannel for every agent.
            Parameters: actions_power: ndarray of shape (num_agents, n_neighbor, 2)
                        * actions_power[:, :, 0]: Selected subchannel indices (int)
                        * actions_power[:, :, 1]: Selected power level indices (int)
        """
        # Initialize received power matrix with the noise floor. Shape = [num_agents, num_SC]
        rsrp = np.full((self.num_agents, self.num_SC), self.sig2)

        # Extract channel selections and power selections from actions_power.
        channel_sel = actions[:, :, 0]    # (n_agent, n_neighbor)
        power_sel = actions[:, :, 1]      # (n_agent, n_neighbor)

        # Loop over each transmitting agent (V2V sender).
        for sender in range(self.num_agents):
            veh_sender = self.agent2veh[sender]

            #loop through V2V receiving agents (only 1)
            for sender_nbr in range(self.n_neighbor):

                #extract sub-channel and power level for V2V transmission
                selected_sc = channel_sel[sender, sender_nbr]
                sender_power_dB = self.V2V_power_dB_list[power_sel[sender, sender_nbr]]

                if selected_sc < 0:
                    continue  # Skip invalid channel selections

                #loop through all V2V receiving agents
                for receiver in range(self.num_agents):
                    veh_receiver = self.agent2veh[receiver]

                    if veh_sender == veh_receiver:
                        continue  # Skip self-interference cases

                    # loop through V2V receiving agents (only 1)
                    for receiver_nbr in range(self.n_neighbor):
                        receiver_destination = self.vehicles_V2V[veh_receiver].destinations[receiver_nbr]

                        if receiver_destination == veh_sender:
                            continue # Skip desired links (not interference)

                        pathloss_dB = self.pathloss_V2V_V2V[sender][receiver_destination]

                        # Compute interference power in linear scale and accumulate it
                        interference_power = 10 ** ((sender_power_dB
                                                     - pathloss_dB
                                                     + 2 * self.veh_ant_gain
                                                     - self.veh_noise_figure) / 10)

                        rsrp[receiver, selected_sc] += interference_power

        # Convert total received power from linear scale to dBm
        self.SC_RSRP_dB = 10 * np.log10(rsrp)

    def compute_reward(self, actions_power):
        """Compute communication rates (spectral efficiencies) for V2I and V2V links
            and derive the reward for V2V links. Rates are computed using the Shannon
            formula without scaling by bandwidth. """
        # Extract channel selections and power selections from actions_power.
        channel_sel = actions_power[:, :, 0]  # (n_agent, n_neighbor)
        power_sel = actions_power[:, :, 1]  # (n_agent, n_neighbor)

        """ Compute interference and date rate of V2I links """
        interference_V2V_V2I = np.zeros(self.num_SC)  # V2I interference
        for i in range(self.num_agents): #loop through V2V senders
            for j in range(self.n_neighbor):    #loop through receiving V2V cars (1)??
                selected_sc = int(channel_sel[i, j])    #do we need int()??????
                if selected_sc != -1:
                    #calculate/accumulate interference contribution from V2V sender i
                    sender_power_dB = self.V2V_power_dB_list[power_sel[i, j]]
                    interference_linear = 10 ** ((sender_power_dB - self.pathloss_V2V_eNB[i]
                                           + self.veh_ant_gain + self.bs_ant_gain
                                           - self.bs_noise_figure) / 10)
                    interference_V2V_V2I[selected_sc] += interference_linear

        # Add noise to V2I interference.
        interference_V2V_V2I += self.sig2

        # --- V2I Signal Calculation ---
        V2I_Signal = 10 ** ((self.V2I_power_dB - self.pathloss_V2I_eNB +
                             self.veh_ant_gain + self.bs_ant_gain - self.bs_noise_figure) / 10)
        # Compute spectral efficiencies (SE) for V2I.
        V2I_SE = np.log2(1 + V2I_Signal / interference_V2V_V2I)
        V2I_SE_ideal = np.log2(1 + V2I_Signal / self.sig2)
        self.V2I_SE, self.V2I_SE_ideal = V2I_SE, V2I_SE_ideal

        """ Compute interference and data rates of V2V links """
        interference_V2V_tot = np.zeros((self.num_agents, 1)) #total intf @ V2V receivers
        interference_V2V_Rx = np.zeros((self.num_agents, self.num_agents)) #intf from each V2V sender @reciever
        interference_V2V_other = np.zeros((self.num_agents, self.num_agents))    #intf from sender to others??
        interference_V2I_Rx = np.zeros((self.num_agents, 1)) #intf @ V2V receiver due to V2I
        signal_V2V = np.zeros((self.num_agents, 1))     #Signal power for each V2V link


        for i in range(self.num_agents): #loop through V2V transmitters
            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[0]
            for j in range(self.n_neighbor):
                selected_sc = int(channel_sel[i, j])
                if selected_sc == -1:
                    continue

                # Compute desired V2V signal for agent i.
                sender_power_dB = self.V2V_power_dB_list[int(power_sel[i, j])]
                signal_V2V[i, 0] = 10 ** ((sender_power_dB - self.pathloss_V2V_V2V[veh_sender, veh_receiver] +
                                           2 * self.veh_ant_gain - self.veh_noise_figure) / 10)

                # V2I interference at the V2V receiver:
                intf_V2I_Rx = 10 ** ((self.V2I_power_dB - self.pathloss_V2I_V2V[selected_sc, veh_receiver]
                                        + 2 * self.veh_ant_gain - self.veh_noise_figure) / 10)
                interference_V2I_Rx[i, 0] = intf_V2I_Rx
                interference_V2V_tot[i, 0] += intf_V2I_Rx

                # V2V interference from other agents sharing the same channel.
                for x in range(self.num_agents):
                    if x == i:
                        continue    #ignore interference between same vehicle

                    # Check if the other agent uses the same channel in the same neighbor index.
                    if int(channel_sel[x, j]) == selected_sc:
                        sender_power_dB = self.V2V_power_dB_list[int(power_sel[x, j])]
                        veh_other_sender = self.agent2veh[x]
                        intf_V2V_Rx = 10 ** ((sender_power_dB
                                              - self.pathloss_V2V_V2V[veh_other_sender, veh_receiver]
                                              +2 * self.veh_ant_gain - self.veh_noise_figure) / 10)

                        interference_V2V_Rx[i, x] = intf_V2V_Rx
                        interference_V2V_tot[i, 0] += intf_V2V_Rx
                        interference_V2V_other[x, i] = intf_V2V_Rx


        # Total interference for V2V links: add noise floor
        interference_V2V_tot += self.sig2

        #compute spectral efficiency
        V2V_SE = np.log2(1 + np.divide(signal_V2V, interference_V2V_tot))

        #store values for debugging
        self.interference_V2V_tot = interference_V2V_tot
        self.interference_V2V_Rx = interference_V2V_Rx
        self.interference_V2V_other = interference_V2V_other
        self.interference_V2I_Rx = interference_V2I_Rx
        self.signal_V2V = signal_V2V
        self.V2V_SE = V2I_SE    #Spectral efficiency

        """ Queue Update """
        # Data rate consumption: decrease queue by data transmitted.
        self.queue -= (V2V_SE * self.time_fast * self.bandwidth_per_SC / self.CAM_size)
        self.queue[self.queue <= 0] = 0  # Ensure no negative queue values.

        """Reward Calculation: For each V2V link, if its queue is empty then reward is 1;
            otherwise, compute a reward based on the achieved SE and a penalty on SE losses. """
        reward_V2V = np.zeros((self.num_agents, 1))
        for i in range(self.num_agents):    #loop through V2V senders
            if self.queue[i] <= 0:  #empty queue == max reward
                reward_V2V[i, 0] = 1.0
            else:
                # Compute the spectral efficiency loss due to interference from others

                # get values for current link
                signal_power = signal_V2V[i, 0]
                total_interference = interference_V2V_tot[i, 0]

                # Sum of interference the current link causes to others
                interference_to_others = interference_V2V_other[i, :].sum()

                # Effective interference doesnt consider intfr the current link causes to others
                effective_interference = total_interference - interference_to_others

                # Spectral efficiency **without** interference from others
                V2V_SE_no_intf = np.log2(1 + np.divide(signal_power, effective_interference))

                # Loss term: Difference between spectral efficiency with-w/o full interference
                loss_term = V2V_SE_no_intf - V2V_SE[i, 0]

                # Compute the reward, scaled by a factor of 10
                raw_reward = (V2V_SE[i, 0] - self.lambda1 * loss_term) / 10

                # Clip reward to be within [-1, 1] to avoid extreme values
                reward_V2V[i, 0] = np.clip(raw_reward, -1, 1)

        # Store individual rewards
        self.individual_rewards = reward_V2V

        # Return spectral efficiencies, updated queue, and computed rewards
        return V2V_SE, V2I_SE, self.queue, reward_V2V

    def compute_V2V_pathloss(self, pos_a, pos_b):
        """compute V2V pathloss for any 2 vehicles (V2V or V2I) given positional
            data between vehicles. Fastfading included if self.fast_fading = True
            Inputs: Positional data of 2 vehicles [m]
            Output: pathloss [dB]"""
        #compute euclidean distance b/w vehicles
        dx, dy = abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1])
        d = math.hypot(dx, dy)

        #compute breakpoint distance
        conversion = 1e9 / 3e8 #Conversion from GHz->Hz (1e9) and speed of light = 3e8
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * conversion

        def PL_Los(d):
            """line-of-sight (LoS) path loss function"""
            if d <= 3:      #minimum case is 3m
                return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5) #fc=2
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return (40.0 * np.log10(d) + 9.45
                            - 17.3 * np.log10(self.h_bs)    #h_bs = 1.5
                            - 17.3 * np.log10(self.h_ms)    #h_ms = 1.5
                            + 2.7 * np.log10(self.fc / 5))
        def PL_NLos(d_a, d_b):
            """Non-line-of-sight (NLoS) path loss function"""
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        #select appropriate path loss model
        if min(dx, dy) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(dx, dy), PL_NLos(dy, dx))

        #compute fast fading if enabled
        if self.fast_fading:
            rayleigh_fading = np.random.rayleigh(scale=1.0)
            PL += 10 * np.log10(rayleigh_fading)

        return PL, d

    def compute_V2I_pathloss(self, pos_a):
        """compute V2I pathloss for any 1 vehicle and BaseStation (V2V or V2I) vehicles.
            Fastfading included if self.fast_fading = True
                    Inputs: Positional data 1 vehicle [m]
                    Output: pathloss [dB]"""
        # compute euclidean distance b/w vehicle and BS
        dx, dy = abs(pos_a[0] - self.eNB_xy[0]), abs(pos_a[1] - self.eNB_xy[1])
        d = math.hypot(dx, dy)

        #factor in height of BaseStation
        d_3d = math.hypot(d, self.h_bs - self.h_ms)

        # Apply the Urban Macrocell Path Loss Model
        PL = 128.1 + 37.6 * np.log10(d_3d / 1000)

        # compute fast fading if enabled
        if self.fast_fading:
            rayleigh_fading = np.random.rayleigh(scale=1.0)
            PL += 10 * np.log10(rayleigh_fading)

        return PL


















if __name__ == "__main__":
    from util.parameters import ParametersMAPPO
    params = ParametersMAPPO()
    env = V2XEnvironment(params)
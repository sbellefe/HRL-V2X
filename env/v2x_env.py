import math, random, sys
import numpy as np
import matplotlib.pyplot as plt

from env.env_helper import sample_veh_positions, remove_test_data_from_veh_pos
from env.render import HighwayVisualizer

import sys, os


class Vehicle:
    """Vehicle class to encode positional data for V2V and V2I cars """
    def __init__(self, veh_id, start_position, velocity):
        self.id = veh_id
        self.position = start_position
        self.velocity = velocity
        self.destinations = []
        self.neighbors = [] #NOT CURRENTLY USED


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

        """Rendering"""
        self.render_control_interval = env_setup['render_mode']
        if self.render_control_interval:
            self.visualizer = HighwayVisualizer()


        """local env parameters"""
        self.n_neighbor = 1  # number of neighboring vehicles need to receive CAM
        #Radio transmission related
        self.fc = 2e9                     # carrier frequency [Hz]
        self.num_SC = 4                 # number of sub-channels and V2I links
        self.V2V_power_dB_list = [23, 15, 5]  # power levels at the PM vehicle side
        self.V2I_power_dB = 23          # dBm
        self.sig2 = 10 ** (-114 / 10)   #-114dB power level of AWGN noise
        self.h_bs = 25.0                 # height of base station antenna [m]
        self.h_ms = 1.5                 # height of mobile station (vehicle) antenna [m]
        # self.eNB_xy = [500, -35]        # position of base station (callibration)
        self.eNB_xy = [0, -35]        # position of base station (journal 4 4)
        # self.eNB_xy = [500, -43]        # position of base station
        self.veh_ant_gain = 3           # Vehicle antenna gain
        self.veh_noise_figure = 9       # Vehicle noise figure
        self.bs_ant_gain = 8            # Base station antenna gain
        self.bs_noise_figure = 5        # Base station noise figure
        self.norm_V2V_channel_factor = 120  # Normalization factor for V2V channel
        self.d_bp = 4 * (self.h_ms-1) * (self.h_ms-1) * self.fc / 3e8   #V2V breakpoint distance

        #Reward constants no AoI (gamemodes 2,4)
        self.lambda1 = 0.01  # reward weight for V2V_SE term, if queue != empty
        self.reward_G = 5.0  # reward constant for queue = empty

        #Reward constants w/ AoI (gamemodes 3,5)
        self.lambda2 = 0.1  # reward weight for V2V_SE term
        self.lambda3 = 1  # reward weight for AoI term
        # self.lambda1 = 0.001  # NOT USED

        # Queue related
        self.NQ = 1  # number of buffer size at each agent
        self.time_fast = 0.001  #queue parameter
        self.bandwidth_per_SC = int(1000000)  # bandwidth of each sub-channel in Hz
        self.CAM_size = 25600  # number of bits per cooperative awareness message (''CAM), in bits'

        """extract and format train/test data samples (veh_pos_data has 600 samples)"""
        if not self.multi_location:  # NFIG & SIG-sloc only
            self.veh_pos_data = sample_veh_positions(veh_pos_data, t0=self.single_loc_idx)
            self.test_data_list = self.veh_pos_data
        else:  # SIG-mloc & POSIG
            self.test_data_list = [sample_veh_positions(veh_pos_data, t0=step, k_max=self.k_max) for step in
                                   self.multi_loc_test_idx]
            self.veh_pos_data = remove_test_data_from_veh_pos(veh_pos_data, self.multi_loc_test_idx)

        """calculated parameters"""
        #1) General Platoon
        self.platoons_V2V = [2 for _ in range(self.num_agents)]
        self.num_veh_V2V = sum(self.platoons_V2V)
        self.num_veh_V2I = self.num_SC #num V2I=num sub-channel

        #2) action dimension and mapping 2 RRA dictionary
        self.num_power_levels = len(self.V2V_power_dB_list)
        self.action_dim = self.num_power_levels * self.num_SC + 1 #+1 for "do-nothing" (null)

        self.action2RRA = {}
        for action in range(self.action_dim - 1):   #actions minus null action
            sc = action // self.num_power_levels    #floor division for subchannel
            pw = action % self.num_power_levels     #modulus for power level
            self.action2RRA[action] = (sc, self.V2V_power_dB_list[pw])
        self.action2RRA[self.action_dim-1] = (-1, -1) #do-nothing (null action)

        # HRL USAGE inverted mapping: (sc, dB_idx) → flat_int
        self.RRA_idx2int = {
            (opt, a): (
                # if opt selects a real sub-channel AND a is a real power index
                (opt * self.num_power_levels + a)
                if (opt < self.num_SC and a < self.num_power_levels)
                # else (either "no-transmit" option or "do-nothing" action) → null action
                else self.action_dim - 1
            )
            for opt in range(self.num_SC + 1)  # 0…num_SC  (last one = NO-TRANSMIT)
            for a in range(self.num_power_levels + 1)  # 0…num_power_levels  (last one = DO-NOTHING)
        }



        # # now every (sc, 3) → 12, every (sc, 0–2) → 0–11
        # self.RRA_idx2int = {
        #     (sc, pw): (sc * self.num_power_levels + pw
        #                if pw < self.num_power_levels
        #                else self.action_dim - 1)
        #     for sc in range(self.num_SC)
        #     for pw in range(self.num_power_levels + 1)
        # }


        # self.RRA_idx2int = {}
        # # 0..11: real (subchannel, power) combos
        # for sc in range(self.num_SC):            # 0,1,2,3
        #     for pw in range(self.num_power_levels):       # 0,1,2
        #         flat = sc * self.num_power_levels + pw
        #         self.RRA_idx2int[(sc, pw)] = flat
        # for pw in range(self.num_power_levels): #"No Transmit"
        #     self.RRA_idx2int[(self.num_SC, pw)] = self.action_dim - 1

        # 3) Agent2Vehicle mapping dictionary, assumes the 1st vehicle in platoon is transmitting agent
            #keys: agent index (i.e. 0, 1, 2, 3,...)
            #values: V2V vehicle index (i.e. 0, 2, 4, 6,...) """
        self.agent2veh = {}
        agent_index, veh_id = 0, 0
        for size in self.platoons_V2V:
            self.agent2veh[agent_index] = veh_id
            veh_id += size  #skip over the rest of the platoon
            agent_index += 1

        #4) State dimension
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

        #5) observation (local state) dimension
        if self.game_mode == 4:
            self.obs_dim = dim_t + dim_G_m + (dim_G_i + dim_G_ji + dim_G_Bi + dim_G_iB + dim_q) // self.num_agents
        elif self.game_mode == 5:
            self.obs_dim = dim_t + dim_G_m + (dim_G_i + dim_G_ji + dim_G_Bi + dim_G_iB + dim_q + dim_AoI) // self.num_agents
        else: self.obs_dim = None

        """dynamic parameters"""
        self.testing_mode = False
        self.current_veh_positions = None   #single positional timestep
        self.episode_veh_positions = None   #single or multi positional timestep
        # self.active_links = None  #NOT USED
        self.queue = None
        self.AoI = None
        self.episode_rewards = 0.0

        # ---- pathloss dictionary setup ----
        self.channel_keys = ['V2I_V2V', 'V2V_V2V', 'V2I_eNB', 'V2V_eNB']
        pathloss_shapes = {
            'V2I_V2V': (self.num_veh_V2I, self.num_veh_V2V,),
            'V2V_V2V': (self.num_veh_V2V, self.num_veh_V2V,),
            'V2I_eNB': (self.num_veh_V2I,),
            'V2V_eNB': (self.num_veh_V2V,),
        }
        # large-scale (slow-fading) pathlosses:
        self.pathlosses_sf = {k: np.zeros(pathloss_shapes[k]) for k in self.channel_keys}
        # total pathlosses including small-scale (fast-fading):
        self.pathlosses_tot = {k: self.pathlosses_sf[k].copy() for k in self.channel_keys}

        # ---- below are kept for debugging/rendering ----
        self.distance_V2I_V2V = None    #dim = [num_veh_v2i, num_veh_v2v]
        self.distance_V2V_V2V = None    #dim = [num_veh_v2v, num_veh_v2v]
        self.individual_rewards = None  #dim = [num_agents]
        self.V2V_SE = None      #Spectral efficiency
        self.V2I_SE = None      #Spectral efficiency
        self.signals = {}       #desired signals (dB)
        self.interferences = {} #

        """Initialize vehicles once by creating vehicle objects and computing neighbors/destinations."""
        self.vehicles_V2V = []
        self.vehicles_V2I = []
        sampled_veh_pos = sample_veh_positions(veh_pos_data, t0=self.single_loc_idx)  # use sloc sample for vehicle init
        self.initialize_vehicles(sampled_veh_pos)

        """DEBUGGING"""
        self.debug = False
        self.n_ep = 0


    def initialize_vehicles(self, sampled_data):
        """Initialize vehicles given position data sample. Called in
            __init__ to create all vehicle objects w/ static params."""

        # Store positional data (for sloc this is the only time we store data, for mloc this is overwritten)
        self.current_veh_positions = self.episode_veh_positions = sampled_data

        def add_vehicle(veh_id, position, velocity, type):
            """vehicle adder function"""
            if type == "V2V" :
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

        #check valid data format
        if len(self.vehicles_V2I) < self.num_veh_V2I or \
            len(self.vehicles_V2V) < self.num_veh_V2V:
            raise ValueError(f"Position sample doesn't include data for enough vehicles!")

        #build dictionary mapping from veh_id string to vehicle objects
        self._veh_map = {
            **{v.id: v for v in self.vehicles_V2I},
            **{v.id: v for v in self.vehicles_V2V},
        }

        #Add destinations indices to V2V vehicle objects (agent vehicles only).
        for agent_idx in range(self.num_agents):
            sender_idx = self.agent2veh[agent_idx]
            receiver_idx = sender_idx + 1   #(next vehicle in vehicles_V2V),
            agent_veh = self.vehicles_V2V[sender_idx]
            agent_veh.destinations = [receiver_idx]

        #in the indended use case (n_neighbor > 1), the idea is that vehicle object attributes:
        #   -neighbors: list of all neighboring vehicles which could provide (transmit or interference) signals
        #   -destinations: the subset of neighbors which is the intended transmit signal.
        #this is not used due to the assumption of 2 V2Vs per communication channel (n_neighbor=1)


    def update_vehicle_positions(self, sampled_veh_pos_data):
        """Updates V2V and V2I vehicle objects in one pass using a dict lookup
            Usage in: env.reset() and env.reset_control()"""
        # iterate with itertuples for a small speed boost over iterrows
        for row in sampled_veh_pos_data.itertuples(index=False):
            # row.id, row.x, row.y, row.speed
            veh = self._veh_map[row.id]
            if veh:
                veh.position = [row.x, row.y]
                veh.velocity = row.speed

    def reset(self, test_idx=None):
        """Create a new random game by reloading vehicle positions, reinitializing channels,
            queues, active links, and AoI values.
            Re-sample vehicle data based on the game mode:
                - For game_mode 1, 2, 3: No AoI; a single control interval is used (1 positional data sample).
                - For game_mode 4, 5: AoI; k_max control intervals (positional sample)
            Previous name was 'new_random_game' """

        """Sample and store new vehicle positions (mloc only)"""
        if self.multi_location is True:     #Multi-Location
            #SAMPLE
            if self.testing_mode is False:  # Training mode
                sampled_data = sample_veh_positions(self.veh_pos_data, k_max=self.k_max)
            else:                           #Testing Mode
                if test_idx is None:    #Select test data at random
                    t0 = np.random.choice(len(self.test_data_list))
                else:   # loop through test indices in order
                    t0 = test_idx % len(self.test_data_list)
                sampled_data = self.test_data_list[t0]

            #STORE episode and current positional data samples
            self.episode_veh_positions = sampled_data
            t0 = sampled_data['time'].unique()[0]  # Get the first unique time index
            self.current_veh_positions = sampled_data[sampled_data['time'] == t0]

            #UPDATE vehicle positions
            self.update_vehicle_positions(self.current_veh_positions)

            #DEBUG
            if self.debug:
                MODE = "TESTING" if self.testing_mode else "TRAINING"
                print(f"\n-----------Starting Episode {self.n_ep}!!!! ({MODE}) t_idx: {t0}")


        """Update channels and other environment parameters"""
        self.renew_channels()   #Compute slow- and fast-fading pathlosses
        self.episode_rewards = 0.0  #reward printing/tracking only
        # self.active_links = np.ones((self.num_agents, self.n_neighbor), dtype='bool')  # TODO:??
        if self.game_mode != 1:
            self.queue = np.ones(self.num_agents)
            # self.queue = np.ones((self.num_agents, self.n_neighbor))
        if self.game_mode in [3,5]: #NOT IMPLEMENTED
            self.AoI = np.ones((self.num_agents, self.n_neighbor))

        global_state, local_states, fp_global_states = self.get_state(t_step=0)

        """rendering of positional data and pathloss info"""
        if self.render_control_interval:
            self.render()

        # print(f"\n-----------Starting Episode {self.n_ep}!!!! t_idx: {t0}")
        # print('GS = ', np.round(global_state[0,:], decimals=2))
        # print("V2I positions", [veh.position for veh in self.vehicles_V2I])

        return global_state, local_states, fp_global_states

    def render(self):
        """extract rendering metrics and send draw frame.Choose between pathlosses:
            1. pathlosses_sf for large-scale fading only (non-changing in control interval)
            2. pathlosses_tot also includes fast-fading (varies per communication interval)
            """
        pathlosses = self.pathlosses_tot
        position_info = {
            'pl_v2i_v2v': pathlosses['V2I_V2V'],  # [m, i*2]
            'pl_v2i_bs': pathlosses['V2I_eNB'],  # [m]
            'pl_v2v_bs': pathlosses['V2V_eNB'],  # [i*2]
            'pl_v2v_v2v': pathlosses['V2V_V2V'],  # [i*2, i*2]
            'd_v2i_v2v': self.distance_V2I_V2V,  # [m, i*2]
            'd_v2v_v2v': self.distance_V2V_V2V,  # [i*2, i*2]
        }
        self.visualizer.draw_frame(self.current_veh_positions, position_info, self.testing_mode)

    def step(self, raw_actions, k, t):
        """Executes one communication interval step in the environment.
            Args:
                raw_actions, list(int): list of raw actions for each agent
                k (int): Index of control interval (will always be 1 unless we include AoI i.e. game_mode in [3,5]).
                t (int): Current time step (communication interval) in the control interval (episode)
            Returns:
                global_next_state: np.array
                local_next_states: dict(np.array)
                global_reward: float
                (TEMPORARY) individual reward: list(float) length(num_agents)
                done: bool """

        # Convert action to RRA, Unpack action tuples into arrays
        RRA_all_agents = [self.action2RRA[a] for a in raw_actions]
        channel_sel = np.array([sc for sc, _ in RRA_all_agents], dtype='int32')  # (num_agents,)
        power_sel = np.array([pw for _, pw in RRA_all_agents], dtype='int32')  # (num_agents,)

        # """DEBUGGING - hardset all actions"""
        # channel_sel = np.array([0, 1, 2, 3], dtype='int32')  # (num_agents,)
        # power_sel = np.array([23 for _ in range(self.num_agents)], dtype='int32')  # (num_agents,)

        # --- Compute Performance Metrics ---
        global_reward, individual_rewards, queue = self.compute_reward(channel_sel, power_sel)

        # --- Update Environment State ---
        self.queue = queue
        if self.fast_fading:
            self.renew_fast_fading()
        self.episode_rewards += global_reward

        # --- Check Episode Completion ---
        done = (t == self.t_max - 1) and (k == self.k_max)

        #---- Get next state ----
        global_next_state, local_next_states, fp_global_next_states = self.get_state(t_step=t+1)

        if self.debug:
            PLji = np.round(self.gs['gji'], decimals=2)
            PLBi = np.round(self.gs['gBi'], decimals=2)
            print(f"**step: {t} | " 
                  f"Channel selection: {[int(c) for c in channel_sel]} | "
                  f"Power selection: {[int(p) for p in power_sel]} | "
                  f"global_reward = {global_reward:.2f} | "
                  f"individual_rewards = {np.round(individual_rewards, decimals=2)} | "
                  f"queue = {np.round(self.queue, decimals=2)} | "
                  )


        # #CHECKING SEs
        # print(f't = {t}. '
        #       f'sc_sel: {channel_sel} | '
        #       f'pw_sel: {power_sel} | '
        #       f'V2V_SE: {np.round(self.V2V_SE, decimals=2)} | '
        #       f'V2I_SE: {np.round(self.V2I_SE, decimals=2)} | '
        #       f'q: {np.round(self.queue, decimals=2)}')

        if done:
            t0 = self.current_veh_positions['time'].unique()[0]
            # print(f"------Episode {self.n_ep} Complete! t0 = {t0} | R_tot = {self.episode_rewards:.2f}")
            # if self.debug:  #IN TRAIING LOOP NOW
            #     print(f"------Episode {self.n_ep} Complete! R_tot = {self.episode_rewards:.2f}")
            if not self.testing_mode: self.n_ep += 1

        #TODO: Remove individal_rewards, should only use global....
        return global_next_state, local_next_states, fp_global_next_states, global_reward, individual_rewards, done

    def renew_channels(self):
        """Renew channels by recalculating all pathloss values in the two dictionaries:
            1. self.pathlosses_sf: (slow-fading only), only updated in this method
            2. self.pathlosses_tot: (slow- and fast- fading), updated in this method and in step method
            Note that the 2 dictionaries are only different if fast_fading = True
            - self.channel_keys = ['V2I_V2V', 'V2V_V2V', 'V2I_eNB', 'V2V_eNB']
            """

        pl_sf = self.pathlosses_sf #make local reference

        #dimension attributes stored for rendering
        self.distance_V2I_V2V = np.zeros_like(pl_sf['V2I_V2V'])
        self.distance_V2V_V2V = np.zeros_like(pl_sf['V2V_V2V'])

        # ---- Channels between V2I and V2V vehicles ----
        for m in range(self.num_veh_V2I):
            for i in range(self.num_veh_V2V):
                xy_v2v, xy_v2i = self.vehicles_V2I[m].position, self.vehicles_V2V[i].position
                pl_sf['V2I_V2V'][m, i] = self.compute_V2V_pathloss(xy_v2v, xy_v2i)
                self.distance_V2I_V2V[m, i] = math.dist(xy_v2v, xy_v2i)

        # ---- Channels between all V2V vehicles ----
        for i in range(self.num_veh_V2V):
            for j in range(i+1, self.num_veh_V2V):
                xy_i, xy_j = self.vehicles_V2V[i].position, self.vehicles_V2V[j].position
                pl_sf['V2V_V2V'][i, j] = pl_sf['V2V_V2V'][j, i] = self.compute_V2V_pathloss(xy_i, xy_j)
                self.distance_V2V_V2V[i, j] = self.distance_V2V_V2V[j, i] = math.dist(xy_i, xy_j)

        # ---- Channels between V2I vehicles and base station (eNB) ----
        for m in range(self.num_veh_V2I):
            pl_sf['V2I_eNB'][m] = self.compute_V2I_pathloss(self.vehicles_V2I[m].position)

        # ---- Channels between V2V vehicles and base station (eNB) ----
        for i in range(self.num_veh_V2V):
            pl_sf['V2V_eNB'][i] = self.compute_V2I_pathloss(self.vehicles_V2V[i].position)

        # ---- Reset/initialize total pathloss as new slow-fading pathloss ----
        self.pathlosses_tot = {k: pl_sf[k].copy() for k in self.channel_keys}

        # ---- Update fast fading pathloss attributes ----
        if self.fast_fading:
            self.renew_fast_fading()


    def renew_fast_fading(self):
        """Superimpose new Rayleigh small-scale fading onto slow-fading losses"""
        for k in self.channel_keys: #['V2I_V2V', 'V2V_V2V', 'V2I_eNB', 'V2V_eNB']
            pl = self.pathlosses_sf[k]   #slow-fading loss
            rayleigh_fading = np.random.rayleigh(scale=1.0, size=pl.shape)
            self.pathlosses_tot[k] = pl + 10 * np.log10(rayleigh_fading)


    def reset_control(self, k):
        """Used only for gamemodes 3,4 with multiple control intervals. At the
            start of non-initial control intervals: update positions,channels,queues, AoI,active links
            previous name 'renew_positions' """
        # Update current positions based on the episode data at the given interval
        episode_data = self.episode_veh_positions
        new_timestep = episode_data['time'].unique()[k - 1]  #get unique time index
        self.current_veh_positions = episode_data[episode_data['time'] == new_timestep]
        self.update_vehicle_positions(self.current_veh_positions)
        self.renew_channels()

        # Reset and update queues and AoI as per control episode requirements.
        if self.game_mode != 1:
            self.queue = np.ones(self.num_agents)
            # self.queue = np.ones((self.num_agents, self.n_neighbor))
        if self.game_mode in [4, 5]:
            self.AoI = np.ones((self.num_agents, self.n_neighbor))

        # Integrated renew_queue functionality:
        self.AoI[self.queue > 0] += 1
        self.AoI[self.queue <= 0] = 1
        # self.queue = np.ones((self.num_agents, self.n_neighbor))
        # self.queue[:, 0] = np.minimum(self.queue[:, 0], self.NQ)

        # Integrated renew_active_links functionality:
        # self.active_links[:, 0] = self.queue[:, 0] > 0

    def get_state(self, t_step):
        """Generate the state representation based on the current game mode.
           Args:
               REMOVED idx (tuple): Tuple of indices; typically, (agent_index, destination_index).
               t_step (int): Current step t within a control interval.
           Returns:
               global_state (np.ndarray): A state vector of shape (1, state_dim).
               local_states (list(np.ndarray)): A list of local state (observation) vectors (1 for each agent),
                                                   each of shape (1, obs_dim).
            TODO: maybe it is best to store state components in attribute dictionaries so that we could move the fp global state code out of here?
           """

        """Current assumption: All V2V transmitting vehicles only have 1 destination in the attribute list"""
        idx = [0, 0]    #[a, d] = [agent_idx, destination_idx]
        N = self.num_agents

        def norm_gain(pathloss):
            """Helper function to normalize channel gain between two positions."""
            return pathloss / self.norm_V2V_channel_factor

        """G_i: Common V2V channel gain from its V2V sender to its destination (all gamemodes)"""
        G_i = []    #dim= [num_agents]
        for i in range(N):
            sender_idx = self.agent2veh[i]
            receiver_idx = self.vehicles_V2V[sender_idx].destinations[idx[1]]
            G_i.append(norm_gain(self.pathlosses_tot['V2V_V2V'][sender_idx, receiver_idx]))


        """G_ji: V2V interference channel gain for each pair of different agents (all gamemodes)"""
        #TODO FIx 2d version!!!!
        G_ji = []   #dim= [num_agents * (num_agents - 1)]
        G_ji_2d = []   #dim= [num_agents, (num_agents - 1)] (for easy slicing)
        for i in range(N):        #loop for agents i to get receiving V2V pair
            agent_idx = self.agent2veh[i]
            receiver_idx = self.vehicles_V2V[agent_idx].destinations[idx[1]]
            G_j = []    #store intf at each agent receiver seperately
            for j in range(N):    #loop for all other V2V senders
                if i == j: continue
                sender_idx = self.agent2veh[j]
                gain = norm_gain(self.pathlosses_tot['V2V_V2V'][sender_idx, receiver_idx])
                G_j.append(gain); G_ji.append(gain)
            G_ji_2d.append(G_j)

        """G_m: V2I channel gain for each subchannel (all gamemodes)"""
        #dim = [num_veh_V2I]
        G_m = [norm_gain(self.pathlosses_tot['V2I_eNB'][m]) for m in range(self.num_veh_V2I)]

        """G_Bi: Interference channel gain V2I (sender) to V2V (receiver) (all gamemodes)"""
        G_Bi = []  #dim = [num_agents * num_veh_V2I]
        G_Bi_2d = []   #dim= [num_agents, num_veh_V2I] (for easy slicing)
        for i in range(N):
            agent_idx = self.agent2veh[i]
            receiver_idx = self.vehicles_V2V[agent_idx].destinations[idx[1]]
            G_B = []    #store intf at each agent receiver seperately
            for m in range(self.num_SC):
                sender_idx = m
                gain = norm_gain(self.pathlosses_tot['V2I_V2V'][sender_idx, receiver_idx])
                G_Bi.append(gain); G_B.append(gain)
            G_Bi_2d.append(G_B)

        """G_iB: Interference channel gain V2V (sender) to BaseStation (receiver) (all gamemodes)"""
        G_iB = []   #dim = [num_agents]
        for i in range(N):
            sender_idx = self.agent2veh[i]
            G_iB.append(norm_gain(self.pathlosses_tot['V2V_eNB'][sender_idx]))

        """Compute state components for queue gamemodes """
        if self.game_mode != 1:
            #t:one-hot encoded communication interval time vector
            t = np.zeros(self.t_max)
            t[min(t_step, self.t_max - 1)] = 1

            #Queue: dim = [num_agents]
            q_i = self.queue.flatten()
            # q_i = (self.queue / self.NQ).flatten()
        else:
            q_i, t = [], []
            # queue_length = self.queue[idx[0], idx[1]] / self.NQ #FROM get_observation

        """AgeofInformation: Not currently implemented."""
        if self.game_mode in [3, 5]:
            AoI = self.AoI.flatten() #dim = [num_agents]
        else:
            AoI = []
            # AoI = self.AoI[idx[0], idx[1]]  #FROM get_observation

        """Concatenate global state based on game mode"""
        if self.game_mode == 1: #NFIG
            global_state = np.hstack((G_i, G_ji, G_m, G_Bi, G_iB))
        elif self.game_mode in [2, 4]: #SIG or POSIG noAoI
            global_state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i))
        elif self.game_mode in [3, 5]: #SIG or POSIG AoI
            global_state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i, AoI))
        else: raise ValueError("Invalid game mode")
        global_state = global_state.reshape((1, self.state_dim))

        local_states = None
        fp_global_states = None

        """Build GS dictionary (SIG only for now)"""
        # self.gs = {
        #     't': np.array(t),  # shape (t_max,)
        #     'gi': np.array(G_i),  # shape (N,)
        #     'gji': np.array(G_ji_2d),  # shape (N, N-1)
        #     'gm':  np.array(G_m),  # shape (M,)
        #     'gBi': np.array(G_Bi_2d),  # shape (N, M)
        #     'giB': np.array(G_iB),  # shape (N,)
        #     'qi': np.array(q_i),  # shape (N,)
        # }


        """Get Observations: Create dictionary of local states,
           Get Feature-pruned (FP) agent-specific global states: dictionary """
        if self.partial_observability:
            local_states, fp_global_states = {}, {}

            #first, convert state components into arrays
            G_i = np.array(G_i)  # shape (N,)
            G_ji = np.array(G_ji_2d)  # shape (N, N-1)
            G_m = np.array(G_m)  # shape (num_veh_V2I,)
            G_Bi = np.array(G_Bi_2d)  # shape (N, num_SC)
            G_iB = np.array(G_iB)  # shape (N,)
            q_i = np.array(q_i)  # shape (N,)
            t = np.array(t)  # shape (t_max,)
            if self.game_mode == 5: AoI = np.array(AoI) # shape (N,)

            #now loop and build states for each agent
            for i in range(N):
                mask = np.arange(N) != i    #mask to exclude agent i
                #build local state Communal to game_mode 4, 5
                components = [t,
                              np.array([G_i[i]]),  # shape (1,)
                              G_ji[i],  # shape (N-1,)
                              G_m,
                              G_Bi[i],  # shape (num_SC,)
                              np.array([G_iB[i]]),
                              np.array([q_i[i]])
                            ]
                if self.game_mode == 5:
                    components.append(np.array([AoI[i]]))

                local_state = np.concatenate(components)

                #build fp global state
                other_components = [
                    G_i[mask],                  # shape (N-1)
                    G_ji[mask].ravel(),         # shape ((N-1)*(N-1))
                    G_Bi[mask].ravel(),         # shape ((N-1)*num_SC)
                    G_iB[mask],                 # shape (N-1)
                    q_i[mask]                   # shape (N-1)
                ]
                if self.game_mode == 5:
                    other_components.append(AoI[mask])  # (N-1,)

                other_components = np.concatenate(other_components) # shape (state_dim - obs_dim)

                local_states[i] = local_state.reshape((1, self.obs_dim))
                fp_global_states[i] = np.concatenate([local_state, other_components]).reshape(1, self.state_dim)

        return global_state, local_states, fp_global_states


    def compute_reward(self, channel_sel, power_sel):
        """
        Compute V2V and V2I spectral efficiencies, update the message queue,
        and determine individual and global reward based on game_mode.
        Stores attributes for debugging purposes:
            individual_rewards: list
            spectral_efficiencies: 2 variables (V2V_SE and V2I_SE) Units [bits/s/Hz]
            interferences: dictionary
            signals: dictionary
        Args:
            raw_actions: list of length num_agents, where each entry is
                        raw integer action to be decoded by self.action2RRA
        Returns:
            global_reward: float
            individual_rewards: np.ndarray, shape = (num_agents)
            queue: np.ndarray, shape = (num_agents)
            """

        # Convert action to RRA, Unpack action tuples into arrays
        # RRA_all_agents = [self.action2RRA[a] for a in raw_actions]
        # channel_sel = np.array([sc for sc, _ in RRA_all_agents], dtype='int32')   # (num_agents,)
        # power_sel =  np.array([pw for _, pw in RRA_all_agents], dtype='int32')  # (num_agents,)

        # if self.debug:
        #     print(f"Channel selection: {[int(c) for c in channel_sel]}. Power selection: {[int(p) for p in power_sel]}")

        #Initialize / reset interferences + signals to 0
        interferences = {
            'V2V_V2I': np.zeros(self.num_SC),       # Interference on each V2I subchannel from V2V transmitters
            'V2I_V2V': np.zeros(self.num_agents),   # Interference on each V2V link from the V2I base‐station transmitter
            'V2V_V2V': np.zeros(self.num_agents),   # Mutual V2V interference (other V2V agents) on each V2V link
            'V2V_tot': np.zeros(self.num_agents),   # Total noise + interference for each V2V link
        }
        signals = {
            'V2I': np.zeros(self.num_SC),       # Received V2I signal power (linear) per subchannel
            'V2V': np.zeros(self.num_agents),   # Received V2V signal power (linear) per transmitter
        }

        """Compute V2I link spectral efficiencies """
        #first compute interference from V2V transmitters on each V2I channel
        for i in range(self.num_agents):
            sc, pw = channel_sel[i], power_sel[i]
            if sc < 0:   #agent does not transmit
                continue
            #compute and accumulate subchannel interference
            interferences['V2V_V2I'][sc] += 10 ** ((pw - self.pathlosses_tot['V2V_eNB'][i] + self.veh_ant_gain
                                                + self.bs_ant_gain - self.bs_noise_figure) / 10)
        interferences['V2V_V2I'] += self.sig2   # Add thermal noise floor

        # Next compute V2I-to-BS Signal power
        signals['V2I'] = 10 ** ((self.V2I_power_dB - self.pathlosses_tot['V2I_eNB'] + self.veh_ant_gain
                             + self.bs_ant_gain - self.bs_noise_figure) / 10)

        #Now compute V2I spectral efficiency using Shannon’s formula ---
        V2I_SE = np.log2(1 + signals['V2I'] / interferences['V2V_V2I'])
        # V2I_SE_ideal = np.log2(1 + signals['V2I'] / self.sig2) #not needed


        """Compute V2V link spectral efficiencies"""
        V2V_SE = np.zeros((self.num_agents,))
        for i in range(self.num_agents):
            sc, pw = channel_sel[i], power_sel[i]
            if sc < 0: continue

            sender_idx = self.agent2veh[i]
            receiver_idx = self.vehicles_V2V[sender_idx].destinations[0]

            #desired V2V signal power at receiver, convert from dB to linear incl antenna gain and noise:
            signals['V2V'][i] = 10 ** ((pw - self.pathlosses_tot['V2V_V2V'][sender_idx, receiver_idx]
                                 +2 * self.veh_ant_gain - self.veh_noise_figure)  / 10)

            #interference from V2I transmissions
            interferences['V2I_V2V'][i] = 10 ** ((self.V2I_power_dB - self.pathlosses_tot['V2I_V2V'][sc, receiver_idx]
                                   + 2 * self.veh_ant_gain - self.veh_noise_figure) / 10)

            #interference from other V2V agents on the same channel
            # intf_V2V_V2V = 0.0
            for j in range(self.num_agents):
                if j == i or channel_sel[j] != sc:
                    continue
                path = (self.agent2veh[j], receiver_idx)
                interferences['V2V_V2V'][i] += 10 ** ((power_sel[j] - self.pathlosses_tot['V2V_V2V'][path]
                                            + 2 * self.veh_ant_gain - self.veh_noise_figure) / 10)

            interferences['V2V_tot'][i] = self.sig2 + interferences['V2I_V2V'][i] + interferences['V2V_V2V'][i]

            V2V_SE[i] = np.log2(1 + signals['V2V'][i] / interferences['V2V_tot'][i])

        """Compute queue update"""
        if self.game_mode != 1:
            queue = self.queue - (V2V_SE * self.time_fast *
                                  self.bandwidth_per_SC / self.CAM_size)
            queue[queue <= 0] = 0 # eliminate negative queue
        else:
            queue = None

        """Calculate reward based on game_mode"""
        if self.game_mode == 1:
            global_reward = np.sum(V2V_SE) + np.sum(V2I_SE)
            individual_rewards = []
        elif self.game_mode in [2, 4]:
            individual_rewards = np.where(queue <= 0, self.reward_G, V2V_SE * self.lambda1)
            individual_rewards += np.sum(V2I_SE) * self.lambda1 / self.num_agents
            global_reward = np.sum(individual_rewards)
            # global_reward = np.array([[np.sum(individual_rewards)]])
            # global_reward = np.sum(individual_rewards) + np.sum(V2I_SE) * self.lambda1
        elif self.game_mode in [3, 5]:
            individual_rewards = -self.AoI * self.lambda3 + V2V_SE * self.lambda2
            global_reward = np.sum(individual_rewards)
        else: raise ValueError("Invalid game_mode")

        """Store values for logging if needed"""
        self.interferences = interferences
        self.signals = signals
        self.V2V_SE, self.V2I_SE = V2V_SE, V2I_SE

        if self.debug:
            print(f'V2V_SE: {np.round(self.V2V_SE, decimals=3)} | V2I_SE: {np.round(self.V2I_SE, decimals=3)}')

        return global_reward, individual_rewards, queue


    def compute_V2V_pathloss(self, pos_a, pos_b):
        """compute V2V pathloss for any 2 vehicles (V2V or V2I) given positional
            data between vehicles. Fastfading included if self.fast_fading = True
            Inputs: Positional data of 2 vehicles [m]
            Output: pathloss [dB]"""
        #compute euclidean distance b/w vehicles
        dx, dy = abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1])
        d = math.hypot(dx, dy)

        def PL_LoS(d):
            """line-of-sight (LoS) path loss function"""
            if d <= 3:      #minimum case is 3m
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5e9) #fc=2e9
            else:
                if d < self.d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5e9)
                else:
                    return (40.0 * np.log10(d) + 9.45
                            - 17.3 * np.log10(self.h_ms)
                            - 17.3 * np.log10(self.h_ms)  #h_ms = 1.5
                            + 2.7 * np.log10(self.fc / 5e9))
        def PL_NLoS(d_a, d_b):
            """Non-line-of-sight (NLoS) path loss function: NOT USED"""
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_LoS(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5e9)

        PL = PL_LoS(d)  #always use LoS for freeway
        # #select appropriate path loss model
        # if min(dx, dy) < 7:
        #     PL = PL_LoS(d)
        # else:
        #     PL = min(PL_NLoS(dx, dy), PL_NLoS(dy, dx))

        return PL

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

        #apply empirical urban-macrocell pathloss formula
        PL = 128.1 + 37.6 * np.log10(d_3d / 1000)

        return PL





if __name__ == "__main__":
    from util.parameters import ParametersMAPPO
    params = ParametersMAPPO()
    env = V2XEnvironment(params)
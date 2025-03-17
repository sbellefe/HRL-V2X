import math, random, sys
import numpy as np
import gymnasium as gym






class Vehicle:
    """Vehicle class to encode positional data for V2V and V2I cars """
    def __init__(self, start_position, velocity):
        self.position = start_position
        self.velocity = velocity
        self.destinations = []


class V2XEnvironment(gym.Env):

    def __init__(self, params, veh_pos_data, test_data_list):
        """configuration hyperparameters"""
        self.game_mode = params.game_mode
        self.fast_fading = params.fast_fading
        self.num_agents = params.num_agents
        self.t_max_control = params.t_max_control

        """local env parameters"""
        self.CAM_size = 25600       # number of bits per cooperative awareness message (''CAM), in bits'
        self.NQ = 1  # number of buffer size at each agent
        self.sig2 = 10 ** (-114 / 10)   # -114dB power level of AWGN noise
        self.num_SC = 4       # number of sub-channels and V2I links
        self.V2V_power_dB_list = [23, 15, 5]  # power levels at the PM vehicle side
        self.h_bs = 1.5     #height of base station antenna [meters]
        self.h_ms = 1.5     #height of mobile station (vehicle) antenna [meters]
        self.fc = 2         #carrier frequency [GHz]
        self.eNB_xy = [500, -43] #position of base station
        self.n_neighbor = 1  # number of neighboring vehicles need to receive CAM
        self.norm_V2V_channel_factor = 120  # Normalization factor for V2V channel
        self.vehAntGain = 3  # Vehicle antenna gain
        self.vehNoiseFigure = 9  # Vehicle noise figure


        """working parameters"""
        self.vehicles_V2V = []
        self.vehicles_V2I = []
        self.pathloss_V2I_V2V = None
        self.pathloss_V2I_eNB = None
        self.pathloss_V2V_eNB = None
        self.pathloss_V2V_V2V = None
        self.SC_RSRP_dB = None      #Received Signal Recieved Power
        self.queue = None
        self.AoI = None

        """calculated parameters"""
        self.platoon_V2V = [2 for _ in range(params.num_agents)]
        self.num_veh_V2V = sum(self.platoon_V2V)
        self.num_veh_V2I = self.num_SC

        #Define null_ID: set of V2V cars that are not agents (receiver)
        self.null_ID = set()
        non_ag_offset = 0
        for p in range(len(self.platoon_V2V)):
            n_veh = self.platoon_V2V[p]
            for v in range(n_veh):
                if v == 0:  # Add the first vehicle
                    self.null_ID.add(v + non_ag_offset)

            non_ag_offset += n_veh

        #Agent --> Vehicle ID mapping
        self.agent2veh = {}
        agent_index = 0
        for veh_idx in range(self.num_veh_V2V):
            if veh_idx not in self.null_ID:
                self.agent2veh[agent_index] = veh_idx
                agent_index += 1
            else:
                continue

        #compute state dimension
        dim_t = self.t_max_control
        dim_G_i = self.num_agents
        dim_G_ij = self.num_agents * (self.num_agents - 1)
        dim_G_m = self.num_SC
        dim_G_Bi = self.num_SC * self.num_agents
        dim_G_iB = self.num_agents
        dim_q = self.num_agents
        dim_AoI = self.num_agents


        if self.game_mode == 1: #State = [G_i, G_ji]
            self.state_dim = dim_G_i + dim_G_ij + dim_G_m + dim_G_iB + dim_G_iB
        elif self.game_mode == 2: #State = [t, G_i, G_ji, G_m, G_Bi, G_iB, q]
            self.state_dim = dim_t + dim_G_i + dim_G_ij + dim_G_m + dim_G_iB + dim_G_iB + dim_q
        elif self.game_mode == 3: #State = [t, G_i, G_ji, G_m, G_Bi, G_iB, q, AoI]
            self.state_dim = dim_t + dim_G_i + dim_G_ij + dim_G_m + dim_G_iB + dim_G_Bi + dim_q + dim_AoI
        elif self.game_mode == 4: #State = [t, G_i=1, G_ji, G_m, G_Bi, G_iB, q] #TODO: Review this is correct!
            self.state_dim = dim_t + 1 + 1 * (self.num_agents - 1) + 1
        else:
            raise ValueError("Invalid game mode")

        #                   t: one hot code comm interval      G_i          G_ji            G_m             G_Bi     G_iB  queue






    def add_vehicles(self, veh_pos_data):
        """Load position/speed of each vehicle from dataframe"""

        #
        def add_vehicle(position, velocity, type):
            """vehicle adder function"""
            if type == "V2V":
                self.vehicles_V2V.append(Vehicle(position, velocity))
            elif type == "V2I":
                self.vehicles_V2I.append(Vehicle(position, velocity))

        #loop through rows, store vehicle info
        for _, row in veh_pos_data.iterrows():
            veh_id = row['id']
            position = [row['x'], row['y']]
            velocity = row['speed']

            if veh_id.startswith("carflowV2I"):
                add_vehicle(position, velocity, "V2I")
            elif veh_id.startswith("carflow"):
                add_vehicle(position, velocity, "V2V")







    def reset(self):
        """Create new game (episode). Previous name was 'new_random_game' """
        self.vehicles_V2V = []

        #TODO: Review how we update loaded_veh_data, Should maybe move this functionality from the trainer to the environment file?
        self.add_vehicles(self.loaded_veh_data)
        self.renew_channel()

        #TODO: Are these all used???
        self.queue = np.ones((self.num_agents, self.n_neighbor))
        self.active_links = np.ones((self.num_agents, self.n_neighbor), dtype='bool')
        self.AoI = np.ones((self.num_agents, self.n_neighbor))






    def step(self, action):
        pass


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


        """t:one-hot encoded communication interval time vector (modes 2, 3, 4)"""
        t = np.zeros(self.t_max_control)
        t[min(i_step, self.t_max_control - 1)] = 1


        """G_i: Common V2V channel gain from its V2V sender to its destination (modes 1, 2, 3)"""
        G_i = []
        for i in range(self.num_agents):
            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
            G_i.append(norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver]))

        """G_ji: V2V interference channel gain for each pair of different agents (modes 1, 2, 3, 4)"""
        G_ji = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                veh_sender = self.agent2veh[i]
                veh_receiver = self.vehicles_V2V[self.agent2veh[j]].destinations[idx[1]]
                G_ji.append(norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver]))

        """G_m: V2I channel gain for each subchannel (modes 1, 2, 3, 4)"""
        G_m = [norm_gain(self.pathloss_V2I_eNB[m]) for m in range(self.num_SC)]

        """G_Bi: Interference channel gain V2I (sender) to V2V (receiver) (modes 1, 2, 3, 4)"""
        G_Bi = []
        for i in range(self.num_agents):
            veh_receiver = self.agent2veh[i]
            for m in range(self.num_SC):
                veh_sender = m
                G_Bi.append(norm_gain(self.pathloss_V2I_V2V[veh_sender, veh_receiver]))

        """G_iB: Interference channel gain V2V (sender) to BaseStation (receiver) (modes 1, 2, 3, 4)"""
        G_iB = []
        for i in range(self.num_agents):
            veh_receiver = self.agent2veh[i]
            for m in range(self.num_SC):
                veh_sender = m
                G_Bi.append(norm_gain(self.pathloss_V2I_V2V[veh_sender, veh_receiver]))

        """Queue: (modes 2, 3, 4)"""
        q_i = (self.queue / self.NQ).flatten()
        #TODO: FIX THIS!

        """AgeofInformation: Not currently implemented. (modes 3, not4)"""
        AoI = self.AoI.flatten()
        # TODO: FIX THIS! its different in get_observation method

        """G_i_po: V2V Channel gain for 1 agent in partial observability (mode 4)"""
        veh_sender = self.agent2veh[idx[0]]
        veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
        G_i_po = norm_gain(self.pathloss_V2V_V2V[veh_sender, veh_receiver])


        """Concatonnate state based on game mode"""
        if self.game_mode == 1:
            state = np.hstack((G_i, G_ji, G_m, G_Bi, G_iB))
        elif self.game_mode == 2:
            state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i))
        elif self.game_mode == 3:
            state = np.hstack((t, G_i, G_ji, G_m, G_Bi, G_iB, q_i, AoI))
        elif self.game_mode == 4:   #AoI not included for now?
            state = np.hstack((t, G_i_po, G_ji, G_m, G_Bi, G_iB, q_i))
        else: raise ValueError("Invalid game mode")

        state = state.reshape((1, self.state_dim))

        return state

    def compute_reward(self, actions_power):
        pass


    def compute_SC_received_power(self, actions_power):
        """Computes the received power (including noise and interference) at each subchannel for every agent.
            Parameters: actions_power: ndarray of shape (num_agents, n_neighbor, 2)
                        * actions_power[:, :, 0]: Selected subchannel indices (int)
                        * actions_power[:, :, 1]: Selected power level indices (int)
        """
        # Initialize received power matrix with the noise floor. Shape = [num_agents, num_SC]
        rsrp = np.full((self.num_agents, self.num_SC), self.sig2)

        # Extract channel selections and power selections from actions_power.
        channel_sel = actions_power[:, :, 0]    # (n_agent, n_neighbor)
        power_sel = actions_power[:, :, 1]      # (n_agent, n_neighbor)

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
                        interference_power = 10 ** (
                                (sender_power_dB - pathloss_dB + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                        rsrp[receiver, selected_sc] += interference_power

        # Convert total received power from linear scale to dBm
        self.SC_RSRP_dB = 10 * np.log10(rsrp)


            for nbr_idx in range(self.n_neighbor):
                sel_channel = int(channel_sel[sender, nbr_idx])
                # Skip if no channel is selected.
                if sel_channel < 0:
                    continue
                tx_power_dB = self.V2V_power_dB_list[int(power_sel[sender, nbr_idx])]

                # Loop over all receiving agents.
                for receiver in range(self.num_agents):
                    veh_receiver = self.agent2veh[receiver]
                    # Loop over receiver's neighbor indices.
                    for rx_nbr in range(self.n_neighbor):
                        # Skip if the sender is the same as the receiver, or if the receiver's destination equals the sender.
                        dest = self.vehicles_V2V[veh_receiver].destinations[rx_nbr]
                        if veh_sender == veh_receiver or dest == veh_sender:
                            continue
                        # Calculate interference power in linear scale.
                        pathloss_dB = self.pathloss_V2V_V2V[sender][dest]
                        # The formula converts dB to linear scale.
                        power_linear = 10 ** (
                                    (tx_power_dB - pathloss_dB + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        # Add the contribution to the receiver's selected sub-channel.
                        rsrp[receiver, sel_channel] += power_linear

        # Convert the linear power values to dBm.
        self.Veh_RSRP_per_SC_all_dBm = 10 * np.log10(rsrp)


    def renew_fastfading(self):
        pass

    def renew_channel(self):
        """Renew channels by recalculating all path loss values"""

        # ---- V2I → V2V interference channel ---- Create a matrix of shape (num_veh_V2I, num_veh_V2V)
        self.pathloss_V2I_V2V = np.array([
            [self.compute_V2V_pathloss(v2v.position, v2i.position)
             for v2v in self.vehicles_V2V]
            for v2i in self.vehicles_V2I ])

        # ---- V2I → eNB Channel ---- Create a matrix of shape (num_veh_V2I)
        self.pathloss_V2I_eNB = np.array([
            self.compute_V2I_pathloss(v2i.position) for v2i in self.vehicles_V2I ])

        # ---- V2V → eNB Channel ---- Create a matrix of shape (num_veh_V2V)
        self.pathloss_V2V_eNB = np.array([
            self.compute_V2I_pathloss(v2v.position) for v2v in self.vehicles_V2V ])

        # ---- V2V → V2V Channel ---- Create a matrix of shape (num_veh_V2V, num_veh_V2V)
        n = self.num_veh_V2V
        self.pathloss_V2V_V2V = np.array((n,n))
        for i in range(n):
            for j in range(i+1, n):
                pl = self.compute_V2V_pathloss(self.vehicles_V2V[i].position, self.vehicles_V2V[j].position)
                self.pathloss_V2V_V2V[i, j] = pl
                self.pathloss_V2V_V2V[j, i] = pl


    def compute_V2V_pathloss(self, pos_a, pos_b):
        """compute V2V pathloss for 2 vehicles given positional data
            (between either V2V or V2I vehicles)
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
                return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return (40.0 * np.log10(d) + 9.45
                            - 17.3 * np.log10(self.h_bs)
                            - 17.3 * np.log10(self.h_ms)
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

        return PL

    def compute_V2I_pathloss(self, pos_a):
        """compute V2I pathloss for 1 vehicle and BaseStation
            (between either V2V or V2I vehicles)
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
    params = {}
    env = V2XEnvironment(params)
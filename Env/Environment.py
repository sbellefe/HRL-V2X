"""
Revise by Mei Jie based on original work of Dr. Liu Tong, 2023-11-22
"""
from __future__ import division
import numpy as np
import random
import math
from collections import deque
import time as tm
import sys


class Vehicle(object):
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, velocity):
        self.position = start_position
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, args):

        # ----------V2V channel ---------
        self.individual_rewards = None
        self.total_V2V_Interference_in_dBm = None
        self.Veh_RSRP_per_SC_all_dBm = None
        self.V2V_pathloss_V2V = None
        self.V2V_pathloss_V2V_fast_fading = None
        self.active_links = None
        self.queue = None
        self.vehicles_V2V = []
        self.V2V_SE = None
        self.total_V2V_Interference = None
        self.V2V_Interference_at_Rx = None
        self.V2V_Interference_to_others = None
        self.destinations = None
        self.neighbors = None
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = args.decorrelation_distance
        self.shadow_std = args.shadow_std
        self.AoI = None                     # AoI in current control interval
        # self.AoI_deque = deque()            # AoI deque of length 2, 
        self.reward_G = args.reward_G


        # ----------V2I settings ---------
        self.V2I_SE_losses = None
        self.V2I_pathloss_V2V = None
        self.V2I_pathloss_fast_fading = None
        self.V2I_interference_at_Rx = None
        self.V2I_Signals = None
        self.V2I_SE = None
        self.V2I_power_dB = 23  # dBm

        self.vehicles_V2I = []
        self.eNB_position = [500, -43]

        self.V2V_Interference_V2I = None
        self.V2V_pathloss_eNB = None
        self.V2I_SE_ideal = None

        self.distance_between_V2I_veh = 33



        # -------------------------
        self.n_SC = args.number_of_SC  # Number of RB
        self.n_Veh = args.n_veh  # Number of V2Vs
        self.n_veh_platoon = args.n_veh_platoon
        self.n_agent = args.n_agent
        self.agent2veh = args.agent2veh
        self.loaded_veh_data = args.loaded_veh_data
        self.position_data = None      # sampled position data of vehicles with [t_max_control] timesteps
        self.null_ID = args.null_ID

        self.distance_between_veh = 16.5
        self.n_neighbor = args.n_neighbor
        self.V2V_power_dB_List = args.V2V_power_dBm_List  # the power levels, in the unit of dBm
        self.sig2_dB = args.sig2_dB  # power leve of AWGN noise
        self.norm_V2V_channel_factor = args.norm_V2V_channel_factor
        self.norm_V2V_interference_factor = args.norm_V2V_interference_factor
        self.bsAntGain = args.bsAntGain
        self.bsNoiseFigure = args.bsNoiseFigure
        self.vehAntGain = args.vehAntGain
        self.vehNoiseFigure = args.vehNoiseFigure
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.if_fastFading = args.if_fastFading

        self.time_fast = 0.001
        self.time_slow = args.n_step_per_episode_communication / 1000  # update slow fading/vehicle position every 50 ms
        self.n_step_per_control_interval = int(args.n_step_per_episode_communication)
        self.bandwidth = int(args.BW_per_SC)  # bandwidth per RB, 1 MHz, 2MHz,180000Hz
        self.NQ = args.NQ
        # self.mu=np.zeros((self.n_Veh - 1, self.n_neighbor)) # 离开率
        self.CAM_size = args.CAM_size  # V2V CAM size: 1060 Bytes every 10 ms
        # self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_SC)) + self.sig2
        self.seed = 1
        self.rewardscale = 1e-2
        self.lambda1 = args.reward_w1
        self.lambda2 = args.reward_w2
        self.lambda3 = args.reward_w3

        np.random.seed(self.seed)
        random.seed(self.seed)
        self.nb_timestep = int(args.n_step_per_episode_communication)  # *args.t_max_control
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.actionDim = args.num_actions  # Dimension of action space
        self.num_of_actions = args.num_actions

        self.n_step_per_episode_communication = args.n_step_per_episode_communication
        self.t_max_control = args.t_max_control




        """
        game mode:
        1: Channel Aware Stochastic Interference Game
        2: Channel and Queue Aware Stochastic Interference Game
        3: Channel and AoI Aware Stochastic Interference Game
        4: Partial Observable Stochastic Interference Game 
        """

        self.game_mode = args.game_mode

        if self.game_mode == 1:
            #                   G_i                 G_ji
            self.stateDim = (self.n_agent) + (self.n_agent) * (self.n_agent - 1)

        elif self.game_mode == 2:
            #                 t: one hot code comm interval          G_i                     G_ji                      queue
            self.stateDim = (self.n_step_per_control_interval) + (self.n_agent) + (self.n_agent) * (self.n_agent - 1) + (self.n_agent)

        elif self.game_mode == 3:
            #                 t: one hot code comm interval          G_i                     G_ji                      queue            AoI
            self.stateDim = (self.n_step_per_control_interval) + (self.n_agent) + (self.n_agent) * (self.n_agent - 1) + (self.n_agent) + (self.n_agent)

        elif self.game_mode == 4:
            # #                 t: one hot code comm interval      G_i          G_ji         queue  AoI
            # self.stateDim = (self.n_step_per_control_interval) + (1) + (self.n_agent - 1) + (1) + (1)
            #                 t: one hot code comm interval      G_i          G_ji         queue
            self.stateDim = (self.n_step_per_control_interval) + (1) + (self.n_agent - 1) + (1)

        else:
            print("Error: game mode doesn't exist")
            sys.exit(1)
            
        self.local_stateDim = (self.n_step_per_control_interval) + (1) + (self.n_agent - 1) + (1)
        
        self.game_dic = {}


    def normalize_state(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        

        if max_val == min_val:
            return np.zeros_like(array)
        
        # Normalize to [0, 1]
        normalized = (array - min_val) / (max_val - min_val)
        
        # Rescale to [-1, 1]
        normalized = normalized * 2 - 1
        
        return normalized


    def V2V_get_path_loss(self, position_A, position_B):
        """
            Calculate path loss
        """
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2)
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        # print(d_bp)
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()
        

    def V2V_get_path_loss_with_fast_fading(self, position_A, position_B):

        path_loss = self.V2V_get_path_loss(position_A, position_B)  # Slow fading (path loss)
        
        rayleigh_fading = np.random.rayleigh(scale=1.0)

        total_loss = path_loss + 10 * np.log10(rayleigh_fading)

        return total_loss


    def V2I_get_path_loss_with_fast_fading(self, position_A):

        path_loss = self.V2I_get_path_loss(position_A)
        
        rayleigh_fading = np.random.rayleigh(scale=1.0)
        
        total_loss = path_loss + 10 * np.log10(rayleigh_fading)
        
        return total_loss

    
    def renew_fast_fading(self):
        """
        Renew slow fading channel
        """

        # reset V2V_pathloss_V2V_fast_fading to the original pathloss value before adding rayleigh_fading
        self.V2V_pathloss_V2V_fast_fading = self.V2V_pathloss_V2V
        self.V2I_pathloss_fast_fading = self.V2I_pathloss_V2V


        for i in range(self.n_Veh):
            for j in range(i + 1, self.n_Veh):

                self.V2V_pathloss_V2V_fast_fading[i][j] = self.V2V_pathloss_V2V_fast_fading[j][i] = self.V2V_get_path_loss_with_fast_fading(
                    self.vehicles_V2V[i].position, self.vehicles_V2V[j].position)

        for i in range(self.n_SC):

            veh_pos = self.vehicles_V2I[i]
            self.V2I_pathloss_fast_fading = self.V2I_get_path_loss_with_fast_fading(self.vehicles_V2I[i].position)



    def V2I_get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.eNB_position[0])
        d2 = abs(position_A[1] - self.eNB_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()


    def V2V_add_new_vehicles(self, start_position, start_velocity):
        self.vehicles_V2V.append(Vehicle(start_position, start_velocity))

    def V2I_add_new_vehicles(self, start_position, start_velocity):
        self.vehicles_V2I.append(Vehicle(start_position, start_velocity))

    def add_new_vehicles_by_number(self):
        """
        Initialize the position and speed of each vehicle: constant speed only
        """

        print("Initialize the position and speed of each vehicle: constant speed only")

        # -------------V2V -------------------

        start_position = [416, 427.75]
        start_velocity = 10
        for i in range(self.n_Veh):
            start_position[0] = start_position[0] - i * self.distance_between_veh
            self.V2V_add_new_vehicles(start_position.copy(), start_velocity)

        # -------------V2I -------------------

        start_position = [416, 434.75]
        start_velocity = 10
        for i in range(self.n_SC):
            start_position[0] = start_position[0] + i * self.distance_between_V2I_veh
            self.V2I_add_new_vehicles(start_position.copy(), start_velocity)



    def add_new_vehicles_by_file(self, df):
        """
        load the position and speed of each vehicle from SUMO csv file

        Each carflow has at least 2 vehicles:

        v0: carflow0_1.0
        V1: carflow0_2.0
        """
        # print("Initialize the position and speed of each vehicle: SUMO")

        # -------------V2V -------------------

        # veh_id = [
        #     "carflow0_1.0", 
        #     "carflow0_2.0", 
        #     "carflow1_1.0", 
        #     "carflow1_2.0", 
        #     "carflow2_1.0", 
        #     "carflow2_2.0"
        # ]

        veh_id = []
        for i, num_vehicles in enumerate(self.n_veh_platoon):
            for j in range(num_vehicles):
                veh_id.append(f"carflow{i}_{j}.0")



        df_by_vid = df[df['id'].isin(veh_id)]


        position_data = df_by_vid[['x', 'y']].values.tolist()
        velocity_data = df_by_vid[['speed']].values.tolist()
        self.position_data = position_data


        initial_position = position_data[0 : self.n_Veh]
        initial_velocity = velocity_data[0 : self.n_Veh]


        for i in range(self.n_Veh):
            self.V2V_add_new_vehicles(initial_position[i], velocity_data[i][0])



        # -------------V2I -------------------

        # start_position = [250, -4.8]
        # start_velocity = 10
        # for i in range(self.n_SC):
        #     start_position[0] = start_position[0] + i * self.distance_between_V2I_veh
        #     self.V2I_add_new_vehicles(start_position.copy(), start_velocity)
        #     # print("start_position: ", start_position)


        # self.V2I_add_new_vehicles([312, -1.6], 10)
        # self.V2I_add_new_vehicles([250, -4.8], 10)
        # self.V2I_add_new_vehicles([283, -4.8], 10)
        # self.V2I_add_new_vehicles([290, -8], 10)

        veh_id = []
        # num_vehicles = 4  # Set the number of vehicles you want

        for j in range(self.n_SC):
            veh_id.append(f"carflowV2I_{j}.0")



        df_by_vid = df[df['id'].isin(veh_id)]


        position_data = df_by_vid[['x', 'y']].values.tolist()
        velocity_data = df_by_vid[['speed']].values.tolist()
        self.position_data = position_data


        initial_position = position_data[0 : self.n_Veh]
        initial_velocity = velocity_data[0 : self.n_Veh]


        for i in range(self.n_SC):
            self.V2I_add_new_vehicles(initial_position[i], velocity_data[i][0])







    def renew_positions(self, control=False, old_state_p=None):
        """This function updates the position of each vehicle"""

        # All vehicles move at a constant speed
        if not control:
            i = 0
            while i < self.n_Veh:
                delta_distance = self.vehicles_V2V[i].velocity * self.time_slow
                self.vehicles_V2V[i].position[0] += delta_distance
                i += 1
        else:
            i = 0
            while i < self.n_Veh and old_state_p[i] is not None:
                self.vehicles_V2V[i].position[0] = old_state_p[i]
                self.vehicles_V2V[i].velocity = old_state_p[i + 5]
                # print(self.vehicles_V2V[i].velocity)
                i += 1


    def renew_positions_by_file(self, time):
        """This function updates the position of each vehicle"""
        position_data = self.position_data
        current_position = position_data[(time - 1) * self.n_Veh : self.n_Veh + (time - 1) * self.n_Veh]


        i = 0
        while i < self.n_Veh:
            # print(i)
            self.vehicles_V2V[i].position = current_position[i]
            i += 1


        # tm.sleep(5)


    def set_positions(self, coordinates):
        """
        For Normal-Form Interference Game
        """
        for i in range(self.n_Veh):
            self.vehicles_V2V[i].position = coordinates[i]



    def renew_neighbor(self):
        """
        Determine the neighbors of each vehicle
        """

        for i in range(self.n_agent):
            veh = self.agent2veh[i]
            self.vehicles_V2V[veh].neighbors = []
            self.vehicles_V2V[veh].actions = []

        for i in range(self.n_agent):
            veh = self.agent2veh[i]
            for j in range(self.n_neighbor):
                self.vehicles_V2V[veh].neighbors.append(veh - 1)
            destination = self.vehicles_V2V[veh].neighbors

            self.vehicles_V2V[veh].destinations = destination


        # for i in range(self.n_agent):
        #     veh = self.agent2veh[i]
        #     print("Agent id: ", i, " veh id: ", veh)
        #     print("Agent neibor:        ", self.vehicles_V2V[veh].neighbors)
        #     print("Agent destination:   ", self.vehicles_V2V[veh].destinations)





    def renew_channel(self):
        """
        Renew slow fading channel
        """

        # ------------   interference channel from V2I to V2V  ---------------------------
        V2I_pathloss_V2V = np.zeros((self.n_SC, self.n_Veh))
        self.V2I_pathloss_V2V = V2I_pathloss_V2V
        for SC_idx in range(self.n_SC):
            for i in range(self.n_Veh):
                self.V2I_pathloss_V2V[SC_idx][i] = self.V2V_get_path_loss(self.vehicles_V2V[i].position,
                                                                          self.vehicles_V2I[SC_idx].position)
        # ------------- channel form V2I vehicles to eNB ---------------------------
        self.V2I_pathloss_eNB = np.zeros(self.n_SC)
        for i in range(self.n_SC):
            self.V2I_pathloss_eNB[i] = self.V2I_get_path_loss(self.vehicles_V2I[i].position)

        # --------------channel from V2V-Tx to BS-------------------
        self.V2V_pathloss_eNB = np.zeros(self.n_Veh)
        for i in range(self.n_Veh):
            self.V2V_pathloss_eNB[i] = self.V2I_get_path_loss(self.vehicles_V2V[i].position)


        # ------------   channel from V2V to V2V  ---------------------------
        V2V_pathloss_V2V = np.zeros((self.n_Veh, self.n_Veh))  # 自己到自己的 pathloss 应该是0？
        self.V2V_pathloss_V2V = V2V_pathloss_V2V
        for i in range(self.n_Veh):
            for j in range(i + 1, self.n_Veh):
                self.V2V_pathloss_V2V[i][j] = self.V2V_pathloss_V2V[j][i] = self.V2V_get_path_loss(
                    self.vehicles_V2V[i].position, self.vehicles_V2V[j].position)





    def Compute_Received_Power_at_per_SC(self, actions_power):
        # RSRP: Received Power strength at each Sub-channel
        Veh_RSRP_per_SC = np.zeros((self.n_agent, self.n_SC)) + self.sig2

        channel_selection = actions_power.copy()[:, :, 0]
        power_selection = actions_power.copy()[:, :, 1]



        # interference from peer V2V links
        for sender in range(self.n_agent):
            veh_sender = self.agent2veh[sender]
            for sender_nbr in range(self.n_neighbor):

                # the index of receiver corresponding to the Vehicle #i
                for receiver in range(self.n_agent):
                    veh_receiver = self.agent2veh[receiver]
                    for rcver_nbr in range(self.n_neighbor):

                        # if i == k or channel_selection[i,j] >= 0:
                        # vehicles_V2V uses veh_id as index; channel_selection uses agent_id as index
                        if veh_sender == veh_receiver or self.vehicles_V2V[veh_receiver].destinations[rcver_nbr] == veh_sender or \
                                channel_selection[sender, sender_nbr] < 0:
                            continue


                        Veh_RSRP_per_SC[receiver, channel_selection[sender, sender_nbr]] += 10 ** (
                                (self.V2V_power_dB_List[power_selection[sender, sender_nbr]]
                                - self.V2V_pathloss_V2V[sender][self.vehicles_V2V[veh_receiver].destinations[rcver_nbr]] +
                                2 * self.vehAntGain - self.vehNoiseFigure) / 10)


        self.Veh_RSRP_per_SC_all_dBm = 10 * np.log10(Veh_RSRP_per_SC)  # in dBm



    def Compute_Performance_Reward(self, actions_power):
        """
        Calculate the communication rate without multiplying by bandwidth (when calculating the reward, there is no need
        to divide by bandwidth)
        """

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # print("actions_power: ", actions_power)
        # print("actions: ", actions)
        # print("power_selection: ", power_selection)




        # ------------ Compute interference and date rate of V2I links -------------------------
        V2V_Interference_V2I = np.zeros(self.n_SC)  # V2I interference
        for i in range(self.n_agent):
            for j in range(self.n_neighbor):

                if actions[i][j] != -1:
                    V2V_Interference_V2I[actions[i][j]] += 10 ** (
                            (self.V2V_power_dB_List[power_selection[i, j]] - self.V2V_pathloss_eNB[i]
                             + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2V_Interference_V2I = V2V_Interference_V2I + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_pathloss_eNB + self.vehAntGain + self.bsAntGain -
                              self.bsNoiseFigure) / 10)
        self.V2I_Signals = V2I_Signals
        V2I_SE = np.log2(1 + np.divide(V2I_Signals, self.V2V_Interference_V2I))
        self.V2I_SE = V2I_SE
        self.V2I_SE_ideal = np.log2(1 + np.divide(V2I_Signals, self.sig2))

        # print("self.V2I_SE: ", self.V2I_SE)
        # print("self.V2I_SE_ideal: ", self.V2I_SE_ideal)



        # ------------ Compute V2V rate -------------------------

        tot_V2V_Interference = np.zeros((self.n_agent, 1))
        V2V_Interference_at_Rx = np.zeros((self.n_agent, self.n_agent))
        V2V_Interference_to_others = np.zeros((self.n_agent, self.n_agent))
        V2I_interference_at_Rx = np.zeros((self.n_agent, 1))
        V2V_Signal = np.zeros((self.n_agent, 1))
        # ----------------------


        # Agent i ---> veh j
        for i in range(self.n_agent):

            sender = i                                  # sender is the V2V agent_id
            veh_sender = self.agent2veh[i]              # veh_id is the vehicle id of Agent

            for j in range(self.n_neighbor):

                if actions[sender][j] != -1:

                    veh_receiver = self.vehicles_V2V[veh_sender].destinations[0]

                    V2V_Signal[sender][0] = 10 ** ((self.V2V_power_dB_List[power_selection[sender, j]]
                                               - self.V2V_pathloss_V2V[
                                                   veh_sender, veh_receiver] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                    # -----------------------interference from V2I link to target V2V link---------------------
                    V2I_interference_at_Rx[sender][0] = 10 ** ((self.V2I_power_dB - self.V2I_pathloss_V2V[      # interference power from a V2I transmitter to the V2V receiver of "sender(i)"
                        actions[sender][j], veh_receiver] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)      # pathloss between the V2I transmitter selected by the action ([sender][j]) and the receiver of the V2V link (receiver_j
                    tot_V2V_Interference[sender][0] += V2I_interference_at_Rx[sender][0] 


                    # ---------------------- V2V 对 V2V 的干扰 --------------------------------------
                    for x in range(self.n_agent):
                        # other_sender = self.agent2veh[x]          # sender is the veh_id of the Agent
                        other_sender = x
                        veh_other_sender = self.agent2veh[i]              # veh_id is the vehicle id of Agent

                        if x != i:
                            if actions[sender][j] == actions[other_sender][j]:

                                V2V_Interference_at_Rx[sender][other_sender] = 10 ** ((self.V2V_power_dB_List[power_selection[other_sender, j]] -
                                                                       self.V2V_pathloss_V2V[veh_other_sender, veh_receiver]
                                                                       + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                                # interference from transmitter Agent x to receiving side of Agent i
                                tot_V2V_Interference[sender][0] += V2V_Interference_at_Rx[sender][other_sender]
                                V2V_Interference_to_others[other_sender][sender] = V2V_Interference_at_Rx[sender][other_sender]




        # Calculate the V2V interference in current time slot
        self.total_V2V_Interference = tot_V2V_Interference + self.sig2
        self.total_V2V_Interference_in_dBm = 10 * np.log10(self.total_V2V_Interference)
        V2V_SE = np.log2(1 + np.divide(V2V_Signal, self.total_V2V_Interference))
        self.V2V_SE = V2V_SE
        self.V2V_Interference_to_others = V2V_Interference_to_others
        self.V2V_Interference_at_Rx = V2V_Interference_at_Rx


        self.queue -= (V2V_SE * self.time_fast * self.bandwidth / self.CAM_size)
        self.queue[self.queue <= 0] = 0  # eliminate negative queue




        """
        Calculate Reward function by differentiate approach
        """


        reward_V2V = np.zeros((self.n_agent, 1))
        V2V_SE_losses = np.zeros((self.n_agent, 1))
        V2I_SE_losses = np.zeros((self.n_SC, 1))


        for i in range(self.n_agent):

            sender = i                              # sender is the agemt_id
            if self.queue[sender] <= 0:
                reward_V2V[sender][0] = 1.0

            else:
                V2V_SE_losses = np.log2(1 + np.divide(V2V_Signal, (self.total_V2V_Interference -
                                                                   np.reshape(self.V2V_Interference_to_others[sender, :],
                                                                              (-1, 1))))) \
                                - V2V_SE
                reward_V2V[sender][0] = min(max((V2V_SE[sender][0] - self.lambda1 * np.sum(V2V_SE_losses)) / 10, -1), 1)


        self.individual_rewards = reward_V2V


        # print("V2I_interference_at_Rx: ", V2I_interference_at_Rx)


        return V2V_SE, V2I_SE, self.queue, reward_V2V




    def step(self, actions, t, interval):
        """
            Calculate reward (global reward or individual reward)
        """
        action_temp = actions.copy()


        # Calculation Process
        V2V_SE, V2I_SE, queue, reward_V2V = self.Compute_Performance_Reward(action_temp)
        self.Compute_Received_Power_at_per_SC(action_temp)
        self.renew_active_links()



        # reward is sum of data rate
        if self.game_mode == 1:
            if t == self.n_step_per_episode_communication - 1:
                done = True
            else:
                done = False

            global_reward = np.array([[np.sum(V2V_SE)]])
            
            return global_reward, V2V_SE, V2I_SE, done

        # reward is sum of data rate + G
        elif self.game_mode == 2:
            individual_rewards = np.zeros_like(self.queue)

            for i in range(self.queue.shape[0]):
                if self.queue[i][0] <= 0:
                    individual_rewards[i][0] = self.reward_G
                else:
                    individual_rewards[i][0] = V2V_SE[i][0] * 0.01

            if t == self.n_step_per_episode_communication - 1:
                done = True
            else:
                done = False

            global_reward = np.array([[np.sum(individual_rewards)]])
            return global_reward, individual_rewards, V2I_SE, done

        # reward is sum of data rate - AoI
        elif self.game_mode == 3 or 4:

            individual_rewards =  -self.AoI * self.lambda3 + V2V_SE * self.lambda2


            # if np.any(self.AoI > 1):
            #     print("AoI: ")
            #     print(self.AoI)
            #     print("V2V_SE: ")
            #     print(V2V_SE)
            #     print("global_reward, individual_ag_rewards: ")
            #     print(np.array([[np.sum(individual_rewards)]]), individual_rewards)
            #     print("queue length: ")
            #     print(self.queue)

            if t == self.n_step_per_episode_communication - 1 and interval == self.t_max_control:
                done = True
            else:
                done = False

            global_reward = np.array([[np.sum(individual_rewards)]])

            return global_reward, individual_rewards, V2I_SE, done


        else:
            print("Error: game mode doesn't exist")
            sys.exit(1)




    def new_random_game(self):
        # make a new game
        self.vehicles_V2V = []


        if self.n_Veh > 0:
            # self.add_new_vehicles_by_number()
            self.add_new_vehicles_by_file(self.loaded_veh_data)
            self.renew_neighbor()
            self.renew_channel()
        else:
            print('Error!!!!')
        self.queue = np.ones((self.n_agent, self.n_neighbor))
        self.active_links = np.ones((self.n_agent, self.n_neighbor), dtype='bool')
        self.AoI = np.ones((self.n_agent, self.n_neighbor))



    def renew_queue(self):

        self.AoI[self.queue > 0] += 1
        self.AoI[self.queue <= 0] = 1


        self.queue = np.ones((self.n_agent, self.n_neighbor))
        for ag_idx in range(self.n_agent):
            if self.queue[ag_idx][0] > self.NQ:
                self.queue[ag_idx][0] = self.NQ

    def renew_active_links(self):
        for ag_idx in range(self.n_agent):
            if self.queue[ag_idx, 0] <= 0:
                self.active_links[ag_idx, 0] = False
            else:
                self.active_links[ag_idx, 0] = True

    def return_avail_actions(self):
        ag_avail_actions_list = []
        for ag_idx in range(self.n_agent):
            if self.active_links[ag_idx, 0]:
                ag_avail_actions_temp = np.ones(self.num_of_actions)
                ag_avail_actions_list.append(ag_avail_actions_temp)
            else:
                ag_avail_actions_temp = np.zeros(self.num_of_actions)
                ag_avail_actions_temp[-1] = 1.0
                ag_avail_actions_list.append(ag_avail_actions_temp)
        return ag_avail_actions_list




    def get_state(self, idx, epsi, i_step):

        state = None

        if self.game_mode == 1:
            state = self.get_simple_state(idx, epsi, i_step)

        elif self.game_mode == 2:
            state = self.get_global_state(idx, epsi, i_step)

        elif self.game_mode == 3:
            state = self.get_global_state_AoI(idx, epsi, i_step)

        elif self.game_mode == 4:
            state = self.get_observation(idx, epsi, i_step)

        else:
            print("Error: game mode doesn't exist")
            sys.exit(1)


        # """
        # check for termination
        # """
        # if self.game_mode == 1 or self.game_mode == 2:
        #     if i_step == self.n_step_per_episode_communication - 1:
        #         done = 1
        #     else:
        #         done = 0

        # else:
        #     if i_step == self.n_step_per_episode_communication - 1 and time == self.t_max_control:
        #         done = 1
        #     else:
        #         done = 0


        return state


    # Channel Aware State - NFIG
    def get_simple_state(self, idx, epsi, i_step):


        state = np.array([])
        G_i = np.array([])
        G_ji = np.array([])

        """
        Channel gain: G_i
        idx: [ag_idx, 0] now assuming only 1 receiver, "0" means the only index in destination
        """
        for i in range(self.n_agent):

            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
            V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                     / self.norm_V2V_channel_factor])
            G_i = np.hstack((G_i, V2V_large_scale_channel_norm))


        """
        Intereference channel gain: G_ji
        """
        # V2V_Interference_at_Rx[sender][other_sender]
        for i in range(self.n_agent):
            for j in range(self.n_agent):

                # agents' intereference to themselves are 0 and are ignored in the state
                if i == j:
                    continue

                # if self.V2V_Interference_at_Rx is not None:
                #     G_ji = np.hstack((G_ji, self.V2V_Interference_at_Rx[i][j]))
                # else:
                #     G_ji = np.hstack((G_ji, np.zeros(1)))


                veh_sender = self.agent2veh[i]                                              # the interference link sender j
                veh_receiver = self.vehicles_V2V[self.agent2veh[j]].destinations[idx[1]]    # the receiver of self.agent2veh[i]
                V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                                     / self.norm_V2V_channel_factor])


                G_ji = np.hstack((G_ji, V2V_large_scale_channel_norm))



        for state_info in [G_i, G_ji]:
            # state_info = self.normalize_state(state_info)
            state = np.hstack((state, state_info))


        state = state.reshape((1, self.stateDim))

        return state


    # Channel & Queue Aware State - SIG 
    def get_global_state(self, idx, epsi, i_step):


        state = np.array([])

        t = np.array([])
        G_i = np.array([])
        G_ji = np.array([])
        queue_length = np.array([])


        """
        t: one hot coded communication interval
        """
        t = np.zeros(self.n_step_per_control_interval)
        t[min(i_step, self.n_step_per_control_interval - 1)] = 1



        """
        Channel gain: G_i
        idx: [ag_idx, 0] now assuming only 1 receiver, "0" means the only index in destination
        """
        for i in range(self.n_agent):

            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]


            V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                     / self.norm_V2V_channel_factor])
            G_i = np.hstack((G_i, V2V_large_scale_channel_norm))


        """
        Intereference channel gain: G_ji
        """
        # V2V_Interference_at_Rx[sender][other_sender]
        for i in range(self.n_agent):
            for j in range(self.n_agent):

                # agents' intereference to themselves are 0 and are ignored in the state
                if i == j:
                    continue

                # if self.V2V_Interference_at_Rx is not None:
                #     G_ji = np.hstack((G_ji, self.V2V_Interference_at_Rx[i][j]))
                # else:
                #     G_ji = np.hstack((G_ji, np.zeros(1)))


                veh_sender = self.agent2veh[i]                                              # the interference link sender j
                veh_receiver = self.vehicles_V2V[self.agent2veh[j]].destinations[idx[1]]    # the receiver of self.agent2veh[i]
                V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                                     / self.norm_V2V_channel_factor])


                G_ji = np.hstack((G_ji, V2V_large_scale_channel_norm))


        """
        queue length by the end of the last communication interval: q_i
        """

        queue_length = self.queue / self.NQ
        queue_length = queue_length.flatten()      

        for state_info in [t, G_i, G_ji, queue_length]:

            # state_info = self.normalize_state(state_info)
            # print("state_info: ", state_info)
            
            state = np.hstack((state, state_info))


        state = state.reshape((1, self.stateDim))

        return state



    # Channel & Queue & AoI Aware State - Not Implemented
    def get_global_state_AoI(self, idx, epsi, i_step):


        state = np.array([])

        t = np.array([])
        G_i = np.array([])
        G_ji = np.array([])
        queue_length = np.array([])
        AoI = np.array([])


        """
        one hot coded communication interval: t
        """
        t = np.zeros(self.n_step_per_control_interval)
        t[min(i_step, self.n_step_per_control_interval - 1)] = 1

        # state = i_step_one_hot_vec.copy()


        """
        Channel gain: G_i
        idx: [ag_idx, 0] now assuming only 1 receiver, "0" means the only index in destination
        """
        for i in range(self.n_agent):

            veh_sender = self.agent2veh[i]
            veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
            V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                     / self.norm_V2V_channel_factor])
            G_i = np.hstack((G_i, V2V_large_scale_channel_norm))


        """
        Intereference channel gain: G_ji
        """
        # V2V_Interference_at_Rx[sender][other_sender]
        for i in range(self.n_agent):
            for j in range(self.n_agent):

                # agents' intereference to themselves are 0 and are ignored in the state
                if i == j:
                    continue

                # if self.V2V_Interference_at_Rx is not None:
                #     G_ji = np.hstack((G_ji, self.V2V_Interference_at_Rx[i][j]))
                # else:
                #     G_ji = np.hstack((G_ji, np.zeros(1)))


                veh_sender = self.agent2veh[i]                                              # the interference link sender j
                veh_receiver = self.vehicles_V2V[self.agent2veh[j]].destinations[idx[1]]    # the receiver of self.agent2veh[i]
                V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                                     / self.norm_V2V_channel_factor])


                G_ji = np.hstack((G_ji, V2V_large_scale_channel_norm))


        """
        queue length by the end of the last communication interval: q_i
        """
        queue_length = self.queue / self.NQ
        queue_length = queue_length.flatten()


        """
        Age of information: τ_i
        """
        AoI = self.AoI
        AoI = AoI.flatten()


        for state_info in [t, G_i, G_ji, queue_length, AoI]:
            # state_info = self.normalize_state(state_info)
            state = np.hstack((state, state_info))


        state = state.reshape((1, self.stateDim))

        return state

    # Partial Observability - POSIG
    def get_observation(self, idx, epsi, i_step):


        state = np.array([])

        t = np.array([])
        G_i = np.array([])
        G_ji = np.array([])
        queue_length = np.array([])
        AoI = np.array([])


        """
        one hot coded communication interval: t
        """
        t = np.zeros(self.n_step_per_control_interval)
        t[min(i_step, self.n_step_per_control_interval - 1)] = 1



        """
        Channel gain: G_i
        idx: [ag_idx, 0] now assuming only 1 receiver, "0" means the only index in destination
        """
        veh_sender = self.agent2veh[idx[0]]
        veh_receiver = self.vehicles_V2V[veh_sender].destinations[idx[1]]
        V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                     / self.norm_V2V_channel_factor])
        G_i = np.hstack((G_i, V2V_large_scale_channel_norm))


        """
        Intereference channel gain: G_ji
        """
        # V2V_Interference_at_Rx[sender][other_sender]
        for i in range(self.n_agent):
            # agents' intereference to themselves are 0 and are ignored in the state
            if i == idx[0]:
                continue

            # if self.V2V_Interference_at_Rx is not None:
            #     state = np.hstack((state, self.V2V_Interference_at_Rx[idx[0]][j]))
            # else:
            #     state = np.hstack((state, np.zeros(1)))
            veh_sender = self.agent2veh[i]                                              # the interference link sender j
            veh_receiver = self.vehicles_V2V[self.agent2veh[idx[0]]].destinations[idx[1]]    # the receiver of self.agent2veh[i]
            V2V_large_scale_channel_norm = np.array([(self.V2V_pathloss_V2V[veh_sender, veh_receiver])
                                                                 / self.norm_V2V_channel_factor])


            G_ji = np.hstack((G_ji, V2V_large_scale_channel_norm))


        """
        queue length by the end of the last communication interval: q_i
        """
        queue_length = self.queue[idx[0], idx[1]] / self.NQ

        

        """
        Age of information: τ_i
        """
        AoI = self.AoI[idx[0], idx[1]]


        for state_info in [t, G_i, G_ji, queue_length]:
            # state_info = self.normalize_state(state_info)
            # print(state_info)
            state = np.hstack((state, state_info))

        state = state.reshape((1, self.local_stateDim))



        return state
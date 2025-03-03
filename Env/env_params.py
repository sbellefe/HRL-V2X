
def parse_agent2veh(values):
    """
    Parses a list of strings in the form "key=value" into a dictionary
    where key and value are both integers.
    
    E.g. ["0=10", "1=11"] -> {0: 10, 1: 11}
    """
    result = {}
    for item in values:
        try:
            key_str, value_str = item.split('=')
            result[int(key_str)] = int(value_str)
        except ValueError:
            raise ValueError(f"Could not parse argument '{item}'. "
                             f"Expected format is key=value with both key/value as integers.")
    return result

class V2Xparams:
    def __init__(self, agent2veh_list=None):
        
        # NFIG: 25.0, 30.0, 35.0, 65.0
        # SIG ML: "ML_FF", "ML_NFF"
        # SIG SL: ["SL_FF", 30.0], ["SL_NFF", 30.0]
        # POSIG: "ML_FF_PO"
        self.env_setup = "ML_FF_PO"

        # Data and Model Parameters
        self.data_dir = 'dataset/path/here/'    # the path of dataset
        self.CM_model_save_dir = 'model/save/path/here' # the path of saving model
        self.CM_log_file_dir = 'logging/save/path/here' # the path of saving logging record
        self.CL_model_save_dir = 'model/save/path/here' # the path of saving model
        self.CL_log_file_dir = 'logging/save/path/here' # the path of saving logging record
        self.Baseline_CM_model_save_dir = 'baseline/model/save/path/here'   # the path of saving model
        self.Baseline_CM_log_file_dir = 'baseline/logging/save/path/here'   # the path of saving logging record
        self.Baseline_CL_model_save_dir = 'baseline/model/save/path/here'   # the path of saving model
        self.Baseline_CL_log_file_dir = 'baseline/logging/save/path/here'   # the path of saving logging record
        self.Baseline_log_file_dir = 'baseline/logging/save/path/here'  # the path of saving logging record
        self.Pro_CM_model_save_dir = 'model/save/path/here' # the path of saving model
        self.Pro_CM_log_file_dir = 'logging/save/path/here' # the path of saving logging record
        self.Pro_CL_model_save_dir = 'model/save/path/here' # the path of saving model
        self.Pro_CL_log_file_dir = 'logging/save/path/here' # the path of saving logging record
        self.Pro_log_file_dir = 'logging/save/path/here'    # the path of saving logging record
        self.V2I_V2V_scenario_state_type = 'normal_version' # definition of state
        self.CL_ideal_model_save_dir = 'model/save/path/here'   # the path of saving model
        self.CL_ideal_log_file_dir = 'logging/save/path/here'   # the path of saving logging record


        # Driving Scenario Parameters
        self.n_veh = 0                          # number of vehicles in total
        self.n_veh_platoon = [2, 2, 2, 2]       # number of vehicles in each platoon    
        self.null_ID = set()                    # number of vehicles in the platoon
        self.n_lane = 1                         # number of lanes in the platoon
        self.number_of_SC = 4                   # number of sub-channels (SCs) for V2X communications
        self.n_neighbor = 1                     # number of neighboring vehicles need to receive CAM
        self.n_agent = 0                        # number of agents
        if agent2veh_list is not None:
            self.agent2veh = parse_agent2veh(agent2veh_list)    # mapping agent index to vehicle
        else:
            self.agent2veh = {}

        # Control Parameters
        self.nb_episodes_control = 100000       # number of control episodes
        self.t_max_control = 10                  # maximum number of time for control
        self.nb_episodes_Test = 2               # number of Test episodes
        self.NQ = 1                             # number of buffer size at each agent
        self.CAM_size = 25600                   # number of bits per cooperative awareness message (''CAM), in bits'
        self.gamma_control = 0.99979            # discount factor of CL part
        self.CL_pretrained = False              # using locally pretrained model
        self.AoI_max = 6                        # maximum value of AoI at each agent
        self.track_er_position_init = 1.5       # initialization of tracking position error
        self.track_er_velocity_init = -1.0      # initialization of tracking velocity error
        self.er_position_norm_range = 2.0       # normalization of position tracking error
        self.er_velocity_norm_range = 1.5       # normalization of velocity tracking error
        self.acceleration_norm_range = 2.6      # normalization of acceleration
        

        # Communication Parameters
        self.game_mode = 2                      # 1: channel aware
                                                # 2: channel and queue aware
                                                # 3: channel, queue and AoI aware
                                                # 4: partial observability
        self.n_step_per_episode_communication = 10   # number of communication time intervals within each control time unit
        self.BW_per_SC = 1000000                # bandwidth of each sub-channel, in Hz                                                        
        self.reward_w1 = 0.001                  # reward weight corresponding to V2I         
        self.reward_w2 = 0.1                    # reward weight corresponding to V2V data rate
        self.reward_w3 = 1                      # reward weight corresponding to AoI
        self.reward_VoI = 100.0                 # reward weight corresponding to V2V data rate
        self.reward_G = 5.0                     # reward weight corresponding to V2V data rate
        self.QMS = 2                            # queue management strategies (QMS) = 1 OR 2
        

        # Optical Parameters
        self.Target_Neural_Network_update_mode = 'hard update'  # definition of state
        self.Soft_update_ag_factor = 0.005      # soft update of the target network's weights at agent side
        self.hard_target_update_freq = 4000     # definition of state
        self.Gamma_communication = 1.0          # discount factor of CM part
        self.pretrained = False                 # using locally pre-trained model. The path of pre-trained model should be given
        self.Batch_size = 64                    # Batch size of learning
        self.CL_Batch_size = 64                 # Batch size of CL learning
        self.learning_rate_init_local_agent = 5e-4  # Initial learning rate at agent
        self.learning_rate_init_center_critic = 1e-4    # Initial learning rate at center
        

        # V2X Communication Parameters
        self.V2V_power_dBm_List = [23, 15, 5]   # power levels at the PM vehicle side
        self.Platoon_Leader_V2V_power_dBm_List = [20]   # power levels at the PL side
        self.sig2_dB = -114                     # power leve of AWGN noise
        self.norm_V2V_channel_factor = 120      # Normalization factor for V2V channel
        self.norm_V2V_interference_factor = 80  # Normalization factor for V2V interference
        self.loaded_veh_data = None             # Loaded vehicle positions/speed from csv file
        
        
        # V2V Control Parameters
        self.track_er_position_maxn = 10        # nominal maximum tracking error position
        self.track_er_velocity_maxn = 10        # nominal maximum tracking error velocity
        self.control_input_max = 2.9            # maximum value of control input
        self.control_input_min = -4.3           # nominal value of control input
        
        
        # V2X Communication
        self.bsAntGain = 8                      # Base station antenna gain
        self.bsNoiseFigure = 5                  # Base station noise figure
        self.vehAntGain = 3                     # Vehicle antenna gain
        self.vehNoiseFigure = 9                 # Vehicle noise figure
        self.decorrelation_distance = 10        # Decorrelation distance
        self.shadow_std = 3                     # Standard deviation for shadowing
        self.if_fastFading = True              # Standard deviation for shadowing


        # ML
        self.seed = 1                           # seed for initializing training
        self.gpu = None                         # GPU id to use
        self.if_Turn_on_immediate_display = False   # GPU id to use


        # Calculated Parameters

        # self.V2I_V2V_scenario_state_type == 'normal_version'
        self.num_pw_levels = len(self.V2V_power_dBm_List)
        if self.V2I_V2V_scenario_state_type == 'simplified_version':
            self.num_actions = self.num_pw_levels + 1
            self.number_of_SC = self.n_veh - 1
            # args.number_of_SC = 1
        else:
            self.num_actions = self.num_pw_levels * self.number_of_SC + 1
        self.nb_episodes_communication = self.nb_episodes_control * self.t_max_control
        self.memory_buffer_length = min(int(self.nb_episodes_communication * self.n_step_per_episode_communication / 3), 100000)
        self.Cl_memory_buffer_length = min(int(self.nb_episodes_control * self.t_max_control / 2), 200000)

        # Define null_ID

        # non_ag_offset = 0
        # for p in range(len(args.n_veh_platoon)):
        #     n_veh = args.n_veh_platoon[p]
        #     for v in range(n_veh):
        #         if v == n_veh - 1:    # Add the last vehicle
        #             args.null_ID.add(v + non_ag_offset)
                    
        #     non_ag_offset += n_veh

        non_ag_offset = 0
        for p in range(len(self.n_veh_platoon)):
            n_veh = self.n_veh_platoon[p]
            for v in range(n_veh):
                if v == 0:  # Add the first vehicle
                    self.null_ID.add(v + non_ag_offset)
                    
            non_ag_offset += n_veh

        # Calculate n_veh based on Platoon

        self.n_veh = sum(self.n_veh_platoon)
        self.n_agent = sum(self.n_veh_platoon) - len(self.null_ID)

        # Agent --> Vehicle ID mapping

        self.agent2veh = {}
        agent_index = 0
        for veh_idx in range(self.n_veh):
            if veh_idx not in self.null_ID:
                self.agent2veh[agent_index] = veh_idx
                agent_index += 1
            else:
                continue
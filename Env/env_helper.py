import numpy as np

class EnvironHelper:

    def __init__(self, params):
        self.state_type = params.V2I_V2V_scenario_state_type
        self.n_pw_levels = params.num_pw_levels
        self.action_dim = params.num_actions
    
    def mapping_action2RRA(self, action):

        if self.state_type == 'simplified_version':


            if action < self.action_dim - 1:
                # convert action index to power allocation and SC allocation
                SC_index = self.ag_idx # self.n_SC - self.ag_idx - 1
                Power_level_index = action % self.n_pw_levels
            else:
                SC_index = -1
                Power_level_index = -1
        else:

            if action < self.action_dim - 1:
                # convert action index to power allocation and SC allocation
                SC_index = int(np.floor(action.cpu().numpy() / self.n_pw_levels))
                Power_level_index = action % self.n_pw_levels
            else:
                SC_index = -1
                Power_level_index = -1
                
        return SC_index, Power_level_index
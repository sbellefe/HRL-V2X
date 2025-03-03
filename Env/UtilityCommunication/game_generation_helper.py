import numpy as np
import pandas as pd
import pickle
from collections import deque
from utils_parser.parameters_interference import args


# action_dim = args.action_dim
# agent_number = args.n_veh - 1               # agent_number will include None Agent




def generate_actions_with_none(agent_idx, current_action, all_actions, action_dim, agent_number, null_ID):
    if agent_idx == agent_number:
        all_actions.append(current_action.copy())
        return
    if agent_idx in null_ID:
        # Agent 1's action is always None
        current_action[agent_idx] = None
        generate_actions_with_none(agent_idx + 1, current_action, all_actions, action_dim, agent_number, null_ID)
    else:
        for action in range(action_dim):
            current_action[agent_idx] = action
            generate_actions_with_none(agent_idx + 1, current_action, all_actions, action_dim, agent_number, null_ID)



def enumerate_all_actions_with_none(action_dim, agent_number, null_ID):
    all_actions = []
    current_action = [0] * agent_number
    generate_actions_with_none(0, current_action, all_actions, action_dim, agent_number, null_ID)
    return all_actions



def create_buffer(file_path):

    df = pd.read_csv(file_path)
    # Count the number of vehicles
    time_counts = df['time'].value_counts()
    n_veh = time_counts.max()

    # Find all time points with exactly four duplicates
    time_groups = df.groupby('time').filter(lambda x: len(x) == n_veh)
    
    # Create a deque to store the data points with four duplicates
    time_deque = deque()
    
    # Populate the deque
    grouped = time_groups.groupby('time')
    for name, group in grouped:
        time_deque.append((name, group))

    return time_deque



# Convert the action combinations to integer indices
def convert_actions_to_int(actions, action_to_int):
    return tuple(action_to_int[action] for action in actions)





def export_reward_matrix(reward_dict, export_file_name):
	# Save the dictionaries to a file
	with open(export_file_name, 'wb') as f:
	    pickle.dump(reward_dict, f)










import numpy as np
import pandas as pd
import random
import sys
import scipy.stats as stats


# loading vehicle postions over a time period. The time period will be greater than
# nb_episodes_control * t_max_control * n_step_per_episode_communication
# nb_episodes_control * (120 control intervals by default) * [50 comm intervals (50ms)]

def load_veh_pos(file_name):
    file_path = file_name
    data = pd.read_csv(file_path)

    return data


import numpy as np
import random
import sys

# This will sample [t_max_control] number of timesteps
# Used for games with continuous control intervals
def sample_veh_positions(t_max_control, data):
    # Get the number of unique time steps in the data
    unique_time_steps = data['time'].nunique()

    # Check if there are enough time steps to sample
    if t_max_control > unique_time_steps:
        print("Error: not enough time steps to sample")
        sys.exit(1)

    # Get the unique time steps in sorted order
    sorted_time_steps = data['time'].drop_duplicates().sort_values()

    # Randomly select a starting index for sampling a block of timesteps
    start_index = np.random.randint(0, unique_time_steps - t_max_control + 1)

    # Select the successive time steps starting from the random start index
    sampled_time_steps = sorted_time_steps.iloc[start_index:start_index + t_max_control]

    # Filter the DataFrame to include only the rows with the selected time steps
    sampled_data = data[data['time'].isin(sampled_time_steps)]

    return sampled_data


# This will sample 1 timestep
# Used for NFIG and queue-aware environments
def sample_veh_position_single(data):
    # Get the number of unique time steps in the data
    unique_time_steps = data['time'].nunique()

    # Check if there are any time steps to sample
    if unique_time_steps == 0:
        print("Error: no time steps to sample")
        sys.exit(1)

    # Get the unique time steps in sorted order
    sorted_time_steps = data['time'].drop_duplicates().sort_values()

    # Randomly select one time step
    sampled_time_step = np.random.choice(sorted_time_steps)

    # Filter the DataFrame to include only the rows with the selected time step
    sampled_data = data[data['time'] == sampled_time_step]

    return sampled_data


def sample_veh_position_from_timestep(data, time_step):
    # Check if the provided time step exists in the data
    if time_step not in data['time'].unique():
        print(f"Error: time step {time_step} not found in the data")
        return None

    # Filter the DataFrame to include only the rows with the provided time step
    sampled_data = data[data['time'] == time_step]

    return sampled_data

# Remove all test data from veh_pos_data
def remove_test_data_from_veh_pos(veh_pos_data, time_steps):
    # Filter veh_pos_data to exclude rows with time in time_steps
    filtered_data = veh_pos_data[~veh_pos_data['time'].isin(time_steps)]
    
    return filtered_data

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


def generate_actions(agent_idx, current_action, all_actions, action_dim, agent_number):
    if agent_idx == agent_number:
        all_actions.append(current_action.copy())
        return
    # Iterate over all possible actions for the current Agent
    for action in range(action_dim):
        current_action[agent_idx] = action
        generate_actions(agent_idx + 1, current_action, all_actions, action_dim, agent_number)

def enumerate_all_actions(action_dim, agent_number):
    all_actions = []
    current_action = [0] * agent_number
    generate_actions(0, current_action, all_actions, action_dim, agent_number)
    return all_actions



def calculate_max_mean_and_ci(data, confidence=0.95):
    """
    Calculate the maximum mean result of any time step and the corresponding confidence interval.
    Also returns the mean and confidence interval over time for all time steps.
    
    Parameters:
    - data: 2D list or array where each row represents a different run of the experiment and each column represents a time step.
    - confidence: Confidence level for the confidence interval (default is 0.95).
    
    Returns:
    - max_mean: Maximum mean result at any time step.
    - max_mean_ci: Confidence interval for the maximum mean result.
    - mean_over_time: Mean result for each time step across all runs.
    - ci_over_time: Confidence interval for each time step across all runs.
    """
    # Convert the data to a NumPy array for easier processing
    data = np.array(data)
    
    # Calculate the mean across runs for each time step
    mean_over_time = np.mean(data, axis=0)
    
    # Find the index of the time step with the maximum mean
    max_mean_index = np.argmax(mean_over_time)
    
    # Extract the data corresponding to the max mean time step
    max_mean_data = data[:, max_mean_index]
    
    # Calculate the mean and standard error for the max mean time step
    max_mean = np.mean(max_mean_data)
    std_error = stats.sem(max_mean_data)
    
    # Calculate the confidence interval for the max mean time step
    max_mean_ci = std_error * stats.t.ppf((1 + confidence) / 2, len(max_mean_data) - 1)
    
    # Calculate confidence intervals over time for all time steps
    std_error_over_time = stats.sem(data, axis=0)
    ci_over_time = std_error_over_time * stats.t.ppf((1 + confidence) / 2, data.shape[0] - 1)
    
    return max_mean, max_mean_ci, mean_over_time, ci_over_time


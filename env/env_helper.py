import numpy as np
import sys

# class ParametersV2X:
def parse_agent2veh(values): #TODO Delete??
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

def sample_veh_positions(data, t0=None, k_max=1):
    """Samples vehicle positions from the DataFrame. handles multiple use cases
        Inputs:
            data: pd.dataFrame
            t0: (Optional) starting timestep (withhold for random) [float:.1f]
            k_max (Optional): number of consecutive timesteps to sample
        Returns: slice of k_max timesteps fron DataFrame, index """

    unique_time_steps = data['time'].drop_duplicates().sort_values().values

    if t0 is None: #if no index provided select at random
        # Randomly sample a start index, ensuring k_max timesteps fit
        start_idx = np.random.randint(0, len(unique_time_steps) - k_max + 1)
    else:
        if t0 not in unique_time_steps:
            raise ValueError(f"Time step {t0} not found in data.")
        start_idx = np.where(unique_time_steps == t0)[0][0]
        if start_idx + k_max > len(unique_time_steps):
            raise ValueError(f"Cannot sample {k_max} timesteps starting from {t0}, out of range.")

    # Select k_max sequential timesteps
    sampled_time_steps = unique_time_steps[start_idx:start_idx + k_max]
    sampled_data = data[data['time'].isin(sampled_time_steps)]

    # print(unique_time_steps)
    # sys.exit()

    start_idx = unique_time_steps[start_idx]

    return sampled_data, start_idx #TODO SHOULD WRONG IDX


def remove_test_data_from_veh_pos(veh_pos_data, time_steps):
    # Remove all test data from veh_pos_data
    # Filter veh_pos_data to exclude rows with time in time_steps
    filtered_data = veh_pos_data[~veh_pos_data['time'].isin(time_steps)]

    return filtered_data
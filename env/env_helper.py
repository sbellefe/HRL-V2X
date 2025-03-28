import numpy as np

# class ParametersV2X:
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

def sample_veh_positions(data, time_step=None, k_max=None):
    """Samples vehicle positions from the DataFrame. handles multiple use cases
        Inputs:
            data: pd.dataFrame
            time_step (int): single timestep to sample
            k_max (int): number of consecutive timesteps to sample
        Returns: slice of dataframe"""
#TODO: allow selection of multiple pos samples starting at specific timestep (test data for AoI)
    #TODO: add functionality to take as input time_step = [list of indexes]
    if k_max is not None:   #Multi-location use case
        unique_time_steps = data['time'].nunique()
        sorted_time_steps = data['time'].drop_duplicates().sort_values()

        # Check if there are enough time steps to sample TODO should this really exit the code?
        if k_max > unique_time_steps: raise ValueError("Not enough time steps to sample")

        # Randomly select a block of time steps
        start_index = np.random.randint(0, unique_time_steps - k_max + 1)
        sampled_time_steps = sorted_time_steps.iloc[start_index:start_index + k_max]

        # Filter the DataFrame to include only the rows with the selected time steps
        sampled_data = data[data['time'].isin(sampled_time_steps)]

    elif time_step is not None: #Single-location use case
        if time_step not in data['time'].unique():
            print(f"Error: time step {time_step} not found in the data")
            return None
        sampled_data = data[data['time'] == time_step]

    else:
        raise ValueError("Either time_step or t_max_control must be provided.")

    return sampled_data





def remove_test_data_from_veh_pos(veh_pos_data, time_steps):
    # Remove all test data from veh_pos_data
    # Filter veh_pos_data to exclude rows with time in time_steps
    filtered_data = veh_pos_data[~veh_pos_data['time'].isin(time_steps)]

    return filtered_data
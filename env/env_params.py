

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

def sample_veh_positions(data, time_step):
    # Check if the provided time step exists in the data
    if time_step not in data['time'].unique():
        print(f"Error: time step {time_step} not found in the data")
        return None

    # Filter the DataFrame to include only the rows with the provided time step
    sampled_data = data[data['time'] == time_step]

    return sampled_data



def remove_test_data_from_veh_pos(veh_pos_data, time_steps):
    # Remove all test data from veh_pos_data
    # Filter veh_pos_data to exclude rows with time in time_steps
    filtered_data = veh_pos_data[~veh_pos_data['time'].isin(time_steps)]

    return filtered_data
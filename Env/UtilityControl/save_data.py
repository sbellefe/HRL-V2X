import pandas as pd


def save_data(time_array, track_ep, track_ev, acc_array, action_array, p_array, v_array, i_episode, i_vehicle):
    dataframe = pd.DataFrame(
        {'time': time_array, 'track_er_p': track_ep, 'track_er_v': track_ev, 'acceleration': acc_array,
         'action': action_array, 'position': p_array, 'velocity': v_array})
    dataframe.to_csv("Platoon_DRL_data/test_" + str(i_episode) + '_vehicle_' + str(i_vehicle) + '.csv', index=False,
                     sep=',')


def save_V2I_SE_losses_data(V2I_SE_losses, i_episode, i_vehicle):
    dataframe = pd.DataFrame(
        {'V2I_SE_losses': V2I_SE_losses})
    dataframe.to_csv("Platoon_DRL_data/test_V2I_losses_" + str(i_episode) + '_vehicle_' + str(i_vehicle) + '.csv',
                     index=False,
                     sep=',')


def save_V2I_TP_tot_data(V2I_SE_TP, i_episode):
    dataframe = pd.DataFrame(
        {'V2I_SE_TP': V2I_SE_TP})
    dataframe.to_csv("Platoon_DRL_data/test_V2I_losses_" + str(i_episode) + '.csv', index=False, sep=',')


def save_reward_data(time_array, reward_array, i_episode, i_vehicle):
    dataframe = pd.DataFrame(
        {'time': time_array, 'reward': reward_array})
    dataframe.to_csv("Platoon_DRL_data/test_reward_" + str(i_episode) + '_vehicle_' + str(i_vehicle) + '.csv', index=False,
                     sep=',')


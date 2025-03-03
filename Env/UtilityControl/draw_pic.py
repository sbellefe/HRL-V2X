import matplotlib
from utils_parser.parameters import args
from scipy import stats
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def draw_VoI(veh_VoI, episode):
    plt.figure(figsize=(16, 9))
    for i in range(args.n_veh - 1):
        plt.plot(veh_VoI[i])
    plt.xlabel('control time interval')
    plt.ylabel('VoI of received message at per following vehicle')
    plt.title('VoI of vehicle')
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.grid()
    plt.savefig('./Platoon_DRL_pic/V2I_TP_performance/V2I_TP_performance_VOI' + '_' + str(episode) + '.png')
    plt.close('all')


def draw_CDF_V2I(V2I_SE_tot, V2I_SE_losses, episode):
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    for i in range(args.n_veh - 1):
        res = stats.relfreq(V2I_SE_losses[i], numbins=40)
        x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
        y = np.cumsum(res.frequency)
        plt.plot(x, y)
        # https://blog.csdn.net/tian_tian_hero/article/details/85245424
    plt.xlabel('V2I SE loss at per Agent')
    plt.ylabel('CDF')
    plt.title('V2I SE losses')
    plt.legend(['SC1', 'SC2', 'SC3', 'SC4'], loc='upper right')
    plt.grid()
    plt.subplot(2, 1, 2)
    res = stats.relfreq(V2I_SE_tot, numbins=40)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y)
    plt.xlabel('Total TP of V2I links')
    plt.ylabel('CDF')
    plt.title('V2I SE')
    plt.grid()
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.97, top=0.95, wspace=0.2, hspace=0.27)
    plt.savefig('./Platoon_DRL_pic/V2I_TP_performance/V2I_TP_performance' + '_' + str(episode) + '.png')
    plt.close('all')


def draw_ep_ev_acc_a_AoI(time_array, track_er_p, track_er_v, acceleration_array, action_array, AoI_array, episode):
    plt.figure(figsize=(16, 9))
    plt.subplot(5, 1, 1)
    for i in range(args.n_veh):
        if i == 0:
            continue
        plt.plot(time_array, track_er_p[i])
    plt.xlabel('control time interval')
    plt.ylabel('tracking error position')
    plt.title('tracking error position of vehicle')
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.grid()
    plt.subplot(5, 1, 2)
    for i in range(args.n_veh):
        if i == 0:
            continue
        plt.plot(time_array, track_er_v[i])
    plt.xlabel('control time interval')
    plt.ylabel('tracking error velocity')
    plt.title('tracking error velocity of vehicle')
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.grid()
    plt.subplot(5, 1, 3)
    for i in range(args.n_veh):
        if i == 0:
            continue
        plt.plot(time_array, acceleration_array[i])
    plt.xlabel('control time interval')
    plt.ylabel('acceleration')
    plt.title('acceleration of vehicle')
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.grid()
    plt.subplot(5, 1, 4)
    for i in range(args.n_veh):
        plt.plot(time_array, action_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('action')
        plt.title('action of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplot(5, 1, 5)
    for i in range(args.n_veh):
        if i == 0:
            continue
        plt.plot(time_array, AoI_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('AoI')
        plt.title('AoI of receiving CPM')
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.grid()
    plt.subplots_adjust(left=0.055, bottom=0.06, right=0.97, top=0.95, wspace=0.2, hspace=0.7)
    plt.savefig('./Platoon_DRL_pic/ep_ev_acc_a/ep_ev_acc_a' + '_' + str(episode) + '.png')
    plt.close('all')


def draw_ep_ev_acc_a(time_array, track_er_p, track_er_v, acceleration_array, action_array, episode):
    plt.figure(figsize=(16, 9))
    plt.subplot(4, 1, 1)
    for i in range(args.n_veh):
        plt.plot(time_array, track_er_p[i])
        plt.xlabel('control time interval')
        plt.ylabel('tracking error position')
        plt.title('tracking error position of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplot(4, 1, 2)
    for i in range(args.n_veh):
        plt.plot(time_array, track_er_v[i])
        plt.xlabel('control time interval')
        plt.ylabel('tracking error velocity')
        plt.title('tracking error velocity of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplot(4, 1, 3)
    for i in range(args.n_veh):
        plt.plot(time_array, acceleration_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('acceleration')
        plt.title('acceleration of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplot(4, 1, 4)
    for i in range(args.n_veh):
        plt.plot(time_array, action_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('action')
        plt.title('action of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplots_adjust(left=0.055, bottom=0.06, right=0.97, top=0.95, wspace=0.2, hspace=0.46)
    plt.savefig('./Platoon_DRL_pic/ep_ev_acc_a/ep_ev_acc_a' + '_' + str(episode) + '.png')
    plt.close('all')


def draw_gl_CM_reward(episode_array, global_reward_array, mode, i_episode):
    plt.figure(figsize=(16, 9))
    for i in range(args.nb_episodes_Test):
        plt.plot(episode_array, global_reward_array[i])
        plt.legend(['episode1', 'episode2'])
    plt.xlabel('episode')
    plt.ylabel('Global reward')
    plt.title('Global reward')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./Platoon_DRL_pic/reward/global_CM_reward' + mode + '_' + str(i_episode) + '.png')
    plt.close('all')


def draw_gl_CL_reward(episode_array, agent_return_of_CL_reward_array, mode, i_episode):
    plt.figure(figsize=(16, 9))
    for i in range(args.n_veh):
        if i == 0:
            continue
        plt.plot(episode_array, agent_return_of_CL_reward_array[i])
    plt.legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
    plt.xlabel('Test episodes')
    plt.ylabel('Return of CL rewards during test episode')
    plt.title('Performance during DRL training')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./Platoon_DRL_pic/reward/reward' + mode + '_' + str(i_episode) + '.png')
    plt.close('all')


def draw_p_v(time_array, position_array, velocity_array, episode):
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    for i in range(args.n_veh):
        plt.plot(time_array, position_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('position')
        plt.title('position of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplot(2, 1, 2)
    for i in range(args.n_veh):
        plt.plot(time_array, velocity_array[i])
        plt.xlabel('control time interval')
        plt.ylabel('velocity')
        plt.title('velocity of vehicle')
        plt.legend(['Leader 0', 'Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], loc='upper right')
        plt.grid()
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.97, top=0.95, wspace=0.2, hspace=0.27)
    plt.savefig('./Platoon_DRL_pic/p_v/p_v' + '_' + str(episode) + '.png')
    plt.close('all')

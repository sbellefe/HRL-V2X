import argparse, time
import sys

import gymnasium as gym

from Env.env_params import V2Xparams
from Env.Environment import Environ
from Env.UtilityCommunication.veh_position_helper import *

# Import runner, trainers, and parameters classes here
from runner.runner import ALGO_Runner
from train.mappo_trainer import MAPPOtrainer
# from train.oc_trainer import OCtrainer
# from train.dac_trainer import DACtrainer
from util.parameters import ParametersPPO, ParametersOC, ParametersDAC

def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "NFIG", "SIG", OR "POSIG".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "ppo" or "oc" or "dac".')
    args = parser.parse_args()

    test_data_list = []
    veh_pos_data = load_veh_pos('./Env/SUMOData/4ag_4V2I.csv')
    # veh_pos_data = load_veh_pos('./Env/SUMOData/4ag_4V2I_SIG.csv')
    # veh_pos_data = load_veh_pos('./Env/SUMOData/8ag_4V2I_NFIG.csv')


    #create environment

    env_params = V2Xparams()        #TODO: Migrate to communal parameters file??
    env = Environ(env_params)
    if args.env == 'NFIG':
        env_name = args.env
        # test_data_list = [sample_veh_position_from_timestep(veh_pos_data, env_params.env_setup)]
        # sampled_data = test_data_list[0]
        raise ValueError("Environment 'NFIG' not implemented.")
    # elif args.env == 'SIG':
    # elif args.env == 'POSIG'
    #     raise ValueError("Environment 'POSIG' not implemented.")
    elif args.env == 'SIG' or args.env == 'POSIG':
        env_name = args.env
        if isinstance(env_params.env_setup, str):
            time_steps = range(35, 45)
            for time in time_steps:
                test_data = sample_veh_position_from_timestep(veh_pos_data, time)
                test_data_list.append(test_data)

            training_data = remove_test_data_from_veh_pos(veh_pos_data, time_steps)
            sampled_data = sample_veh_positions(1, training_data)
        else:
            test_data_list = [sample_veh_position_from_timestep(veh_pos_data, env_params.env_setup[1])]
            sampled_data = test_data_list[0]
    else:
        raise ValueError("Environment name incorrect or found")

    #Add loaded vehicle position data to env class
    env_params.loaded_veh_data = sampled_data
    env.veh_pos_data = veh_pos_data
    env.loaded_veh_data = sampled_data
    env.test_data_list = test_data_list


    #assign params and trainer classes based on algo input
    if args.algo == 'ppo':
        params = ParametersPPO()
        trainer = lambda: MAPPOtrainer()
    elif args.algo == 'oc':
        params = ParametersOC()
        trainer = lambda: OCtrainer()
    elif args.algo == 'dac':
        params = ParametersDAC()
        trainer = lambda: DACtrainer()
    else:
        raise ValueError("Algorithm name incorrect or not found")

    #add environment specific parameters
    params.env_name = env_name
    params.state_dim = env.stateDim
    params.action_dim = env.actionDim


    #define runner and run experiment
    runner = ALGO_Runner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



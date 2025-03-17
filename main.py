import argparse, time, sys
import pandas as pd

import gymnasium as gym

from Envs.env_params import V2Xparams
from env.v2x_env import V2XEnvironment
from env.env_helper import *

# from Envs.Environment import Environ
# from Envs.UtilityCommunication.veh_position_helper import *

# Import runner, trainers, and parameters classes here
from runner.runner import ALGO_Runner
from train.mappo_trainer import MAPPOtrainer
# from train.oc_trainer import OCtrainer
# from train.dac_trainer import DACtrainer
from util.parameters import ParametersMAPPO, ParametersOC, ParametersDAC

def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "NFIG", "SIG", OR "POSIG".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "ppo" or "oc" or "dac".')
    args = parser.parse_args()

    """Load algorithm params and trainers"""
    if args.algo == 'mappo':
        params = ParametersMAPPO()
        trainer = lambda: MAPPOtrainer()
    elif args.algo == 'oc':
        params = ParametersOC()
        trainer = lambda: OCtrainer()
    elif args.algo == 'dac':
        params = ParametersDAC()
        trainer = lambda: DACtrainer()
    else:
        raise ValueError("Algorithm name incorrect or not found")

    """Create environment"""
    #load position data from .csv
    test_data_list = []
    veh_pos_data = pd.read_csv('env/SUMOData/4ag_4V2I.csv')
    # print(veh_pos_data)


    # print(test_data_list)



    #Load test positional data
    if args.env == 'NFIG':
        #slice a single positional data sample
        test_data_list = [sample_veh_positions(veh_pos_data, params.single_loc_idx)]

    elif args.env == 'SIG' or args.env == 'POSIG':
        if params.multi_location is False: #Single Location
            # slice a single positional data sample
            test_data_list = [sample_veh_positions(veh_pos_data, params.single_loc_idx)]
        else:   #Multi Location
            #slice multiple positional data samples
            time_steps = range(params.multi_loc_idx[0], params.multi_loc_idx[0])
            for step in time_steps:
                test_data = sample_veh_positions(veh_pos_data, step)
                test_data_list.append(test_data)
            veh_pos_data = remove_test_data_from_veh_pos(veh_pos_data, time_steps)
    else:
        raise ValueError("Environment name incorrect or found")

    #Load environment
    env = V2XEnvironment(params,veh_pos_data,test_data_list)



    #define runner and run experiment
    runner = ALGO_Runner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



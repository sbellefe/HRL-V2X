import argparse, time, sys
import pandas as pd

from env.v2x_env import V2XEnvironment
from env.env_helper import sample_veh_positions, remove_test_data_from_veh_pos

# Import runner, trainers, and parameters classes here
from runner.runner import ALGO_Runner
from train.mappo_trainer import MAPPOtrainer
# from train.oc_trainer import OCtrainer
# from train.dac_trainer import DACtrainer
from util.parameters import ParametersMAPPO, ParametersOC, ParametersDAC

def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "NFIG", "SIG", OR "POSIG".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "mappo" or "oc" or "dac".')
    args = parser.parse_args()

    """Load algorithm params and trainers"""
    if args.algo == 'mappo':
        params = ParametersMAPPO()
        trainer = lambda: MAPPOtrainer()
    elif args.algo == 'oc':
        raise ValueError('not implemented')
        params = ParametersOC()
        trainer = lambda: OCtrainer()
    elif args.algo == 'dac':
        raise ValueError('not implemented')
        params = ParametersDAC()
        trainer = lambda: DACtrainer()
    else:
        raise ValueError("Algorithm name incorrect or not found")

    """Create environment"""
    #load position data from .csv
    veh_pos_data = pd.read_csv('env/SUMOData/4ag_4V2I.csv')
    # print("Raw pd.read_csv:",veh_pos_data)

    #Determine game_mode
    if args.env == 'NFIG':
        game_mode, k_max = 1, 1
    elif args.env == 'SIG': #TODO Review nested conditionals
        if params.include_AoI is False:
            game_mode, k_max = 2, 10
        else:
            game_mode, k_max = 3, params.k_max
    elif args.env == 'POSIG':
        game_mode, k_max = 4, params.k_max
    else:
        raise ValueError("Unknown environment")

    #load env_setup params
    env_setup = {
        'game_mode': game_mode,
        'k_max': k_max,  # number of control intervals
        'fast_fading': params.fast_fading,
        'num_agents': params.num_agents,
        'multi_location': params.multi_location,
        't_max_control': params.t_max_control,
        'single_loc_idx': params.single_loc_idx,
        'multi_loc_test_idx': params.multi_loc_test_idx,
    }

    #Load environment
    env = V2XEnvironment(env_setup, veh_pos_data)

    #add additional parameters
    params.state_dim = env.state_dim
    params.action_dim = env.action_dim

    #define runner and run experiment
    runner = ALGO_Runner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



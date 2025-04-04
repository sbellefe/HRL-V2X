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

    #Determine game_mode and PO boolean
    if args.env == 'NFIG':
        game_mode = 1
        partial_observability = False
    elif args.env == 'SIG':
        game_mode = 2 if not params.include_AoI else 3
        partial_observability = False
    elif args.env == 'POSIG':
        game_mode = 4 if not params.include_AoI else 5
        partial_observability = True
    else:
        raise ValueError("Unknown environment")

    #determine number of control intervals per episode
    if game_mode in [1,2,4]:
        k_max = 1
        params.k_max = k_max
    elif game_mode in [3,5]:
        k_max = params.k_max

    #load env_setup params
    env_setup = {
        'game_mode': game_mode,
        'k_max': k_max,  #number of control intervals
        't_max': params.t_max,  #length of control interval
        'num_agents': params.num_agents,
        'fast_fading': params.fast_fading,
        'multi_location': params.multi_location,
        'single_loc_idx': params.single_loc_idx,
        'multi_loc_test_idx': params.multi_loc_test_idx,
        'partial_observability': partial_observability,
    }

    #Load environment
    env = V2XEnvironment(env_setup, veh_pos_data)

    #add additional parameters
    params.state_dim = env.state_dim
    params.obs_dim = env.obs_dim
    params.action_dim = env.action_dim
    params.partial_observability = partial_observability

    #define runner and run experiment
    runner = ALGO_Runner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



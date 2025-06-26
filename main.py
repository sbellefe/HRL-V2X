import argparse, time, os, sys
from datetime import datetime
import pandas as pd

# Import runner, trainers, and parameters classes here
from util.parameters import ParametersMAPPO, ParametersDAC, ParametersOC
from runner.runner import ALGO_Runner
from train.mappo_trainer import MAPPOtrainer
from train.dac_trainer import DACtrainer
# from train.oc_trainer import OCtrainer

from env.v2x_env import V2XEnvironment


def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "NFIG", "SIG", OR "POSIG".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "mappo" or "oc" or "dac".')
    args = parser.parse_args()

    """Load algorithm params and trainers"""

    """Create environment"""
    if args.algo == 'mappo':
        params = ParametersMAPPO()
        trainer = lambda: MAPPOtrainer()
    elif args.algo == 'dac':
        params = ParametersDAC()
        trainer = lambda: DACtrainer()
    elif args.algo == 'oc':
        raise ValueError('not implemented')
        params = ParametersOC()
        trainer = lambda: OCtrainer()
    else:
        raise ValueError("Algorithm name incorrect or not found")
    #load position data from .csv
    veh_pos_data = pd.read_csv(params.veh_data_dir)
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

    #set number of control intervals / episode
    if game_mode in [1,2,4]:  #no AoI
        k_max = 1
        params.k_max = k_max
    elif game_mode in [3,5]:  #AoI
        k_max = params.k_max

    #override t_max for NFIG setup
    t_max = 1 if game_mode == 1 else params.t_max

    #load env_setup params
    env_setup = {
        'game_mode': game_mode,
        'k_max': k_max,  #number of control intervals
        't_max': t_max,  #length of control interval
        'num_agents': params.num_agents,
        'fast_fading': params.fast_fading,
        'multi_location': params.multi_location,
        'single_loc_idx': params.single_loc_idx,
        'multi_loc_test_idx': params.multi_loc_test_idx,
        'partial_observability': partial_observability,
        'render_mode': params.render_mode,
    }

    #Load environment
    env = V2XEnvironment(env_setup, veh_pos_data)

    """add additional parameters"""
    params.state_dim = env.state_dim
    params.obs_dim = env.obs_dim
    params.action_dim = env.action_dim
    params.partial_observability = partial_observability

    #OVERRDE for HRL setups
    if args.algo in ['dac', 'oc']:
        params.action_dim = env.num_power_levels + 1 #including 0dB action
        params.num_options = env.num_SC + 1

    """Build run directory // store params"""
    runs_root = os.path.join(os.getcwd(), "runs")
    os.makedirs(runs_root, exist_ok=True)

    # ts = datetime.now().strftime("%Y-%m-%d_%I.%M%p")  # e.g. "2025-06-09_12:34PM"
    ts = datetime.now().strftime("%Y-%m-%d_%H.%M")  # e.g. "2025-06-09_14.34"
    algo_str = args.algo.upper()
    env_str = args.env
    ff_str = "FF" if params.fast_fading else "NFF"
    run_dir = os.path.join(runs_root, f"{ts}_{algo_str}_{env_str}_{ff_str}")

    os.makedirs(run_dir, exist_ok=False)    #Change to exist_ok=False for running full experiments
    params.run_dir = run_dir

    info_path = os.path.join(run_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write("Hyperparameters:\n")
        for k, v in vars(params).items():
            f.write(f"  - {k}: {v}\n")


    """define runner and run experiment"""
    runner = ALGO_Runner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



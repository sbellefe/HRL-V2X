import argparse, time
import gymnasium as gym

from env.fourrooms import FourRooms

# Import runner, trainers, and parameters classes here
from runner.runner import ALGOrunner
from train.ppo_trainer import PPOtrainer
from util.parameters import ParametersPPO

def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "cartpole" or "fourrooms".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "dqn", "ppo", or "a2c".')
    args = parser.parse_args()

    #create environment
    if args.env == 'cartpole':
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
    elif args.env == 'fourrooms':
        env_name = None
        env = FourRooms()
    else:
        raise ValueError("Environment name incorrect or found")

    #assign params and trainer classes based on algo input
    if args.algo == 'ppo':
        params = ParametersPPO()
        trainer = lambda: PPOtrainer()
    elif args.algo == 'a2c':
        raise ValueError("Algo not implemented")
    elif args.algo == 'dqn':
        raise ValueError("Algo not implemented")
    else:
        raise ValueError("Algorithm name incorrect or not found")

    #add environment specific parameters
    params.env_name = env_name
    params.state_dim = env.observation_space.shape[0]
    params.action_dim = env.action_space.n
    # TODO: add logic for discrete vs. continuous spaces?

    #define runner and run experiment
    runner = ALGOrunner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    print(f"Execution Time: {(end_time - start_time):.2f} seconds")



import os, time, sys
import numpy as np
from util.benchmarker import Utils


class ALGO_Runner():
    def __init__(self, env, trainer):
        self.env = env
        self.trainer = trainer

    def run_experiment(self, params, load_save_result=False):
        """Change load_save_result to True to plot from existing test/train rewards"""
        start = time.time()
        if load_save_result is False or not os.path.isfile('all_train_returns.npy') or not os.path.isfile('all_test_returns.npy'):
            all_train_returns = []
            all_test_returns = []

            for trial in range(params.num_trials):
                print(f"Trial: {trial + 1}")

                #initialize trainer and call train method
                trainer = self.trainer()
                train_rewards, test_rewards = trainer.train(trial, self.env, params)

                all_train_returns.append(train_rewards)
                all_test_returns.append(test_rewards)

            print(f"Experiment completed in {(time.time() - start):.2f} seconds")
            np.save('all_train_returns.npy', all_train_returns)
            np.save('all_test_returns.npy', all_test_returns)
        else:
            all_train_returns = np.load('all_train_returns.npy')
            all_test_returns = np.load('all_test_returns.npy')

        utils = Utils()
        average_returns, max_return, max_return_ci, individual_returns = utils.benchmark_plot(all_train_returns,
                                                                                              all_test_returns,
                                                                                              params.test_interval)
        print(f"Average Return: {np.round(average_returns, 2)}")
        print(f"Overall Max Return w/ 95% CI: {max_return:.3f} +- {max_return_ci:.3f}")
        print(f"Individual Run Overall Max Returns: {np.round(individual_returns, 3)}")
        print("Completed experiment")
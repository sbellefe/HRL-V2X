import os, time, sys
import numpy as np
from util.benchmarker import Utils


class ALGO_Runner():
    def __init__(self, env, trainer):
        self.env = env
        self.trainer = trainer

    def run_experiment(self, params, load_save_result=False):
        """Change load_save_result to True to plot from existing test/train rewards. In that case,
        expects the two .npy files in the repo root, and writes results & plots back into the root."""
        start = time.time()
        run_dir = params.run_dir

        if load_save_result is False:
            all_train_returns = []
            all_test_returns = []

            for trial in range(params.num_trials):
                print(f"Trial: {trial + 1}")

                #initialize trainer and call train method
                trainer = self.trainer()
                train_returns, test_returns = trainer.train(trial, self.env, params)

                all_train_returns.append(train_returns)
                all_test_returns.append(test_returns)
                print(f"Trial {trial+1} Complete. Max test return {max(test_returns):.3f}.")

            print(f"Experiment completed in {(time.time() - start):.2f} seconds")
            np.save(os.path.join(run_dir, 'all_train_returns.npy'), all_train_returns)
            np.save(os.path.join(run_dir, 'all_test_returns.npy'), all_test_returns)
        else:
            if not os.path.isfile('all_train_returns.npy') or not os.path.isfile('all_test_returns.npy'):
                raise ValueError("'all_test_returns.npy' and 'all_train_returns.npy' must be in repo root.")
            all_train_returns = np.load('all_train_returns.npy')
            all_test_returns = np.load('all_test_returns.npy')
            run_dir = os.getcwd()

        utils = Utils()
        avg_ret, max_ret, max_ret_ci, indiv_ret = utils.benchmark_plot(all_train_returns,
                                                                       all_test_returns,
                                                                       params.test_interval,
                                                                       run_dir)
        results = [
            f"Experiment completed in {(time.time() - start):.2f} seconds",
            f"Average Return: {np.round(avg_ret, 2)}",
            f"Overall Max Return w/ 95% CI: {max_ret:.3f} +- {max_ret_ci:.3f}",
            f"Individual Run Overall Max Returns: {np.round(indiv_ret, 3)}",
            "Completed experiment"
        ]

        # print & write them to results.txt
        with open(os.path.join(run_dir, 'results.txt'), 'w') as f:
            for line in results:
                print(line)
                f.write(line + "\n")

        # print(f"Average Return: {np.round(avg_ret, 2)}")
        # print(f"Overall Max Return w/ 95% CI: {max_ret:.3f} +- {max_ret_ci:.3f}")
        # print(f"Individual Run Overall Max Returns: {np.round(indiv_ret, 3)}")
        # print("Completed experiment")
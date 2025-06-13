# Hierarchical RL - V2X Experiment

Implementation and benchmarking of multi-agent HRL algorithms in V2X environments. Compares OC and DAC to MAPPO.

## requirements: 
```pip install -r requirements.txt```

## main.py:

Use this command in terminal to run code:

```python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE```\
e.g ```python main.py --env SIG --algo ippo```

## env: V2X Environment 
Choose from the following options in the command line:
- ```NFIG```: single timestep game with channel awareness
- ```SIG```: multi timestep game with channel and queue awareness
- ```POSIG```: same as ```SIG``` but with partial observability (i.e. agents don't see global state)

Additional game mode configurations (set by changing attributes of util/parameters/```SharedParams``` class):
- multi-location (mloc) vs. single location (sloc): ```self.multi_location``` (bool)
- fast-fading (ff) vs. no fast-fading (nff): ```self.fast_fading``` (bool)
- positional data time index to use for sloc: ```self.single_loc_idx``` (e.g. 25.0)
- positional data time indices reserved for testing in mloc: ```self.multi_loc_test_idx``` (e.g. range\[20, 30])

env/SUMOData: Contains positional vehicle data in .csv files obtained from simulation.

env/v2x_env.py: Contains ```Vehicle``` and ```V2XEnvironment``` classes for managing vehicle positions and calculating pathlosses & rewards.
- Adjust local environment variables/configurations in ```V2XEnvironment.__init__```

## util:

benchmarker.py: Compute evaluation metrics, plot training and testing results.
- Each run is benchmarked after completion
- Additional method to benchmark multiple runs

parameters.py: 
- Shared hyperparameter class ```SharedParams``` used for additional environment configuration and high-level train/test loop structure.
- Hyperparameter classes for each algorithm inherit from ```SharedParams```. Used for algorithm specific hyperparameters.

## agent: 
mappo.py: Contains ```MAPPOActor``` and ```MAPPOCritic``` classes.

oc.py: Contains ```OC_SingleOptionNet``` and ```OC_Networks``` classes. UPDATE

dac.py: Contains ```DAC_SingleOptionNet``` and ```DAC_Network``` classes. UPDATE

## runner:
runner.py: Contains ```ALGO_Runner``` class and the method ```run_experiment``` used for all algorithms.

## trainer: UPDATE
mappo_trainer.py: Contains ```MAPPOtrainer``` class with methods ```train_FO```, ```train_PO```, ```test_FO```, ```test_PO```.

oc_trainer.py: Contains ```OCtrainer``` class with methods ```train```, ```test```.

dac_trainer.py: Contains ```DACtrainer``` class with methods ```train```, ```learn```, ```test```. 



## helper:

Contains additional classes and functions for training algorithms (e.g. replay buffer, pre-processing...).



 

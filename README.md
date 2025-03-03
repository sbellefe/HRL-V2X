# Hierarchical RL

Implementation and benchmarking of multi-agent HRL algorithms in V2X environments. Compares OC and DAC to IPPO.

## requirements: 
```pip install -r requirements.txt```

## agent: 
ippo.py: Contains ```IPPOActor``` and ```IPPOCritic``` classes.

oc.py: Contains ```OC_SingleOptionNet``` and ```OC_Networks``` classes.

dac.py: Contains ```DAC_SingleOptionNet``` and ```DAC_Network``` classes.

## runner:
runner.py: Contains ```ALGO_Runner``` class and the method ```run_experiment``` used for all algorithms.

## trainer:
ippo_trainer.py: Contains ```IPPOtrainer``` class with methods ```train```, ```test```.

oc_trainer.py: Contains ```OCtrainer``` class with methods ```train```, ```test```.

dac_trainer.py: Contains ```DACtrainer``` class with methods ```train```, ```learn```, ```test```. 



## helper:

Contains additional classes and functions for training algorithms (e.g. replay buffer, pre-processing...).

## V2X Env:
Choose from:
- ```NFIG```: single timestep game
- ```SIG```: multi timestep game with channel and queue awareness
- ```POSIG```: same as ```SIG``` but with partial observability (i.e. no access to global state)

Env/SUMOData: Contains raw data in .csv files obtained from simulation.

Environment.py: Contains ```Vehichle``` and ```Environ``` classes for managing vehicle positions and calculating fast fading & rewards.

env_params.py: Contains ```V2Xparams``` class with environment configuration parameters.

env_helper.py and /UtilityCommunication/veh_position_helper.py: Contains various helper code.

## util:

benchmarker.py: Compute evaluation metrics, plot training and testing results.
- Each run is benchmarked after completion
- Additional method to benchmark multiple runs

parameters.py: 
- Hyperparameter classes for each algorithm
- Merged with ```V2Xparams``` class

## main.py:

Use this command in terminal to run code:

```python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE```\
e.g ```python main.py --env SIG --algo ippo``` 

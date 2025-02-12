# Hierarchical RL

Implementation and benchmarking of HRL algorithms in FourRooms Environment. Compares OC and DAC to standard Single RL PPO.

## requirements: 
```pip install -r requirements.txt```

## agent: 
ppo.py: Contains ```PPOActor``` and ```PPOCritic``` classes.

oc.py: Contains ```OptionCritic``` class.

dac.py: Contains ```DAC_SingleOptionNet``` and ```DAC_Network``` classes

## runner:
runner.py: Contains ```ALGO_Runner``` class and the method ```run_experiment``` used for all algorithms.

## trainer:
ppo_trainer.py: Contains ```PPOtrainer``` class with methods ```train```, ```test```.

oc_trainer.py: Contains ```OCtrainer``` class with methods ```train```, ```test```.

dac_trainer.py: Contains ```DACtrainer``` class with methods ```train```, ```learn```, ```test```. 



## helper:

Contains additional classes and functions for training algorithms (e.g. replay buffer, pre-processing...).

## env:
FourRooms Gridworlds:

fourrooms.py: Contains ```FouRooms``` and  ```FourRooms_m``` classes:

- Classes inherit from Gymnasium Env class
- ```FouRooms``` is the original layout from OC paper
- ```FourRooms_m``` is a modified layout with a zig-zag pattern
- Both classes have pygame rendering and the ability to display real-time agent variable
- For HRL agents, rendering will show variables including: current option, policies, beta...

## util:

benchmarker.py: Compute evaluation metrics, plot training and testing results.

parameters.py: 
- Hyperparameter classes for each algorithm
- Shared benchmarking hyperparameter class. Setup here to render the environment during testing. 


## main.py:

Use this command in terminal to run code:

```python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE```\
e.g ```python main.py --env fourrooms --algo dac``` 
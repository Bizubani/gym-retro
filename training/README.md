# The Brothers Grimm - Two AI agents

This project contains two game playing agent generation systems. One use the BRUTE alogrithm and the other uses PPO.
The environment are set up to work with the following games:

- Gradius III
- R-Type III
- Darius Force
- Final Fight 3
- Final Fight 2
- Super Double Dragon
- Double Dragon and Battletoads

Other games can be added trivially be updating main to support the new game and add the ROM to your ROM folder (see Reto Gym cmds for more info).

## Agents

Wilhelm is the agent that uses PPO and Jacob is the agent that uses BRUTE.
To make a call to the agents, you need to run the following command from the training folder:

```
python main.py --grimm <grimm-name> --game <game> (extra parameters)
```

The grimm-name is either wilhelm or jacob. The game is the name of the game you want to train on. The extra parameters are optional and can be used to change the default parameters. The default parameters are:

```
common parameters:
--state <state> --scenario <scenario> --record_path <path to record bk2> --use_custom_integrations
Jacob parameters:
--file_to_load <bk2 input to load> --penalty_scale <max penalty> --episode_steps <max timesteps per episode> --time_step_limit <max time steps for Jacob>
Wilhelm parameters:
--model_to_load <model to load for wilhelm> --n_games <number of games to run> --play_only <do no learning> --tag <tag the generated files>
```

## Installation

To install the project, you need to install the following packages:
if you are using gym-retro, you need to install the following packages:
pip install gym-retro
pip install gym==0.25.2

if you are using stable-retro
pip3 install git+https://github.com/Farama-Foundation/stable-retro.git

### common dependencies

pip install numpy
pip install pytorch
pip install torchvision

if you have a compatible GPU, you can install the nvidia cuda toolkit to speed up the training process
https://developer.nvidia.com/cuda-downloads
https://nvidia.github.io/cuda-python/install.html

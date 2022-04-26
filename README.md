# PPO Tetris

An implementation of a Proximal Policy Optimisation [[1]](#references) agent
and a Trust Region-Guided Proximal Policy Optimisation [[2]](#references) agent
on a Tetris environment.

![Reinforcement learning agent playing Tetris](.github/example_gif.gif)

**Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [Agents and environments](#agents-and-environments)
4. [References](#references)

## Installation

### Downloading the files

The simplest way to run the project is to download the files.

### Cloning the repository

Alternatively you could clone the repository.

1. Fork the project
2. Download it using:
```shell
git clone https://github.com/<YOUR-USERNAME>/PPO_Tetris
```

### Installing dependencies

Once downloaded, install the dependencies for this project using pip.
```shell
cd PPO_Tetris
pip install -r requirements.txt
```

## Usage

The project is built following a specific folder structure. The training folder
is used as the entry point for storing for each run. To create a new test,
first start by creating a new folder in the training folder with the test name
and adding a config.ini file, weights folder and gifs folder to this.

### Running the agent
You can run the test using:
```shell
python run.py --name <test_name>
```

For example, the agent included with this project can be run with:
```shell
python run.py --name ppo
```

**Note: This will continue training. To restart the training use the --reset
option.**

You can run multiple agents in series using:
```shell
python run_all.py --names <test_name_1> <test_name_2> ...
```

For a list of all the command line arguments supported by these files, use the
-h flag on any of the files.

### Generating the logs
You can generate logs from a previous run using:
```shell
python run_data.py --names <path_to_test_folder>
```

For example, to generate logs for the agent included with this project:
```shell
python run_data.py --names training/ppo
```

**Note: The checkpoints required to run the data logging are not included with
this repository due to their size, please rerun the test to generate the
checkpoints.**

## Agents and environments

The project provides a configuration for a single run that uses PPO. The
configuration of the test can be found in [training/ppo](./training/ppo).

The project uses
[gym-simpletetris](https://github.com/tristanrussell/gym-simpletetris), a
custom Tetris environment that was built entirely in Python to allow easy
modification. It supports many options that can be enabled in the configuration
file. Instructions for how to use the environment and what the options do can
be found at the
[project repository](https://github.com/tristanrussell/gym-simpletetris).

## References

[1] [J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, Prox-
imal policy optimization algorithms, 2017. arXiv: 1707.06347 [cs.LG]](https://arxiv.org/abs/1707.06347).

[2] [Y. Wang, H. He, X. Tan, and Y. Gan, Trust region-guided proximal
policy optimization, 2019. arXiv: 1901.10314 [cs.LG]](https://arxiv.org/abs/1901.10314).

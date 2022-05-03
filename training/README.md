# Training data

A directory for storing training data. The ppo test included with this project
can be found in the [ppo directory](./ppo).

To create new tests, create a new directory and use the following directory
structure:
- config.ini - configuration file for the test
- weights - directory for storing the training weights
- gifs - directory for storing the gifs

## Example PPO test

In the [ppo directory](./ppo) you will find an example agent that has been run
for 100000 iterations of 4096 timesteps each iteration. Weights are included
for every 10000 timesteps and the [results directory](./ppo/results) includes
some graphs that were constructed from the tensorboard logs.

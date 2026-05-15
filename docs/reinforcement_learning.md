# Reinforcement Learning in hmls

Computer needs to learn how to play the game
Use a technique called reinforcement learning.
Have some sample tanks to train; can also generate your own

## What is reinforcement learning

Take a move, get a reward.
Play lots of games, use reward feedback to train the model

## Running the trainer

Give basic command, pointing to sample_configs

Highlight key portions of config file:
- Selecting tanks
- Rewards
- Training parameters, with high level explanations

Note that one training run is unlikely to produce a good model. Will need several runs

## Available tanks

Point out mk-i/ii/iii tanks with different NN configurations (mention CNN and GRU)

Random tank moves randomly, fires deterministically. Better opponent for initial training

## Implementing new tanks

Quick explanation of creating a new package, in particular entry point and required contracts to implement
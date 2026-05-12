"""Rule-based random tank for the HMLS tank game.

This package provides :class:`~hmls.randomtank.model.RandomTankModel`,
a minimal :class:`~hmls.nncore.model.TankModelBase` implementation with
no real neural network, and :class:`~hmls.randomtank.player.RandomTankPlayer`,
a player that selects actions using simple probabilistic rules rather
than learned policy weights.

The player inspects the egocentric visibility patch directly:

1. If an alive enemy occupies the cell directly in front, **fire**.
2. If the cell in front is impassable (boundary, impassable terrain,
   or occupied by any tank — alive or destroyed), **turn** left or
   right with configurable probability.
3. If the cell in front is passable, **move forward**, **turn left**,
   or **turn right** with configurable probabilities.

This tank is intended as a training opponent (set ``"train": false``
in the trainer config) and ignores all training and checkpoint
instructions from the reinforcement training loop.
"""

"""Mk-II neural network model for the HMLS tank game (single tank).

This package provides :class:`~hmls.singlemkii.model.MkIITankPolicyNetwork`,
a PyTorch CNN → GRU₁ → GRU₂ → policy-head architecture for single-tank
action selection, along with :class:`~hmls.singlemkii.model.MkIIModelConfig`
for hyperparameters and a :data:`~hmls.singlemkii.persistence.PERSISTENCE`
instance for model save/load and player creation.

The dual stacked GRU design gives the model deeper temporal reasoning
compared to the single-GRU Mk-I architecture, with independently
configurable hidden sizes for each GRU layer.

The player infrastructure (:class:`~hmls.nncore.player.NNPlayer`) and
training trajectory storage live in :mod:`hmls.nncore`.
"""

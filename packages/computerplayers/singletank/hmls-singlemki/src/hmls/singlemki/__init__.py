"""Mk-I neural network model for the HMLS tank game (single tank).

This package provides :class:`~hmls.singlemki.model.MkITankPolicyNetwork`,
a PyTorch CNN→GRU→policy-head architecture for single-tank action
selection, along with :class:`~hmls.singlemki.model.MkIModelConfig`
for hyperparameters and a :data:`~hmls.singlemki.persistence.PERSISTENCE`
instance for model save/load and player creation.

The player infrastructure (:class:`~hmls.nncore.player.NNPlayer`) and
training trajectory storage live in :mod:`hmls.nncore`.
"""

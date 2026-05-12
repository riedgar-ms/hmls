"""Mk-III neural network model for the HMLS tank game (single tank).

This package provides :class:`~hmls.singlemkiii.model.MkIIITankPolicyNetwork`,
a PyTorch Flatten→GRU→policy-head architecture for single-tank action
selection, along with :class:`~hmls.singlemkiii.model.MkIIIModelConfig`
for hyperparameters and a :data:`~hmls.singlemkiii.persistence.PERSISTENCE`
instance for model save/load and player creation.

Unlike the Mk-I and Mk-II architectures, the Mk-III bypasses all
convolutional processing: the 5-channel encoded patch is flattened directly
and fed into a single GRU cell.  This makes the model smaller and faster
at the cost of losing learned spatial feature extraction.

The player infrastructure (:class:`~hmls.nncore.player.NNPlayer`) and
training trajectory storage live in :mod:`hmls.nncore`.
"""

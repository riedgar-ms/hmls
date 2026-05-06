"""Neural network player for the HMLS tank game (single tank).

This package provides a PyTorch-based :class:`~hmls.core.player.Player`
implementation that uses a CNN→GRU→policy-head architecture to choose
actions.  It supports both inference ("play") and training ("learn")
modes, with REINFORCE-compatible trajectory storage.
"""

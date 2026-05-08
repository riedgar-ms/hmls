"""Mk-II neural network player for the HMLS tank game (single tank).

This package provides a PyTorch-based :class:`~hmls.core.player.Player`
implementation that uses a CNN → GRU₁ → GRU₂ → policy-head architecture
to choose actions.  The dual stacked GRU design gives the model deeper
temporal reasoning compared to the single-GRU Mk-I architecture, with
independently configurable hidden sizes for each GRU layer.
"""

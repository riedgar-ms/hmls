"""REINFORCE policy gradient trainer for HMLS tank models.

This package provides a CLI-driven training loop that pits two neural
network players against each other on randomly generated maps, using
the REINFORCE algorithm to improve their policies.  It is model-agnostic
and discovers concrete model types at runtime via the ``hmls.models``
entry-point registry (see :mod:`hmls.nncore.registry`).
"""

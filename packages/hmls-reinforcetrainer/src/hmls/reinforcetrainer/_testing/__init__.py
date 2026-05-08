"""Internal testing utilities for the reinforcetrainer package.

This sub-package provides stub implementations of the tank model,
player, and persistence interfaces defined in :mod:`hmls.nncore`.
These stubs are used by the reinforcetrainer test suite to avoid
depending on concrete model packages (e.g. ``hmls-singlemki``).

.. warning::
    This package is **not** part of the public API.  It is intended
    exclusively for testing within ``hmls-reinforcetrainer``.
"""

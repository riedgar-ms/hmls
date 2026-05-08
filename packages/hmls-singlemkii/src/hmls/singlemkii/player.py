"""Neural-network player implementation for Mk-II CNN→GRU₁→GRU₂→policy head.

The :class:`NNPlayer` is re-exported from :mod:`hmls.nncore.player` for
backward compatibility.  All model architectures share the same concrete
player implementation since it operates via the
:class:`~hmls.nncore.model.TankModelBase` interface.
"""

from hmls.nncore.player import NNPlayer

__all__ = ["NNPlayer"]

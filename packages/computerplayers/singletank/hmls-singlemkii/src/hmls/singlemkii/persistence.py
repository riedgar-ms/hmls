"""Model persistence for the Mk-II tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemkii.model.MkIIModelConfig`
and :class:`~hmls.singlemkii.model.MkIITankPolicyNetwork`.

This module is registered as the ``singlemkii`` entry point under
the ``hmls.models`` group.  It can also be discovered via the
``model_id`` field in ``model_config.json``.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIIModelConfig, MkIITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

"""Model persistence for the Mk-I tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemki.model.MkIModelConfig`
and :class:`~hmls.singlemki.model.MkITankPolicyNetwork`.

This module is registered as the ``singlemki`` entry point under
the ``hmls.models`` group.  It can also be discovered via the
``model_id`` field in ``model_config.json``.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemki.model import MkIModelConfig, MkITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIModelConfig, MkITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

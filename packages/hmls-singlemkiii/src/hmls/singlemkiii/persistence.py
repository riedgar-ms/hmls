"""Model persistence for the Mk-III tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemkiii.model.MkIIIModelConfig`
and :class:`~hmls.singlemkiii.model.MkIIITankPolicyNetwork`.

This module is registered as the ``singlemkiii`` entry point under
the ``hmls.models`` group.  It can also be discovered via the
``model_id`` field in ``model_config.json``.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemkiii.model import MkIIIModelConfig, MkIIITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIIIModelConfig, MkIIITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

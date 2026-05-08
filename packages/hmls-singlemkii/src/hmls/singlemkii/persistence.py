"""Model persistence for the Mk-II tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemkii.model.MkIIModelConfig`
and :class:`~hmls.singlemkii.model.MkIITankPolicyNetwork`.

The generic loader in :mod:`hmls.nncore.persistence` discovers this
module via the ``model_package`` field in ``model_config.json`` and
delegates all persistence operations to :data:`PERSISTENCE`.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIIModelConfig, MkIITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

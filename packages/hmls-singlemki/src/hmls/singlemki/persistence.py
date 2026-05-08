"""Model persistence for the Mk-I tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemki.model.MkIModelConfig`
and :class:`~hmls.singlemki.model.MkITankPolicyNetwork`.

The generic loader in :mod:`hmls.nncore.persistence` discovers this
module via the ``model_package`` field in ``model_config.json`` and
delegates all persistence operations to :data:`PERSISTENCE`.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemki.model import MkIModelConfig, MkITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIModelConfig, MkITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

"""Model persistence for the Mk-III tank model.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`~hmls.singlemkiii.model.MkIIIModelConfig`
and :class:`~hmls.singlemkiii.model.MkIIITankPolicyNetwork`.

The generic loader in :mod:`hmls.nncore.persistence` discovers this
module via the ``model_package`` field in ``model_config.json`` and
delegates all persistence operations to :data:`PERSISTENCE`.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.singlemkiii.model import MkIIIModelConfig, MkIIITankPolicyNetwork

PERSISTENCE = NNPlayerModelPersistence(MkIIIModelConfig, MkIIITankPolicyNetwork)

__all__ = ["PERSISTENCE"]

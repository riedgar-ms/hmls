"""Persistence module for the random tank.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`RandomTankModelConfig` and
:class:`RandomTankModel`, using :class:`RandomTankPlayer` as the
player factory.

This module is registered as the ``randomtank`` entry point under
the ``hmls.models`` group.  It can also be discovered via the
``model_id`` field in ``model_config.json``.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.randomtank.model import RandomTankModel, RandomTankModelConfig
from hmls.randomtank.player import RandomTankPlayer

PERSISTENCE = NNPlayerModelPersistence(
    RandomTankModelConfig,
    RandomTankModel,
    player_factory=lambda team, model, mode: RandomTankPlayer(
        team=team,
        model=model,
        mode=mode,
    ),
)

__all__ = ["PERSISTENCE"]

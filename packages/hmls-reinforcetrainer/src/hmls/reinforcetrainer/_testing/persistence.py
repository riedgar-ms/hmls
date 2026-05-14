"""Stub persistence module for testing.

Exposes a :data:`PERSISTENCE` instance of
:class:`~hmls.nncore.persistence.NNPlayerModelPersistence`
parameterised with :class:`StubModelConfig` and :class:`StubTankModel`,
using :class:`StubNNPlayer` as the player factory.

This enables the reinforcetrainer tests to use
:func:`~hmls.nncore.persistence.load_or_create_model` and
:func:`~hmls.nncore.persistence.create_player` without depending
on any concrete model package.

This module is registered as the ``reinforcetrainer-testing`` entry
point under the ``hmls.models`` group.
"""

from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer._testing.stub_player import StubNNPlayer

PERSISTENCE = NNPlayerModelPersistence(
    StubModelConfig,
    StubTankModel,
    player_factory=lambda team, model, mode: StubNNPlayer(
        team=team,
        model=model,
        mode=mode,
    ),
)

__all__ = ["PERSISTENCE"]

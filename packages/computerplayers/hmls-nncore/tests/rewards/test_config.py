"""Nested config JSON tests."""

from __future__ import annotations

import json

from hmls.nncore.reward_config import (
    ActionsRewardConfig,
    ExplorationRewardConfig,
    FiringRewardConfig,
    GameStateRewardConfig,
    RewardConfig,
    SituationalRewardConfig,
)


def test_reward_config_nested_json() -> None:
    """Nested config round-trips through JSON correctly."""
    config = RewardConfig(
        actions=ActionsRewardConfig(move_forward=0.04, consecutive_turn=-0.03),
        firing=FiringRewardConfig(hit=0.7),
        game_state=GameStateRewardConfig(win=2.0),
        exploration=ExplorationRewardConfig(see_cell=0.05, occupy_cell=0.1),
        situational=SituationalRewardConfig(
            enemy_in_cone=0.02, enemy_in_cone_distance_discount=0.8
        ),
    )
    data = json.loads(config.model_dump_json())
    assert data["actions"]["move_forward"] == 0.04
    assert data["actions"]["consecutive_turn"] == -0.03
    assert data["firing"]["hit"] == 0.7
    assert data["game_state"]["win"] == 2.0
    assert data["exploration"]["see_cell"] == 0.05
    assert data["exploration"]["occupy_cell"] == 0.1
    assert data["situational"]["enemy_in_cone"] == 0.02
    assert data["situational"]["enemy_in_cone_distance_discount"] == 0.8

    restored = RewardConfig.model_validate(data)
    assert restored == config

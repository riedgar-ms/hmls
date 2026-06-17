"""Firing outcome and consecutive miss reward tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hmls.core.types import Action
from hmls.nncore.reward import RewardFunction
from hmls.nncore.reward_config import (
    ActionsRewardConfig,
    FiringRewardConfig,
    GameStateRewardConfig,
    RewardConfig,
)

if TYPE_CHECKING:
    from tests.rewards.conftest import MakeEntryFactory, MakePatchFactory


def test_fire_neglect_reward_enemy_ahead(
    make_entry: MakeEntryFactory,
    make_patch_with_enemy_ahead: MakePatchFactory,
) -> None:
    """Non-fire action when enemy is directly ahead incurs neglect reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + neglect + enemy_in_cone (enemy is also in cone)
    expected = -0.01 + -0.1 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_on_fire(
    make_entry: MakeEntryFactory,
    make_patch_with_enemy_ahead: MakePatchFactory,
) -> None:
    """Fire neglect reward is NOT applied when the action was FIRE (hit)."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.FIRE, hit=True)
    patch = make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + hit + enemy_in_cone (enemy visible)
    expected = -0.01 + 0.5 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_no_enemy(
    make_entry: MakeEntryFactory,
    make_empty_patch: MakePatchFactory,
) -> None:
    """No fire neglect reward when there is no enemy directly ahead."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.TURN_LEFT)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step
    assert abs(reward - (-0.01)) < 1e-7


# ── Consecutive miss reward tests ───────────────────────────────────


def test_consecutive_miss_reward_disabled_by_default(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Default config has consecutive_miss=0.0, so no extra penalty."""
    reward_fn = RewardFunction()
    assert reward_fn.config.firing.consecutive_miss == 0.0

    reward_fn.reset()
    entry = make_entry(action=Action.FIRE, hit=False)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step (-0.01) + miss (-0.05) = -0.06, no escalating penalty
    assert abs(reward - (-0.06)) < 1e-7


def test_consecutive_miss_reward_escalates(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Consecutive misses incur escalating reward: reward × streak_count."""
    config = RewardConfig(
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # First miss: streak=1, penalty = -0.02 × 1 = -0.02
    r1 = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r1 - (-0.02)) < 1e-7

    # Second consecutive miss: streak=2, penalty = -0.02 × 2 = -0.04
    r2 = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r2 - (-0.04)) < 1e-7

    # Third consecutive miss: streak=3, penalty = -0.02 × 3 = -0.06
    r3 = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r3 - (-0.06)) < 1e-7


def test_consecutive_miss_reward_resets_on_valid_move_forward(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Valid move forward resets the miss streak to zero."""
    config = RewardConfig(
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive misses
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Move forward resets streak
    reward_fn.compute_step_reward(make_entry(action=Action.MOVE_FORWARD), patch=patch, team="alpha")

    # Next miss starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_miss_reward_resets_on_fire_hit(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire-and-hit resets the miss streak to zero."""
    config = RewardConfig(
        firing=FiringRewardConfig(hit=0.0, miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive misses
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Fire hit resets streak
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=True), patch=patch, team="alpha"
    )

    # Next miss starts fresh: streak=1
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_miss_reward_not_reset_by_pass(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Pass action does NOT reset the miss streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(pass_action=0.0),
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive misses: streak reaches 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Pass: miss streak stays at 2
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Next miss: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_miss_reward_not_reset_by_turn(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Turn action does NOT reset the miss streak."""
    config = RewardConfig(
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive misses: streak reaches 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Turn: miss streak stays at 2
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")

    # Next miss: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_miss_reward_not_reset_by_invalid_move(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Invalid move does NOT reset the miss streak."""
    config = RewardConfig(
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0, invalid_move=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive misses: streak reaches 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Invalid move forward: miss streak stays at 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.MOVE_FORWARD, valid=False), patch=patch, team="alpha"
    )

    # Next miss: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_miss_reward_resets_on_episode(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """reset() clears miss streak tracking between episodes."""
    config = RewardConfig(
        firing=FiringRewardConfig(miss=0.0, consecutive_miss=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Build up a streak
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Reset for new episode
    reward_fn.reset()

    # First miss of new episode starts fresh: streak=1
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_miss_config_round_trip() -> None:
    """consecutive_miss survives config serialisation round-trip."""
    config = RewardConfig(firing=FiringRewardConfig(consecutive_miss=-0.03))
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored.firing.consecutive_miss == -0.03

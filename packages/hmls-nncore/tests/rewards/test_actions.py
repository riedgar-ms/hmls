"""Action reward tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hmls.core.types import Action
from hmls.nncore.reward import (
    ActionsRewardConfig,
    FiringRewardConfig,
    GameStateRewardConfig,
    RewardConfig,
    RewardFunction,
)

if TYPE_CHECKING:
    from tests.rewards.conftest import MakeEntryFactory, MakePatchFactory


def test_pass_reward_deliberate(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """A deliberate PASS incurs the pass_action reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.PASS, valid=True)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + pass_action
    expected = -0.01 + -0.02
    assert abs(reward - expected) < 1e-7


def test_pass_reward_not_applied_on_invalid(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Invalid action converted to PASS does NOT incur pass_action reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD, valid=False)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move (no pass_action)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_turn_left_reward_applied(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Turn left reward is applied when the action is TURN_LEFT and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_left=0.05))
    reward_fn = RewardFunction(config=config)
    entry = make_entry(action=Action.TURN_LEFT)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.05
    assert abs(reward - expected) < 1e-7


def test_turn_right_reward_applied(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Turn right reward is applied when the action is TURN_RIGHT and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_right=0.03))
    reward_fn = RewardFunction(config=config)
    entry = make_entry(action=Action.TURN_RIGHT)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.03
    assert abs(reward - expected) < 1e-7


def test_move_forward_reward_applied(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Move forward reward is applied when the action is MOVE_FORWARD and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(move_forward=0.04))
    reward_fn = RewardFunction(config=config)
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.04
    assert abs(reward - expected) < 1e-7


def test_fire_action_reward_applied(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire action reward is applied when FIRE is valid (in addition to hit/miss)."""
    config = RewardConfig(
        actions=ActionsRewardConfig(fire=0.03),
        firing=FiringRewardConfig(hit=0.0, miss=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    entry = make_entry(action=Action.FIRE, hit=True)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - 0.03) < 1e-7


def test_turn_left_reward_not_applied_when_invalid(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Turn left reward is NOT applied when the action is invalid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_left=0.05))
    reward_fn = RewardFunction(config=config)
    entry = make_entry(action=Action.TURN_LEFT, valid=False)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move (no turn_left)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_action_rewards_default_to_zero(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Movement rewards default to zero and don't affect existing behaviour."""
    reward_fn = RewardFunction()
    assert reward_fn.config.actions.turn_left == 0.0
    assert reward_fn.config.actions.turn_right == 0.0
    assert reward_fn.config.actions.move_forward == 0.0
    assert reward_fn.config.actions.fire == 0.0

    entry = make_entry(action=Action.TURN_LEFT)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step, no movement bonus
    assert abs(reward - (-0.01)) < 1e-7


# ── Consecutive turn penalty tests ──────────────────────────────────


def test_consecutive_turn_reward_disabled_by_default(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Default config has consecutive_turn=0.0, so no extra penalty."""
    reward_fn = RewardFunction()
    assert reward_fn.config.actions.consecutive_turn == 0.0

    reward_fn.reset()
    entry = make_entry(action=Action.TURN_LEFT)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step, no escalating penalty
    assert abs(reward - (-0.01)) < 1e-7


def test_consecutive_turn_reward_escalates(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Consecutive turns incur escalating reward: reward × streak_count."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # First turn: streak=1, penalty = -0.02 × 1 = -0.02
    entry1 = make_entry(action=Action.TURN_LEFT)
    r1 = reward_fn.compute_step_reward(entry1, patch=patch, team="alpha")
    assert abs(r1 - (-0.02)) < 1e-7

    # Second consecutive turn: streak=2, penalty = -0.02 × 2 = -0.04
    entry2 = make_entry(action=Action.TURN_RIGHT)
    r2 = reward_fn.compute_step_reward(entry2, patch=patch, team="alpha")
    assert abs(r2 - (-0.04)) < 1e-7

    # Third consecutive turn: streak=3, penalty = -0.02 × 3 = -0.06
    entry3 = make_entry(action=Action.TURN_LEFT)
    r3 = reward_fn.compute_step_reward(entry3, patch=patch, team="alpha")
    assert abs(r3 - (-0.06)) < 1e-7


def test_consecutive_turn_reward_resets_on_valid_move_forward(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Valid move forward resets the streak to zero."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive turns
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Move forward resets streak
    reward_fn.compute_step_reward(make_entry(action=Action.MOVE_FORWARD), patch=patch, team="alpha")

    # Next turn starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_reward_resets_on_fire_hit(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire-and-hit resets the streak to zero."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        firing=FiringRewardConfig(hit=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive turns
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Fire hit resets streak
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=True), patch=patch, team="alpha"
    )

    # Next turn starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_fire_miss(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire-and-miss does NOT reset the streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        firing=FiringRewardConfig(miss=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Fire miss: streak stays at 2 (no penalty applied for the fire step)
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_pass(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Pass action does NOT reset the streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Pass: streak stays at 2
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_invalid_move(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Invalid move does NOT reset the streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0, invalid_move=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Invalid move forward: streak stays at 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.MOVE_FORWARD, valid=False), patch=patch, team="alpha"
    )

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_resets_on_episode(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """reset() clears streak tracking between episodes."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Build up a streak
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")

    # Reset for new episode
    reward_fn.reset()

    # First turn of new episode starts fresh: streak=1
    r = reward_fn.compute_step_reward(
        make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_config_round_trip() -> None:
    """consecutive_turn survives config serialisation round-trip."""
    config = RewardConfig(actions=ActionsRewardConfig(consecutive_turn=-0.03))
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored.actions.consecutive_turn == -0.03


# ── Consecutive pass reward tests ───────────────────────────────────


def test_consecutive_pass_reward_disabled_by_default(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Default config has consecutive_pass=0.0, so no extra penalty."""
    reward_fn = RewardFunction()
    assert reward_fn.config.actions.consecutive_pass == 0.0

    reward_fn.reset()
    entry = make_entry(action=Action.PASS)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step (-0.01) + pass_action (-0.02) = -0.03, no escalating penalty
    assert abs(reward - (-0.03)) < 1e-7


def test_consecutive_pass_reward_escalates(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Consecutive passes incur escalating reward: reward × streak_count."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # First pass: streak=1, penalty = -0.02 × 1 = -0.02
    r1 = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r1 - (-0.02)) < 1e-7

    # Second consecutive pass: streak=2, penalty = -0.02 × 2 = -0.04
    r2 = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r2 - (-0.04)) < 1e-7

    # Third consecutive pass: streak=3, penalty = -0.02 × 3 = -0.06
    r3 = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r3 - (-0.06)) < 1e-7


def test_consecutive_pass_reward_resets_on_valid_move_forward(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Valid move forward resets the pass streak to zero."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive passes
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Move forward resets streak
    reward_fn.compute_step_reward(make_entry(action=Action.MOVE_FORWARD), patch=patch, team="alpha")

    # Next pass starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_pass_reward_resets_on_fire_hit(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire-and-hit resets the pass streak to zero."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        firing=FiringRewardConfig(hit=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive passes
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Fire hit resets streak
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=True), patch=patch, team="alpha"
    )

    # Next pass starts fresh: streak=1
    r = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_pass_reward_not_reset_by_fire_miss(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Fire-and-miss does NOT reset the pass streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        firing=FiringRewardConfig(miss=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive passes: streak reaches 2
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Fire miss: streak stays at 2
    reward_fn.compute_step_reward(
        make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Next pass: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_pass_reward_not_reset_by_turn(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Turn action does NOT reset the pass streak."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Two consecutive passes: streak reaches 2
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Turn: pass streak stays at 2
    reward_fn.compute_step_reward(make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")

    # Next pass: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_pass_reward_resets_on_episode(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """reset() clears pass streak tracking between episodes."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_pass=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    reward_fn.reset()
    patch = make_empty_patch()

    # Build up a streak
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Reset for new episode
    reward_fn.reset()

    # First pass of new episode starts fresh: streak=1
    r = reward_fn.compute_step_reward(make_entry(action=Action.PASS), patch=patch, team="alpha")
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_pass_config_round_trip() -> None:
    """consecutive_pass survives config serialisation round-trip."""
    config = RewardConfig(actions=ActionsRewardConfig(consecutive_pass=-0.03))
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored.actions.consecutive_pass == -0.03

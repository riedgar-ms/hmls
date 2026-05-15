"""Basic step / hit / miss / win / loss reward tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hmls.core.types import Action, Position
from hmls.nncore.reward import (
    ActionsRewardConfig,
    ExplorationRewardConfig,
    FiringRewardConfig,
    GameStateRewardConfig,
    RewardConfig,
    RewardFunction,
    SituationalRewardConfig,
)

if TYPE_CHECKING:
    from tests.rewards.conftest import MakeEntryFactory, MakePatchAtFactory, MakePatchFactory


def test_default_reward_step_reward(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """A plain step incurs the step reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - (-0.01)) < 1e-7


def test_default_reward_hit(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """A successful hit adds the hit reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.FIRE, hit=True)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_exploration_see_cell(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """New positions add see_cell exploration bonus."""
    reward_fn = RewardFunction()
    reward_fn.reset()
    entry = make_entry()
    patch = make_empty_patch(size=3)
    # First observation discovers all cells; a 3x3 fully visible patch = 9 cells
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + see_cell * 9
    expected = -0.01 + 0.02 * 9
    assert abs(reward - expected) < 1e-7

    # Second observation of same patch discovers nothing new
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_occupy_cell_reward(
    make_entry: MakeEntryFactory, make_patch_at: MakePatchAtFactory
) -> None:
    """Moving to a new cell gives occupy_cell reward."""
    config = RewardConfig(
        exploration=ExplorationRewardConfig(see_cell=0.0, occupy_cell=0.05),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config)
    reward_fn.reset()

    # First position — new cell
    patch1 = make_patch_at(Position(0, 0), size=3)
    reward_fn.observe_patch(patch1)
    entry = make_entry(action=Action.MOVE_FORWARD)
    r1 = reward_fn.compute_step_reward(entry, patch=patch1, team="alpha")
    assert abs(r1 - 0.05) < 1e-7

    # Same position — no reward
    reward_fn.observe_patch(patch1)
    r2 = reward_fn.compute_step_reward(entry, patch=patch1, team="alpha")
    assert abs(r2) < 1e-7

    # New position — reward again
    patch2 = make_patch_at(Position(1, 0), size=3)
    reward_fn.observe_patch(patch2)
    r3 = reward_fn.compute_step_reward(entry, patch=patch2, team="alpha")
    assert abs(r3 - 0.05) < 1e-7


def test_default_reward_invalid_action(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Invalid action incurs the dedicated invalid_move reward."""
    reward_fn = RewardFunction()
    entry = make_entry(valid=False)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move
    assert abs(reward - (-0.01 + -0.1)) < 1e-7


def test_default_reward_fire_miss(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Firing and missing incurs the miss reward (penalty)."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.FIRE, hit=False)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + miss
    assert abs(reward - (-0.01 + -0.05)) < 1e-7


def test_default_reward_hit_no_miss_penalty(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """A successful hit does NOT incur the fire miss reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.FIRE, hit=True)
    patch = make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + hit only (no miss)
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_episode_win() -> None:
    """Winning gives positive terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=True) == 1.0


def test_default_reward_episode_loss() -> None:
    """Losing gives negative terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=False) == -1.0


def test_default_reward_episode_draw() -> None:
    """Draw gives zero terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=None) == 0.0


def test_reward_from_config() -> None:
    """Construction from an explicit config works correctly."""
    config = RewardConfig(
        firing=FiringRewardConfig(hit=1.0),
        game_state=GameStateRewardConfig(step=-0.05),
    )
    reward_fn = RewardFunction(config=config)
    assert reward_fn.config.firing.hit == 1.0
    assert reward_fn.config.game_state.step == -0.05
    # Other defaults are preserved
    assert reward_fn.config.game_state.win == 1.0
    assert reward_fn.config.exploration.see_cell == 0.02


def test_reward_config_round_trip() -> None:
    """Config survives serialisation round-trip via model_dump/model_validate."""
    config = RewardConfig(
        firing=FiringRewardConfig(hit=0.8, miss=-0.08, neglect=-0.15),
        game_state=GameStateRewardConfig(
            win=2.0, loss=-2.0, step=-0.02, invalid_move=-0.2, death=-0.5
        ),
        actions=ActionsRewardConfig(pass_action=-0.03),
        exploration=ExplorationRewardConfig(see_cell=0.05),
        situational=SituationalRewardConfig(enemy_in_cone=0.05),
    )
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored == config


def test_reward_config_json_round_trip() -> None:
    """Config survives JSON serialisation round-trip."""
    config = RewardConfig(firing=FiringRewardConfig(hit=0.75))
    json_str = config.model_dump_json()
    restored = RewardConfig.model_validate_json(json_str)
    assert restored == config


def test_reward_config_is_frozen() -> None:
    """Config should be immutable."""
    config = RewardConfig()
    try:
        config.actions = ActionsRewardConfig()  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except TypeError, ValueError, AttributeError:
        pass

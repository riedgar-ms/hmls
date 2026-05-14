"""Reset / state tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hmls.core.types import Action, Position
from hmls.nncore.reward import (
    ActionsRewardConfig,
    ExplorationRewardConfig,
    RewardConfig,
    RewardFunction,
)

if TYPE_CHECKING:
    from tests.rewards.conftest import MakeEntryFactory, MakePatchAtFactory


def test_reset_clears_all_state(
    make_entry: MakeEntryFactory, make_patch_at: MakePatchAtFactory
) -> None:
    """Reset clears seen, occupied, and streak state.

    After reset, exploration bonuses should be re-earned for the same
    positions, and streak counters should restart from zero.
    """
    cfg = RewardConfig(
        exploration=ExplorationRewardConfig(see_cell=0.1, occupy_cell=0.5),
        actions=ActionsRewardConfig(consecutive_turn=-0.1),
    )
    reward_fn = RewardFunction(cfg)

    # Build up state: observe some patches and generate a turn streak
    patch = make_patch_at(Position(0, 0))
    reward_fn.observe_patch(patch)
    entry_turn = make_entry(action=Action.TURN_LEFT)
    first_reward = reward_fn.compute_step_reward(entry_turn, patch, "alpha")

    # Second turn in streak should have higher penalty
    reward_fn.observe_patch(patch)
    second_reward = reward_fn.compute_step_reward(entry_turn, patch, "alpha")
    assert second_reward < first_reward  # streak penalty escalates

    # After reset, same patch should give exploration bonuses again
    reward_fn.reset()
    reward_fn.observe_patch(patch)
    reward_after_reset = reward_fn.compute_step_reward(entry_turn, patch, "alpha")
    # Should match first_reward since streak and exploration reset
    assert abs(reward_after_reset - first_reward) < 1e-9

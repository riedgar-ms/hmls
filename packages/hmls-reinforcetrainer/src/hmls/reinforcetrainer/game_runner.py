"""Game runner: sets up and plays a single training game.

Generates a map, places tanks, creates player instances via dynamic
dispatch, runs the game via :class:`GameEngine`, and computes rewards
for each player's trajectory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from hmls.core.engine import GameEngine, GameResult, HistoryEntry
from hmls.core.map import GameMap
from hmls.core.placement import place_tanks
from hmls.mapgenerator import StrategyConfigBase, generate_map_from_config
from hmls.nncore.model import TankModelBase
from hmls.nncore.persistence import create_player
from hmls.nncore.player import NNPlayerBase
from hmls.nncore.reward import RewardFunction
from hmls.reinforcetrainer.lethargy import LethargyPolicy


@dataclass
class GameOutcome:
    """Result of a single training game.

    Attributes:
        result: The full GameResult (map, history, winner).
        player_a: The NNPlayer for team A (with trajectory if learning).
        player_b: The NNPlayer for team B (with trajectory if learning).
        lethargy_loser: Team that lost due to lethargy (e.g. ``"A"``),
            or ``None`` if the game ended normally.
    """

    result: GameResult
    player_a: NNPlayerBase
    player_b: NNPlayerBase
    lethargy_loser: str | None = None


def create_map(
    width: int,
    height: int,
    impassable_fraction: float,
    strategy_config: StrategyConfigBase,
    seed: int | None = None,
) -> GameMap:
    """Generate a random map using the specified strategy configuration.

    Args:
        width: Map width in cells.
        height: Map height in cells.
        impassable_fraction: Fraction of cells to make impassable.
        strategy_config: A concrete :class:`StrategyConfigBase` subclass
            instance describing which strategy and parameters to use.
        seed: Optional random seed.

    Returns:
        A newly generated GameMap.
    """
    return generate_map_from_config(
        width,
        height,
        impassable_fraction=impassable_fraction,
        strategy_config=strategy_config,
        seed=seed,
    )


def run_game(
    game_map: GameMap,
    model_a: TankModelBase,
    model_b: TankModelBase,
    *,
    train_a: bool = True,
    train_b: bool = True,
    max_turns: int = 200,
    patch_size: int = 9,
    reward_fn_a: RewardFunction | None = None,
    reward_fn_b: RewardFunction | None = None,
    lethargy_policy: LethargyPolicy | None = None,
    rng: random.Random | None = None,
) -> GameOutcome:
    """Run a single game between two NN models.

    Creates player instances via the model package registry (each
    model's ``model_id`` determines the concrete player type),
    places tanks, runs the game engine, and computes step-by-step
    rewards for learning players.

    Args:
        game_map: The map to play on.
        model_a: Neural network for team A.
        model_b: Neural network for team B.
        train_a: Whether player A is in learn mode.
        train_b: Whether player B is in learn mode.
        max_turns: Maximum turns before the game is a draw.
        patch_size: Side length of visibility patches for the game engine.
        reward_fn_a: Reward function for player A (defaults to RewardFunction()).
        reward_fn_b: Reward function for player B (defaults to RewardFunction()).
        lethargy_policy: Optional policy for detecting degenerate play
            (e.g. spinning in place).  If triggered, the offending tank's
            team loses immediately.
        rng: Random number generator for tank placement.

    Returns:
        A GameOutcome with the result and player references.
    """
    if reward_fn_a is None:
        reward_fn_a = RewardFunction()
    if reward_fn_b is None:
        reward_fn_b = RewardFunction()

    # NOTE (REINFORCE_AUDIT item C): If future models use dropout or
    # batch normalisation, add model.train() / model.eval() calls
    # here — model.train() for trainable models, model.eval() for
    # frozen ones.  Currently unnecessary because no model uses these
    # layers, but this is the correct place for the calls.
    player_a = create_player(
        model_id=model_a.config.model_id,
        team="A",
        model=model_a,
        mode="learn" if train_a else "play",
    )
    player_b = create_player(
        model_id=model_b.config.model_id,
        team="B",
        model=model_b,
        mode="learn" if train_b else "play",
    )

    # Reset episode state
    player_a.reset_episode()
    player_b.reset_episode()
    reward_fn_a.reset()
    reward_fn_b.reset()

    # Place tanks (1 per team for single-tank training)
    placement_seed = rng.randint(0, 2**31) if rng else None
    tanks = place_tanks(game_map, tanks_per_player=1, seed=placement_seed)

    engine = GameEngine(
        game_map=game_map,
        tanks=tanks,
        players={"A": player_a, "B": player_b},
        max_turns=max_turns,
        patch_size=patch_size,
    )

    if lethargy_policy is not None:
        lethargy_policy.reset()

    # Step through the game, computing rewards after each step
    lethargy_loser: str | None = None
    while not engine.game_over:
        entry: HistoryEntry = engine.step()

        # Determine which player just acted
        acting_team = entry.tank_id[0]  # Tank IDs are like "A1", "B1"
        if acting_team == "A" and train_a:
            _assign_step_reward(player_a, entry, reward_fn_a)
        elif acting_team == "B" and train_b:
            _assign_step_reward(player_b, entry, reward_fn_b)

        # Check lethargy policy
        if lethargy_policy is not None:
            lethargy_loser = lethargy_policy.observe_action(entry)
            if lethargy_loser is not None:
                break

    # Compute end-of-episode rewards
    if lethargy_loser is not None:
        # A tank was caught being lethargic — build the result with the
        # *other* team as winner.
        lethargy_winner = "B" if lethargy_loser == "A" else "A"
        partial_result = engine.make_result()
        result = GameResult(
            winner=lethargy_winner,
            game_map=partial_result.game_map,
            initial_state=partial_result.initial_state,
            history=partial_result.history,
            turns_played=partial_result.turns_played,
        )
        # Only penalise the lethargic player.  The "winning" player
        # receives no end-of-episode reward because it did not earn
        # the win — the opponent simply self-destructed, and the
        # "winner" may itself have been playing poorly.
        _apply_lethargy_loss(
            loser_team=lethargy_loser,
            player_a=player_a,
            player_b=player_b,
            train_a=train_a,
            train_b=train_b,
            reward_fn_a=reward_fn_a,
            reward_fn_b=reward_fn_b,
        )
    else:
        result = engine.make_result()
        winner = result.winner
        _apply_normal_end_rewards(
            winner=winner,
            player_a=player_a,
            player_b=player_b,
            train_a=train_a,
            train_b=train_b,
            reward_fn_a=reward_fn_a,
            reward_fn_b=reward_fn_b,
        )

    return GameOutcome(
        result=result,
        player_a=player_a,
        player_b=player_b,
        lethargy_loser=lethargy_loser,
    )


def _assign_step_reward(
    player: NNPlayerBase,
    entry: HistoryEntry,
    reward_fn: RewardFunction,
) -> None:
    """Compute and assign the reward for the most recent step.

    Args:
        player: The NN player that just acted.
        entry: The history entry from the engine.
        reward_fn: Reward function to compute the reward.
    """
    patch = player.last_patch
    assert patch is not None, "Player must have seen a patch before reward is computed"
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(
        entry,
        patch,
        player.team,
    )
    step_idx = len(player.episode) - 1
    if step_idx >= 0:
        player.episode.set_reward(step_idx, reward)


def _apply_normal_end_rewards(
    winner: str | None,
    player_a: NNPlayerBase,
    player_b: NNPlayerBase,
    train_a: bool,
    train_b: bool,
    reward_fn_a: RewardFunction,
    reward_fn_b: RewardFunction,
) -> None:
    """Apply end-of-episode rewards for a game that ended normally.

    Args:
        winner: Winning team name, or ``None`` for a draw.
        player_a: Player A instance.
        player_b: Player B instance.
        train_a: Whether player A is training.
        train_b: Whether player B is training.
        reward_fn_a: Reward function for player A.
        reward_fn_b: Reward function for player B.
    """
    if train_a:
        won_a = True if winner == "A" else (False if winner == "B" else None)
        end_reward_a = reward_fn_a.compute_episode_end_reward(won_a)
        if len(player_a.episode) > 0:
            last_idx = len(player_a.episode) - 1
            current = player_a.episode.steps[last_idx].reward
            player_a.episode.set_reward(last_idx, current + end_reward_a)

    if train_b:
        won_b = True if winner == "B" else (False if winner == "A" else None)
        end_reward_b = reward_fn_b.compute_episode_end_reward(won_b)
        if len(player_b.episode) > 0:
            last_idx = len(player_b.episode) - 1
            current = player_b.episode.steps[last_idx].reward
            player_b.episode.set_reward(last_idx, current + end_reward_b)


def _apply_lethargy_loss(
    loser_team: str,
    player_a: NNPlayerBase,
    player_b: NNPlayerBase,
    train_a: bool,
    train_b: bool,
    reward_fn_a: RewardFunction,
    reward_fn_b: RewardFunction,
) -> None:
    """Apply end-of-episode reward when a game ends due to lethargy.

    Only the lethargic player receives a loss reward.  The opponent
    receives **no** end-of-episode reward: it did not earn the win —
    the opponent simply self-destructed, and the "winner" may itself
    have been playing poorly or even spinning in the same way.

    Args:
        loser_team: Team name of the lethargic player (``"A"`` or ``"B"``).
        player_a: Player A instance.
        player_b: Player B instance.
        train_a: Whether player A is training.
        train_b: Whether player B is training.
        reward_fn_a: Reward function for player A.
        reward_fn_b: Reward function for player B.
    """
    if loser_team == "A" and train_a:
        end_reward = reward_fn_a.compute_episode_end_reward(won=False)
        if len(player_a.episode) > 0:
            last_idx = len(player_a.episode) - 1
            current = player_a.episode.steps[last_idx].reward
            player_a.episode.set_reward(last_idx, current + end_reward)
    elif loser_team == "B" and train_b:
        end_reward = reward_fn_b.compute_episode_end_reward(won=False)
        if len(player_b.episode) > 0:
            last_idx = len(player_b.episode) - 1
            current = player_b.episode.steps[last_idx].reward
            player_b.episode.set_reward(last_idx, current + end_reward)


def save_sample_game(result: GameResult, directory: Path, game_number: int) -> Path:
    """Save a game result as a JSON replay file.

    Args:
        result: The game result to save.
        directory: Directory to write the file into.
        game_number: Used to generate the filename.

    Returns:
        The path to the saved file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"game_{game_number:06d}.json"
    filepath = directory / filename
    filepath.write_text(result.model_dump_json(indent=2))
    return filepath

"""Game runner: sets up and plays a single training game.

Generates a map, places tanks, creates :class:`NNPlayer` instances,
runs the game via :class:`GameEngine`, and computes rewards for each
player's trajectory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from hmls.core.engine import GameEngine, GameResult, HistoryEntry
from hmls.core.map import GameMap
from hmls.core.placement import place_tanks
from hmls.mapgenerator import STRATEGY_REGISTRY, MapStrategy, generate_map
from hmls.singletanknn.model import TankPolicyNetwork
from hmls.singletanknn.player import NNPlayer
from hmls.singletanknn.reward import DefaultReward, RewardFunction


@dataclass
class GameOutcome:
    """Result of a single training game.

    Attributes:
        result: The full GameResult (map, history, winner).
        player_a: The NNPlayer for team A (with trajectory if learning).
        player_b: The NNPlayer for team B (with trajectory if learning).
    """

    result: GameResult
    player_a: NNPlayer
    player_b: NNPlayer


def create_map(
    width: int,
    height: int,
    impassable_fraction: float,
    strategy_name: str,
    seed: int | None = None,
) -> GameMap:
    """Generate a random map using the specified strategy.

    Args:
        width: Map width in cells.
        height: Map height in cells.
        impassable_fraction: Fraction of cells to make impassable.
        strategy_name: Name of a registered map strategy.
        seed: Optional random seed.

    Returns:
        A newly generated GameMap.

    Raises:
        KeyError: If strategy_name is not in the registry.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Unknown map strategy '{strategy_name}'. Available: {available}")
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    strategy: MapStrategy = strategy_cls()
    return generate_map(
        width,
        height,
        impassable_fraction=impassable_fraction,
        seed=seed,
        strategy=strategy,
    )


def run_game(
    game_map: GameMap,
    model_a: TankPolicyNetwork,
    model_b: TankPolicyNetwork,
    *,
    train_a: bool = True,
    train_b: bool = True,
    max_turns: int = 200,
    reward_fn: RewardFunction | None = None,
    rng: random.Random | None = None,
) -> GameOutcome:
    """Run a single game between two NN models.

    Creates NNPlayer instances, places tanks, runs the game engine,
    and computes step-by-step rewards for learning players.

    Args:
        game_map: The map to play on.
        model_a: Neural network for team A.
        model_b: Neural network for team B.
        train_a: Whether player A is in learn mode.
        train_b: Whether player B is in learn mode.
        max_turns: Maximum turns before the game is a draw.
        reward_fn: Reward function to use (defaults to DefaultReward).
        rng: Random number generator for tank placement.

    Returns:
        A GameOutcome with the result and player references.
    """
    if reward_fn is None:
        reward_fn = DefaultReward()

    player_a = NNPlayer(
        team="A",
        model=model_a,
        mode="learn" if train_a else "play",
    )
    player_b = NNPlayer(
        team="B",
        model=model_b,
        mode="learn" if train_b else "play",
    )

    # Reset episode state
    player_a.reset_episode()
    player_b.reset_episode()

    # Place tanks (1 per team for single-tank training)
    placement_seed = rng.randint(0, 2**31) if rng else None
    tanks = place_tanks(game_map, tanks_per_player=1, seed=placement_seed)

    engine = GameEngine(
        game_map=game_map,
        tanks=tanks,
        players={"A": player_a, "B": player_b},
        max_turns=max_turns,
    )

    # Step through the game, computing rewards after each step
    while not engine.game_over:
        entry: HistoryEntry = engine.step()

        # Determine which player just acted
        acting_team = entry.tank_id[0]  # Tank IDs are like "A1", "B1"
        if acting_team == "A" and train_a:
            _assign_step_reward(player_a, entry, reward_fn)
        elif acting_team == "B" and train_b:
            _assign_step_reward(player_b, entry, reward_fn)

    # Compute end-of-episode rewards
    result = engine.make_result()
    winner = result.winner

    if train_a:
        won_a = True if winner == "A" else (False if winner == "B" else None)
        end_reward_a = reward_fn.compute_episode_end_reward(won_a, len(player_a.explored_positions))
        if len(player_a.episode) > 0:
            last_idx = len(player_a.episode) - 1
            current = player_a.episode.steps[last_idx].reward
            player_a.episode.set_reward(last_idx, current + end_reward_a)

    if train_b:
        won_b = True if winner == "B" else (False if winner == "A" else None)
        end_reward_b = reward_fn.compute_episode_end_reward(won_b, len(player_b.explored_positions))
        if len(player_b.episode) > 0:
            last_idx = len(player_b.episode) - 1
            current = player_b.episode.steps[last_idx].reward
            player_b.episode.set_reward(last_idx, current + end_reward_b)

    return GameOutcome(result=result, player_a=player_a, player_b=player_b)


def _assign_step_reward(
    player: NNPlayer,
    entry: HistoryEntry,
    reward_fn: RewardFunction,
) -> None:
    """Compute and assign the reward for the most recent step.

    Args:
        player: The NNPlayer that just acted.
        entry: The history entry from the engine.
        reward_fn: Reward function to compute the reward.
    """
    reward = reward_fn.compute_step_reward(
        entry,
        player.explored_positions,
        player.last_step_new_positions(),
    )
    step_idx = len(player.episode) - 1
    if step_idx >= 0:
        player.episode.set_reward(step_idx, reward)


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

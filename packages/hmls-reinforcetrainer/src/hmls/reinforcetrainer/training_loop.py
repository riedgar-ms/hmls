"""Main training loop orchestration.

Coordinates map generation, game execution, policy updates, and
periodic saving of weights and sample replays.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from hmls.reinforcetrainer.config import TrainerConfig
from hmls.reinforcetrainer.game_runner import (
    GameOutcome,
    create_map,
    run_game,
    save_sample_game,
)
from hmls.reinforcetrainer.updater import reinforce_update
from hmls.singletanknn.model import ModelConfig, TankPolicyNetwork
from hmls.singletanknn.persistence import load_model, save_model


def load_or_create_model(model_dir: Path) -> TankPolicyNetwork:
    """Load an existing model from a directory or create a fresh one.

    Looks for a ``model.pt`` file in the directory.  If found, loads it.
    Otherwise creates a new model with default configuration.

    Args:
        model_dir: Directory that may contain a ``model.pt`` file.

    Returns:
        A TankPolicyNetwork (either loaded or freshly initialised).
    """
    model_path = model_dir / "model.pt"
    if model_path.exists():
        model, _metadata = load_model(model_path)
        return model
    return TankPolicyNetwork(ModelConfig())


def _save_weights(
    model: TankPolicyNetwork,
    model_dir: Path,
    games_played: int,
) -> None:
    """Save model weights to the model directory.

    Args:
        model: The model to save.
        model_dir: Target directory.
        games_played: Current game count (stored as metadata).
    """
    save_model(
        model,
        model_dir / "model.pt",
        metadata={"games_played": games_played},
    )


def train(config: TrainerConfig) -> None:
    """Run the full training loop.

    Args:
        config: Training configuration.
    """
    # Seed
    rng = random.Random(config.seed)
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Load or create models
    model_a = load_or_create_model(config.model_a_dir)
    model_b = load_or_create_model(config.model_b_dir)

    # Set up optimizers for models that are training
    optimizer_a = (
        torch.optim.Adam(model_a.parameters(), lr=config.learning_rate) if config.train_a else None
    )
    optimizer_b = (
        torch.optim.Adam(model_b.parameters(), lr=config.learning_rate) if config.train_b else None
    )

    # Training stats
    total_games = 0
    wins_a = 0
    wins_b = 0
    draws = 0
    total_loss_a = 0.0
    total_loss_b = 0.0

    total_games_planned = config.total_maps * config.games_per_map
    print(
        f"Starting training: {config.total_maps} maps × "
        f"{config.games_per_map} games/map = {total_games_planned} total games"
    )
    print(f"  Train A: {config.train_a}, Train B: {config.train_b}")
    print(
        f"  Map size: {config.map_width}×{config.map_height}, "
        f"impassable: {config.impassable_fraction:.0%}"
    )
    print(f"  Max turns/game: {config.max_turns}, γ={config.gamma}, lr={config.learning_rate}")
    print()

    for map_idx in range(config.total_maps):
        # Generate a new map
        map_seed = rng.randint(0, 2**31)
        game_map = create_map(
            config.map_width,
            config.map_height,
            config.impassable_fraction,
            config.map_strategy,
            seed=map_seed,
        )

        for game_idx in range(config.games_per_map):
            total_games += 1

            # Run game
            outcome: GameOutcome = run_game(
                game_map,
                model_a,
                model_b,
                train_a=config.train_a,
                train_b=config.train_b,
                max_turns=config.max_turns,
                rng=rng,
            )

            # Track wins
            winner = outcome.result.winner
            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1
            else:
                draws += 1

            # Policy gradient updates
            if config.train_a and optimizer_a is not None:
                loss_a = reinforce_update(
                    outcome.player_a.episode,
                    optimizer_a,
                    config.gamma,
                    log_prob_tensors=outcome.player_a.log_prob_tensors,
                )
                total_loss_a += loss_a

            if config.train_b and optimizer_b is not None:
                loss_b = reinforce_update(
                    outcome.player_b.episode,
                    optimizer_b,
                    config.gamma,
                    log_prob_tensors=outcome.player_b.log_prob_tensors,
                )
                total_loss_b += loss_b

            # Save sample game
            if total_games % config.sample_game_interval == 0:
                save_sample_game(
                    outcome.result,
                    config.sample_game_dir,
                    total_games,
                )

            # Save weights periodically
            if total_games % config.save_weights_interval == 0:
                if config.train_a:
                    _save_weights(model_a, config.model_a_dir, total_games)
                if config.train_b:
                    _save_weights(model_b, config.model_b_dir, total_games)

            # Progress logging
            if total_games % config.games_per_map == 0:
                _log_progress(
                    total_games,
                    total_games_planned,
                    map_idx + 1,
                    wins_a,
                    wins_b,
                    draws,
                    total_loss_a,
                    total_loss_b,
                    config.train_a,
                    config.train_b,
                )

    # Final save
    if config.train_a:
        _save_weights(model_a, config.model_a_dir, total_games)
    if config.train_b:
        _save_weights(model_b, config.model_b_dir, total_games)

    print(f"\nTraining complete. {total_games} games played.")
    print(f"  Final record: A wins={wins_a}, B wins={wins_b}, draws={draws}")


def _log_progress(
    total_games: int,
    planned: int,
    maps_done: int,
    wins_a: int,
    wins_b: int,
    draws: int,
    loss_a: float,
    loss_b: float,
    train_a: bool,
    train_b: bool,
) -> None:
    """Print a progress line to stdout.

    Args:
        total_games: Games completed so far.
        planned: Total games planned.
        maps_done: Number of maps completed.
        wins_a: Cumulative wins for team A.
        wins_b: Cumulative wins for team B.
        draws: Cumulative draws.
        loss_a: Cumulative loss for model A.
        loss_b: Cumulative loss for model B.
        train_a: Whether model A is training.
        train_b: Whether model B is training.
    """
    pct = 100.0 * total_games / planned if planned > 0 else 0.0
    avg_loss_a = loss_a / total_games if train_a and total_games > 0 else 0.0
    avg_loss_b = loss_b / total_games if train_b and total_games > 0 else 0.0

    parts = [
        f"[{pct:5.1f}%]",
        f"games={total_games}/{planned}",
        f"maps={maps_done}",
        f"A={wins_a}",
        f"B={wins_b}",
        f"draws={draws}",
    ]
    if train_a:
        parts.append(f"loss_a={avg_loss_a:.4f}")
    if train_b:
        parts.append(f"loss_b={avg_loss_b:.4f}")

    print("  ".join(parts), flush=True)

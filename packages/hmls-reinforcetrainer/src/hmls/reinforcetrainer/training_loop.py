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
from hmls.singletanknn.persistence import (
    load_model,
    load_model_config,
    load_reward_config,
    save_model,
)
from hmls.singletanknn.reward import DefaultReward, RewardFunction


def load_or_create_model(model_dir: Path) -> TankPolicyNetwork:
    """Load an existing model from a directory or create a fresh one.

    Reads ``model_config.json`` from the directory (must exist).  If a
    ``model.pt`` file is also present, loads the trained weights.
    Otherwise creates a new model with the configuration from the JSON.

    Args:
        model_dir: Directory containing ``model_config.json`` and
            optionally ``model.pt``.

    Returns:
        A TankPolicyNetwork (either loaded or freshly initialised).

    Raises:
        FileNotFoundError: If ``model_config.json`` is missing.
    """
    config = load_model_config(model_dir)

    model_path = model_dir / "model.pt"
    if model_path.exists():
        model, _metadata = load_model(model_path)
        return model
    return TankPolicyNetwork(config)


def _validate_model_configs(config_a: ModelConfig, config_b: ModelConfig) -> None:
    """Validate that the two model configurations are compatible.

    The models may differ in ``cnn_channels`` and ``gru_hidden_size``,
    but ``patch_size`` must be identical (it determines observation
    encoding size).

    Args:
        config_a: Configuration for model A.
        config_b: Configuration for model B.

    Raises:
        ValueError: If configurations are incompatible.
    """
    if config_a.patch_size != config_b.patch_size:
        raise ValueError(
            f"Model configurations are incompatible: patch_size differs "
            f"(A={config_a.patch_size}, B={config_b.patch_size}). "
            f"Both models must use the same patch_size."
        )


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

    Raises:
        FileNotFoundError: If model directories lack required config files.
        ValueError: If model configurations are incompatible.
    """
    # Seed
    rng = random.Random(config.hyperparameters.seed)
    if config.hyperparameters.seed is not None:
        torch.manual_seed(config.hyperparameters.seed)

    # Load model configs and validate compatibility
    model_config_a = load_model_config(config.model_a.dir)
    model_config_b = load_model_config(config.model_b.dir)
    _validate_model_configs(model_config_a, model_config_b)

    # Load reward configs
    reward_config_a = load_reward_config(config.model_a.dir)
    reward_config_b = load_reward_config(config.model_b.dir)
    reward_fn_a: RewardFunction = DefaultReward(reward_config_a)
    reward_fn_b: RewardFunction = DefaultReward(reward_config_b)

    # Load or create models
    model_a = load_or_create_model(config.model_a.dir)
    model_b = load_or_create_model(config.model_b.dir)

    # Set up optimizers for models that are training
    optimizer_a = (
        torch.optim.Adam(model_a.parameters(), lr=config.hyperparameters.learning_rate)
        if config.model_a.train
        else None
    )
    optimizer_b = (
        torch.optim.Adam(model_b.parameters(), lr=config.hyperparameters.learning_rate)
        if config.model_b.train
        else None
    )

    # Training stats
    total_games = 0
    wins_a = 0
    wins_b = 0
    draws = 0
    total_loss_a = 0.0
    total_loss_b = 0.0

    total_games_planned = config.game.total_maps * config.game.games_per_map
    print(
        f"Starting training: {config.game.total_maps} maps × "
        f"{config.game.games_per_map} games/map = {total_games_planned} total games"
    )
    print(f"  Train A: {config.model_a.train}, Train B: {config.model_b.train}")
    print(
        f"  Map size: {config.map.min_size}–{config.map.max_size} (random), "
        f"impassable: {config.map.impassable_fraction:.0%}"
    )
    print(
        f"  Max turns/game: {config.game.max_turns}, "
        f"γ={config.hyperparameters.gamma}, lr={config.hyperparameters.learning_rate}"
    )
    print()

    for map_idx in range(config.game.total_maps):
        # Generate a new map
        map_seed = rng.randint(0, 2**31)
        map_width = rng.randint(config.map.min_size, config.map.max_size)
        map_height = rng.randint(config.map.min_size, config.map.max_size)
        game_map = create_map(
            map_width,
            map_height,
            config.map.impassable_fraction,
            config.map.strategy,
            seed=map_seed,
        )

        for game_idx in range(config.game.games_per_map):
            total_games += 1

            # Run game
            outcome: GameOutcome = run_game(
                game_map,
                model_a,
                model_b,
                train_a=config.model_a.train,
                train_b=config.model_b.train,
                max_turns=config.game.max_turns,
                reward_fn_a=reward_fn_a,
                reward_fn_b=reward_fn_b,
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
            if config.model_a.train and optimizer_a is not None:
                loss_a = reinforce_update(
                    outcome.player_a.episode,
                    optimizer_a,
                    config.hyperparameters.gamma,
                    log_prob_tensors=outcome.player_a.log_prob_tensors,
                )
                total_loss_a += loss_a

            if config.model_b.train and optimizer_b is not None:
                loss_b = reinforce_update(
                    outcome.player_b.episode,
                    optimizer_b,
                    config.hyperparameters.gamma,
                    log_prob_tensors=outcome.player_b.log_prob_tensors,
                )
                total_loss_b += loss_b

            # Save sample game
            if total_games % config.output.sample_game_interval == 0:
                save_sample_game(
                    outcome.result,
                    config.output.sample_game_dir,
                    total_games,
                )

            # Save weights periodically
            if total_games % config.output.save_weights_interval == 0:
                if config.model_a.train:
                    _save_weights(model_a, config.model_a.dir, total_games)
                if config.model_b.train:
                    _save_weights(model_b, config.model_b.dir, total_games)

            # Progress logging
            if total_games % config.game.games_per_map == 0:
                _log_progress(
                    total_games,
                    total_games_planned,
                    map_idx + 1,
                    wins_a,
                    wins_b,
                    draws,
                    total_loss_a,
                    total_loss_b,
                    config.model_a.train,
                    config.model_b.train,
                )

    # Final save
    if config.model_a.train:
        _save_weights(model_a, config.model_a.dir, total_games)
    if config.model_b.train:
        _save_weights(model_b, config.model_b.dir, total_games)

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

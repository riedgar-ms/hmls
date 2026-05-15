"""Main training loop orchestration.

Coordinates map generation, game execution, policy updates, and
periodic saving of weights and sample replays.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.persistence import (
    load_model_config,
    load_or_create_model,
    save_model,
)
from hmls.nncore.reward import RewardFunction
from hmls.reinforcetrainer.config import LethargyConfig, TrainerConfig
from hmls.reinforcetrainer.game_runner import (
    GameOutcome,
    create_map,
    run_game,
    save_sample_game,
)
from hmls.reinforcetrainer.lethargy import (
    ConsecutiveTurnLimit,
    LethargyPolicy,
    NoLethargyCheck,
)
from hmls.reinforcetrainer.updater import ReturnBaseline, reinforce_update

logger = logging.getLogger(__name__)


def _create_lethargy_policy(config: LethargyConfig) -> LethargyPolicy:
    """Instantiate a lethargy policy from configuration.

    Args:
        config: The lethargy section of the trainer config.

    Returns:
        A :class:`LethargyPolicy` instance matching the configured policy.

    Raises:
        ValueError: If the policy name is unrecognised.
    """
    if config.policy == "none":
        return NoLethargyCheck()
    elif config.policy == "consecutive_turn_limit":
        return ConsecutiveTurnLimit(
            max_consecutive_turns=config.max_consecutive_turns,
        )
    else:
        raise ValueError(f"Unknown lethargy policy: {config.policy!r}")


def _validate_model_configs(config_a: TankModelConfig, config_b: TankModelConfig) -> None:
    """Validate that the two model configurations are compatible.

    The models may differ in architecture (and even type), but
    ``patch_size`` must be identical (it determines observation
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


def _validate_game_patch_size(
    game_patch_size: int,
    model_config_a: TankModelConfig,
    model_config_b: TankModelConfig,
) -> None:
    """Validate that the game patch_size matches both model configs.

    Args:
        game_patch_size: The patch_size from GameConfig.
        model_config_a: Configuration for model A.
        model_config_b: Configuration for model B.

    Raises:
        ValueError: If the game patch_size doesn't match a model's patch_size.
    """
    if game_patch_size != model_config_a.patch_size:
        raise ValueError(
            f"GameConfig patch_size ({game_patch_size}) does not match "
            f"model A patch_size ({model_config_a.patch_size}). "
            f"The game and model configurations must agree on patch_size."
        )
    if game_patch_size != model_config_b.patch_size:
        raise ValueError(
            f"GameConfig patch_size ({game_patch_size}) does not match "
            f"model B patch_size ({model_config_b.patch_size}). "
            f"The game and model configurations must agree on patch_size."
        )


def _save_weights(
    model: TankModelBase,
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
    logger.info(
        "Model A config: model_id=%s, dir=%s",
        model_config_a.model_id,
        config.model_a.dir,
    )
    logger.info(
        "Model B config: model_id=%s, dir=%s",
        model_config_b.model_id,
        config.model_b.dir,
    )
    _validate_model_configs(model_config_a, model_config_b)
    _validate_game_patch_size(config.game.patch_size, model_config_a, model_config_b)

    # Create reward functions from training config
    reward_fn_a: RewardFunction = RewardFunction(config.model_a.reward)
    reward_fn_b: RewardFunction = RewardFunction(config.model_b.reward)

    # Instantiate lethargy policy
    lethargy_policy = _create_lethargy_policy(config.lethargy)

    # Load or create models
    model_a = load_or_create_model(config.model_a.dir)
    model_b = load_or_create_model(config.model_b.dir)

    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    logger.info("Model A trainable weights: %d", params_a)
    logger.info("Model B trainable weights: %d", params_b)

    # NOTE (REINFORCE_AUDIT item C): If future models use dropout or
    # batch normalisation, call model_a.train() / model_b.train() for
    # trainable models and model_a.eval() / model_b.eval() for frozen
    # ones here (and in game_runner.py where players are created).
    # Currently unnecessary because no model uses these layers.

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

    # ── Cross-episode baselines ──────────────────────────────────
    #
    # One baseline per trainable player.  Each tracks a running mean
    # of discounted returns across episodes so that advantages reflect
    # how good an episode was *relative to the long-run average*,
    # rather than being normalised within each episode (which destroys
    # the signal on degenerate episodes — see ReturnBaseline docstring).
    #
    baseline_a = (
        ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
        if config.model_a.train
        else None
    )
    baseline_b = (
        ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
        if config.model_b.train
        else None
    )

    # Training stats
    total_games = 0
    wins_a = 0
    wins_b = 0
    draws = 0
    lethargy_a = 0
    lethargy_b = 0
    total_loss_a = 0.0
    total_loss_b = 0.0

    total_games_planned = config.game.total_maps * config.game.games_per_map
    logger.info(
        "Starting training: %d maps × %d games/map = %d total games",
        config.game.total_maps,
        config.game.games_per_map,
        total_games_planned,
    )
    logger.info("  Train A: %s, Train B: %s", config.model_a.train, config.model_b.train)
    logger.info(
        "  Map size: %d–%d (random), impassable: %.0f%%",
        config.map.min_size,
        config.map.max_size,
        config.map.impassable_fraction * 100,
    )
    logger.info(
        "  Max turns/game: %d, γ=%s, lr=%s",
        config.game.max_turns,
        config.hyperparameters.gamma,
        config.hyperparameters.learning_rate,
    )

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
        logger.debug(
            "Map %d/%d: %dx%d, seed=%d, strategy=%s",
            map_idx + 1,
            config.game.total_maps,
            map_width,
            map_height,
            map_seed,
            config.map.strategy,
        )

        for _game_idx in range(config.game.games_per_map):
            total_games += 1

            # Run game
            outcome: GameOutcome = run_game(
                game_map,
                model_a,
                model_b,
                train_a=config.model_a.train,
                train_b=config.model_b.train,
                max_turns=config.game.max_turns,
                patch_size=config.game.patch_size,
                reward_fn_a=reward_fn_a,
                reward_fn_b=reward_fn_b,
                lethargy_policy=lethargy_policy,
                rng=rng,
            )

            # Track wins and lethargy losses
            winner = outcome.result.winner
            if outcome.lethargy_loser == "A":
                lethargy_a += 1
            elif outcome.lethargy_loser == "B":
                lethargy_b += 1
            elif winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1
            else:
                draws += 1

            logger.debug(
                "Game %d: winner=%s, turns=%d, lethargy=%s",
                total_games,
                winner or "draw",
                outcome.result.turns_played,
                outcome.lethargy_loser or "none",
            )

            # Policy gradient updates
            if config.model_a.train and optimizer_a is not None:
                loss_a = reinforce_update(
                    outcome.player_a.episode,
                    optimizer_a,
                    config.hyperparameters.gamma,
                    log_prob_tensors=outcome.player_a.log_prob_tensors,
                    entropy_tensors=outcome.player_a.entropy_tensors,
                    entropy_coeff=config.hyperparameters.entropy_coeff,
                    baseline=baseline_a,
                    reduction=config.hyperparameters.loss_reduction,
                    max_grad_norm=config.hyperparameters.max_grad_norm,
                )
                total_loss_a += loss_a
                logger.debug("Game %d: loss_a=%.6f", total_games, loss_a)

            if config.model_b.train and optimizer_b is not None:
                loss_b = reinforce_update(
                    outcome.player_b.episode,
                    optimizer_b,
                    config.hyperparameters.gamma,
                    log_prob_tensors=outcome.player_b.log_prob_tensors,
                    entropy_tensors=outcome.player_b.entropy_tensors,
                    entropy_coeff=config.hyperparameters.entropy_coeff,
                    baseline=baseline_b,
                    reduction=config.hyperparameters.loss_reduction,
                    max_grad_norm=config.hyperparameters.max_grad_norm,
                )
                total_loss_b += loss_b
                logger.debug("Game %d: loss_b=%.6f", total_games, loss_b)

            # Save sample game
            if total_games % config.output.sample_game_interval == 0:
                save_sample_game(
                    outcome.result,
                    config.output.sample_game_dir,
                    total_games,
                )
                logger.debug(
                    "Saved sample game %d to %s",
                    total_games,
                    config.output.sample_game_dir,
                )

            # Save weights periodically
            if total_games % config.output.save_weights_interval == 0:
                if config.model_a.train:
                    _save_weights(model_a, config.model_a.dir, total_games)
                    logger.info("Saved model A weights at game %d", total_games)
                if config.model_b.train:
                    _save_weights(model_b, config.model_b.dir, total_games)
                    logger.info("Saved model B weights at game %d", total_games)

            # Progress logging
            if total_games % config.game.games_per_map == 0:
                _log_progress(
                    total_games,
                    total_games_planned,
                    map_idx + 1,
                    wins_a,
                    wins_b,
                    draws,
                    lethargy_a,
                    lethargy_b,
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

    logger.info("Training complete. %d games played.", total_games)
    logger.info(
        "  Final record: A wins=%d, B wins=%d, draws=%d, lethargy_a=%d, lethargy_b=%d",
        wins_a,
        wins_b,
        draws,
        lethargy_a,
        lethargy_b,
    )


def _log_progress(
    total_games: int,
    planned: int,
    maps_done: int,
    wins_a: int,
    wins_b: int,
    draws: int,
    lethargy_a: int,
    lethargy_b: int,
    loss_a: float,
    loss_b: float,
    train_a: bool,
    train_b: bool,
) -> None:
    """Log a progress line at INFO level.

    Args:
        total_games: Games completed so far.
        planned: Total games planned.
        maps_done: Number of maps completed.
        wins_a: Cumulative wins for team A.
        wins_b: Cumulative wins for team B.
        draws: Cumulative draws.
        lethargy_a: Games lost by team A due to lethargy.
        lethargy_b: Games lost by team B due to lethargy.
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
    if lethargy_a > 0 or lethargy_b > 0:
        parts.append(f"leth_a={lethargy_a}")
        parts.append(f"leth_b={lethargy_b}")
    if train_a:
        parts.append(f"loss_a={avg_loss_a:.4f}")
    if train_b:
        parts.append(f"loss_b={avg_loss_b:.4f}")

    logger.info("  ".join(parts))

"""Main training loop orchestration.

Coordinates map generation, game execution, policy updates, and
periodic saving of weights and sample replays.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.optim import Optimizer

from hmls.core.map import GameMap
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


class TrainingSession:
    """Encapsulates the state and logic for a single REINFORCE training run.

    Holds models, optimizers, baselines, reward functions, lethargy policy,
    cumulative statistics, and the RNG.  The :meth:`train_one_game` method
    executes a single game on a given map, updates statistics, and performs
    the policy gradient update for trainable models.

    This class is the primary unit of work within :func:`train`, which
    handles map generation and delegates per-game logic here.

    Args:
        config: Training configuration.

    Raises:
        FileNotFoundError: If model directories lack required config files.
        ValueError: If model configurations are incompatible.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

        # Seed
        self.rng = random.Random(config.hyperparameters.seed)
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
        self.reward_fn_a: RewardFunction = RewardFunction(config.model_a.reward)
        self.reward_fn_b: RewardFunction = RewardFunction(config.model_b.reward)

        # Instantiate lethargy policy
        self.lethargy_policy: LethargyPolicy = _create_lethargy_policy(config.lethargy)

        # Load or create models
        self.model_a: TankModelBase = load_or_create_model(config.model_a.dir)
        self.model_b: TankModelBase = load_or_create_model(config.model_b.dir)

        params_a = sum(p.numel() for p in self.model_a.parameters())
        params_b = sum(p.numel() for p in self.model_b.parameters())
        logger.info("Model A trainable weights: %d", params_a)
        logger.info("Model B trainable weights: %d", params_b)

        # NOTE (REINFORCE_AUDIT item C): If future models use dropout or
        # batch normalisation, call model_a.train() / model_b.train() for
        # trainable models and model_a.eval() / model_b.eval() for frozen
        # ones here (and in game_runner.py where players are created).
        # Currently unnecessary because no model uses these layers.

        # Set up optimizers for models that are training
        self.optimizer_a: Optimizer | None = (
            torch.optim.Adam(self.model_a.parameters(), lr=config.hyperparameters.learning_rate)
            if config.model_a.train
            else None
        )
        self.optimizer_b: Optimizer | None = (
            torch.optim.Adam(self.model_b.parameters(), lr=config.hyperparameters.learning_rate)
            if config.model_b.train
            else None
        )

        # Cross-episode baselines (one per trainable player)
        self.baseline_a: ReturnBaseline | None = (
            ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
            if config.model_a.train
            else None
        )
        self.baseline_b: ReturnBaseline | None = (
            ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
            if config.model_b.train
            else None
        )

        # Training stats
        self.total_games: int = 0
        self.wins_a: int = 0
        self.wins_b: int = 0
        self.draws: int = 0
        self.lethargy_a: int = 0
        self.lethargy_b: int = 0
        self.total_loss_a: float = 0.0
        self.total_loss_b: float = 0.0

        self.total_games_planned: int = config.game.total_maps * config.game.games_per_map

    def train_one_game(self, game_map: GameMap) -> GameOutcome:
        """Run a single training game on the given map.

        Executes the game, updates cumulative statistics (wins, draws,
        lethargy losses), and performs REINFORCE policy gradient updates
        for any trainable models.

        Args:
            game_map: The map to play on.

        Returns:
            The outcome of the game.
        """
        self.total_games += 1

        outcome: GameOutcome = run_game(
            game_map,
            self.model_a,
            self.model_b,
            train_a=self.config.model_a.train,
            train_b=self.config.model_b.train,
            max_turns=self.config.game.max_turns,
            patch_size=self.config.game.patch_size,
            reward_fn_a=self.reward_fn_a,
            reward_fn_b=self.reward_fn_b,
            lethargy_policy=self.lethargy_policy,
            rng=self.rng,
        )

        self._update_stats(outcome)
        self._policy_update(outcome)

        return outcome

    def _update_stats(self, outcome: GameOutcome) -> None:
        """Update win/draw/lethargy counters from a game outcome."""
        winner = outcome.result.winner
        if outcome.lethargy_loser == "A":
            self.lethargy_a += 1
        elif outcome.lethargy_loser == "B":
            self.lethargy_b += 1
        elif winner == "A":
            self.wins_a += 1
        elif winner == "B":
            self.wins_b += 1
        else:
            self.draws += 1

        logger.debug(
            "Game %d: winner=%s, turns=%d, lethargy=%s",
            self.total_games,
            winner or "draw",
            outcome.result.turns_played,
            outcome.lethargy_loser or "none",
        )

    def _policy_update(self, outcome: GameOutcome) -> None:
        """Perform REINFORCE policy gradient updates for trainable models."""
        if self.config.model_a.train and self.optimizer_a is not None:
            loss_a = reinforce_update(
                outcome.player_a.episode,
                self.optimizer_a,
                self.config.hyperparameters.gamma,
                log_prob_tensors=outcome.player_a.log_prob_tensors,
                entropy_tensors=outcome.player_a.entropy_tensors,
                entropy_coeff=self.config.hyperparameters.entropy_coeff,
                baseline=self.baseline_a,
                reduction=self.config.hyperparameters.loss_reduction,
                max_grad_norm=self.config.hyperparameters.max_grad_norm,
            )
            self.total_loss_a += loss_a
            logger.debug("Game %d: loss_a=%.6f", self.total_games, loss_a)

        if self.config.model_b.train and self.optimizer_b is not None:
            loss_b = reinforce_update(
                outcome.player_b.episode,
                self.optimizer_b,
                self.config.hyperparameters.gamma,
                log_prob_tensors=outcome.player_b.log_prob_tensors,
                entropy_tensors=outcome.player_b.entropy_tensors,
                entropy_coeff=self.config.hyperparameters.entropy_coeff,
                baseline=self.baseline_b,
                reduction=self.config.hyperparameters.loss_reduction,
                max_grad_norm=self.config.hyperparameters.max_grad_norm,
            )
            self.total_loss_b += loss_b
            logger.debug("Game %d: loss_b=%.6f", self.total_games, loss_b)

    def save_sample_if_due(self, outcome: GameOutcome) -> None:
        """Save a sample game replay if the current game is at the configured interval.

        Args:
            outcome: The game outcome to potentially save.
        """
        if self.total_games % self.config.output.sample_game_interval == 0:
            save_sample_game(
                outcome.result,
                self.config.output.sample_game_dir,
                self.total_games,
            )
            logger.debug(
                "Saved sample game %d to %s",
                self.total_games,
                self.config.output.sample_game_dir,
            )

    def save_weights_if_due(self) -> None:
        """Save model weights if the current game is at the configured interval."""
        if self.total_games % self.config.output.save_weights_interval == 0:
            if self.config.model_a.train:
                _save_weights(self.model_a, self.config.model_a.dir, self.total_games)
                logger.info("Saved model A weights at game %d", self.total_games)
            if self.config.model_b.train:
                _save_weights(self.model_b, self.config.model_b.dir, self.total_games)
                logger.info("Saved model B weights at game %d", self.total_games)

    def log_progress_if_due(self, map_idx: int) -> None:
        """Log a progress summary if a full map's worth of games has been played.

        Args:
            map_idx: Zero-based index of the current map.
        """
        if self.total_games % self.config.game.games_per_map == 0:
            _log_progress(
                self.total_games,
                self.total_games_planned,
                map_idx + 1,
                self.wins_a,
                self.wins_b,
                self.draws,
                self.lethargy_a,
                self.lethargy_b,
                self.total_loss_a,
                self.total_loss_b,
                self.config.model_a.train,
                self.config.model_b.train,
            )

    def save_final_weights(self) -> None:
        """Save model weights at the end of training for all trainable models."""
        if self.config.model_a.train:
            _save_weights(self.model_a, self.config.model_a.dir, self.total_games)
        if self.config.model_b.train:
            _save_weights(self.model_b, self.config.model_b.dir, self.total_games)

    def log_training_start(self) -> None:
        """Log a summary of the training configuration at INFO level."""
        logger.info(
            "Starting training: %d maps × %d games/map = %d total games",
            self.config.game.total_maps,
            self.config.game.games_per_map,
            self.total_games_planned,
        )
        logger.info(
            "  Train A: %s, Train B: %s",
            self.config.model_a.train,
            self.config.model_b.train,
        )
        logger.info(
            "  Map size: %d–%d (random), impassable: %.0f%%",
            self.config.map.min_size,
            self.config.map.max_size,
            self.config.map.impassable_fraction * 100,
        )
        logger.info(
            "  Max turns/game: %d, γ=%s, lr=%s",
            self.config.game.max_turns,
            self.config.hyperparameters.gamma,
            self.config.hyperparameters.learning_rate,
        )

    def log_training_complete(self) -> None:
        """Log a summary of training results at INFO level."""
        logger.info("Training complete. %d games played.", self.total_games)
        logger.info(
            "  Final record: A wins=%d, B wins=%d, draws=%d, lethargy_a=%d, lethargy_b=%d",
            self.wins_a,
            self.wins_b,
            self.draws,
            self.lethargy_a,
            self.lethargy_b,
        )


def _generate_map(
    config: TrainerConfig,
    rng: random.Random,
    map_idx: int,
) -> GameMap:
    """Generate a random map for the training loop.

    Args:
        config: Training configuration (map size/strategy settings).
        rng: Random number generator for map dimensions and seed.
        map_idx: Zero-based map index (used for logging).

    Returns:
        A newly generated GameMap.
    """
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
    return game_map


def train(config: TrainerConfig) -> None:
    """Run the full training loop.

    Instantiates a :class:`TrainingSession`, generates maps, and
    delegates per-game logic to the session object.

    Args:
        config: Training configuration.

    Raises:
        FileNotFoundError: If model directories lack required config files.
        ValueError: If model configurations are incompatible.
    """
    session = TrainingSession(config)
    session.log_training_start()

    for map_idx in range(config.game.total_maps):
        game_map = _generate_map(config, session.rng, map_idx)
        for _ in range(config.game.games_per_map):
            outcome = session.train_one_game(game_map)
            session.save_sample_if_due(outcome)
            session.save_weights_if_due()
            session.log_progress_if_due(map_idx)

    session.save_final_weights()
    session.log_training_complete()


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

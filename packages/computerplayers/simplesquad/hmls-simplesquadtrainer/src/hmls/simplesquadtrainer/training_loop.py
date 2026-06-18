"""Main training loop for the simple squad trainer.

Coordinates map generation, game execution, and policy updates
for both the planner and executor models.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.optim import Optimizer

from hmls.core.map import GameMap
from hmls.reinforcetrainer.updater import ReturnBaseline
from hmls.simplesquadexecutor.persistence import save_executor
from hmls.simplesquadplanner.persistence import save_planner
from hmls.simplesquadplayer.player import SimpleSquadPlayer
from hmls.simplesquadtrainer.config import (
    SquadTeamRef,
    SquadTrainerConfig,
)
from hmls.simplesquadtrainer.game_runner import (
    GameOutcome,
    TeamState,
    create_map,
    create_team_state,
    run_game,
)
from hmls.simplesquadtrainer.updater import executor_update, planner_update

logger = logging.getLogger(__name__)


class SquadTrainingSession:
    """Encapsulates the state for a squad REINFORCE training run.

    Holds models, optimizers, baselines, and statistics.  The
    :meth:`train_one_game` method executes a single game and performs
    policy gradient updates.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: SquadTrainerConfig) -> None:
        self.config = config

        # Seed
        self.rng = random.Random(config.hyperparameters.seed)
        if config.hyperparameters.seed is not None:
            torch.manual_seed(config.hyperparameters.seed)

        # We defer team state creation to per-game (needs map dimensions)
        # but set up optimizers once models are loaded
        self._team_a_state: TeamState | None = None
        self._team_b_state: TeamState | None = None

        # Optimizers (created on first game when models are loaded)
        self._executor_optimizer_a: Optimizer | None = None
        self._planner_optimizer_a: Optimizer | None = None
        self._executor_optimizer_b: Optimizer | None = None
        self._planner_optimizer_b: Optimizer | None = None

        # Baselines
        self._executor_baseline_a = ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
        self._planner_baseline_a = ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
        self._executor_baseline_b = ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)
        self._planner_baseline_b = ReturnBaseline(alpha=config.hyperparameters.baseline_alpha)

        # Stats
        self.total_games = 0
        self.wins_a = 0
        self.wins_b = 0
        self.draws = 0
        self.total_games_planned = config.game.total_maps * config.game.games_per_map

    def _ensure_teams(self, map_width: int, map_height: int) -> tuple[TeamState, TeamState]:
        """Create or return team states (lazy init on first map)."""
        if self._team_a_state is None:
            self._team_a_state = create_team_state(
                self.config.team_a,
                "A",
                self.config.game.tanks_per_team,
                self.config.game.patch_size,
                map_width,
                map_height,
            )
            # Create optimizers for squad teams
            if isinstance(self.config.team_a, SquadTeamRef) and self.config.team_a.train:
                player = self._team_a_state.player
                assert isinstance(player, SimpleSquadPlayer)
                self._executor_optimizer_a = torch.optim.Adam(
                    player.executor.parameters(),
                    lr=self.config.hyperparameters.executor_learning_rate,
                )
                self._planner_optimizer_a = torch.optim.Adam(
                    player.planner.parameters(),
                    lr=self.config.hyperparameters.planner_learning_rate,
                )

        if self._team_b_state is None:
            self._team_b_state = create_team_state(
                self.config.team_b,
                "B",
                self.config.game.tanks_per_team,
                self.config.game.patch_size,
                map_width,
                map_height,
            )
            if isinstance(self.config.team_b, SquadTeamRef) and self.config.team_b.train:
                player = self._team_b_state.player
                assert isinstance(player, SimpleSquadPlayer)
                self._executor_optimizer_b = torch.optim.Adam(
                    player.executor.parameters(),
                    lr=self.config.hyperparameters.executor_learning_rate,
                )
                self._planner_optimizer_b = torch.optim.Adam(
                    player.planner.parameters(),
                    lr=self.config.hyperparameters.planner_learning_rate,
                )

        return self._team_a_state, self._team_b_state

    def train_one_game(self, game_map: GameMap) -> GameOutcome:
        """Run one training game and perform policy updates.

        Args:
            game_map: The map to play on.

        Returns:
            The game outcome.
        """
        self.total_games += 1
        team_a, team_b = self._ensure_teams(game_map.width, game_map.height)

        outcome = run_game(
            game_map,
            team_a,
            team_b,
            max_turns=self.config.game.max_turns,
            patch_size=self.config.game.patch_size,
            tanks_per_team=self.config.game.tanks_per_team,
            rng=self.rng,
        )

        self._update_stats(outcome)
        self._policy_update(outcome)

        return outcome

    def _update_stats(self, outcome: GameOutcome) -> None:
        """Update win/draw counters."""
        winner = outcome.result.winner
        if winner == "A":
            self.wins_a += 1
        elif winner == "B":
            self.wins_b += 1
        else:
            self.draws += 1

    def _policy_update(self, outcome: GameOutcome) -> None:
        """Perform REINFORCE updates for trainable teams."""
        hp = self.config.hyperparameters

        # Team A
        if isinstance(self.config.team_a, SquadTeamRef) and self.config.team_a.train:
            player = outcome.team_a.player
            assert isinstance(player, SimpleSquadPlayer)
            if self._executor_optimizer_a is not None:
                executor_update(
                    episodes=player.episodes,
                    log_prob_tensors=player.log_prob_tensors,
                    entropy_tensors=player.entropy_tensors,
                    optimizer=self._executor_optimizer_a,
                    gamma=hp.gamma,
                    baseline=self._executor_baseline_a,
                    entropy_coeff=hp.entropy_coeff,
                    reduction=hp.loss_reduction,
                    max_grad_norm=hp.max_grad_norm,
                )
            if self._planner_optimizer_a is not None:
                planner_update(
                    episode=player.planner_episode,
                    log_prob_tensors=player.planner_log_prob_tensors,
                    entropy_tensors=player.planner_entropy_tensors,
                    optimizer=self._planner_optimizer_a,
                    gamma=hp.gamma,
                    baseline=self._planner_baseline_a,
                    entropy_coeff=hp.planner_entropy_coeff,
                    reduction=hp.loss_reduction,
                    max_grad_norm=hp.max_grad_norm,
                )

        # Team B
        if isinstance(self.config.team_b, SquadTeamRef) and self.config.team_b.train:
            player = outcome.team_b.player
            assert isinstance(player, SimpleSquadPlayer)
            if self._executor_optimizer_b is not None:
                executor_update(
                    episodes=player.episodes,
                    log_prob_tensors=player.log_prob_tensors,
                    entropy_tensors=player.entropy_tensors,
                    optimizer=self._executor_optimizer_b,
                    gamma=hp.gamma,
                    baseline=self._executor_baseline_b,
                    entropy_coeff=hp.entropy_coeff,
                    reduction=hp.loss_reduction,
                    max_grad_norm=hp.max_grad_norm,
                )
            if self._planner_optimizer_b is not None:
                planner_update(
                    episode=player.planner_episode,
                    log_prob_tensors=player.planner_log_prob_tensors,
                    entropy_tensors=player.planner_entropy_tensors,
                    optimizer=self._planner_optimizer_b,
                    gamma=hp.gamma,
                    baseline=self._planner_baseline_b,
                    entropy_coeff=hp.planner_entropy_coeff,
                    reduction=hp.loss_reduction,
                    max_grad_norm=hp.max_grad_norm,
                )

    def save_weights(self) -> None:
        """Save model weights for all trainable squad teams."""
        if isinstance(self.config.team_a, SquadTeamRef) and self.config.team_a.train:
            player = self._team_a_state.player if self._team_a_state else None
            if isinstance(player, SimpleSquadPlayer):
                save_planner(
                    player.planner,
                    self.config.team_a.dir / "planner",
                    metadata={"games_played": self.total_games},
                )
                save_executor(
                    player.executor,
                    self.config.team_a.dir / "executor",
                    metadata={"games_played": self.total_games},
                )
        if isinstance(self.config.team_b, SquadTeamRef) and self.config.team_b.train:
            player = self._team_b_state.player if self._team_b_state else None
            if isinstance(player, SimpleSquadPlayer):
                save_planner(
                    player.planner,
                    self.config.team_b.dir / "planner",
                    metadata={"games_played": self.total_games},
                )
                save_executor(
                    player.executor,
                    self.config.team_b.dir / "executor",
                    metadata={"games_played": self.total_games},
                )

    def log_progress(self, map_idx: int) -> None:
        """Log training progress."""
        pct = (
            100.0 * self.total_games / self.total_games_planned
            if self.total_games_planned > 0
            else 0.0
        )
        logger.info(
            "[%5.1f%%] games=%d/%d maps=%d A=%d B=%d draws=%d",
            pct,
            self.total_games,
            self.total_games_planned,
            map_idx + 1,
            self.wins_a,
            self.wins_b,
            self.draws,
        )


def train(config: SquadTrainerConfig) -> None:
    """Run the full squad training loop.

    Args:
        config: Training configuration.
    """
    session = SquadTrainingSession(config)
    logger.info(
        "Starting squad training: %d maps × %d games/map = %d total, %d tanks/team",
        config.game.total_maps,
        config.game.games_per_map,
        session.total_games_planned,
        config.game.tanks_per_team,
    )

    for map_idx in range(config.game.total_maps):
        strategy_config = config.map.strategies[map_idx % len(config.map.strategies)]
        map_seed = session.rng.randint(0, 2**31)
        map_width = session.rng.randint(config.map.min_size, config.map.max_size)
        map_height = session.rng.randint(config.map.min_size, config.map.max_size)
        game_map = create_map(
            map_width,
            map_height,
            config.map.impassable_fraction,
            strategy_config,
            seed=map_seed,
        )

        for _ in range(config.game.games_per_map):
            session.train_one_game(game_map)

            if session.total_games % config.output.save_weights_interval == 0:
                session.save_weights()

        session.log_progress(map_idx)

    session.save_weights()
    logger.info(
        "Training complete. %d games: A=%d B=%d draws=%d",
        session.total_games,
        session.wins_a,
        session.wins_b,
        session.draws,
    )

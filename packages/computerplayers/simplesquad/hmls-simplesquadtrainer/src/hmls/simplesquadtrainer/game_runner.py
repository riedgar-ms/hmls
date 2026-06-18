"""Game runner: sets up and plays a single squad training game.

Handles both squad and independent team types, multi-tank placement,
per-tank reward assignment, and planner reward aggregation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from hmls.core.engine import GameEngine
from hmls.core.map import GameMap
from hmls.core.placement import place_tanks
from hmls.core.player import Player
from hmls.core.results import GameResult, HistoryEntry
from hmls.mapgenerator import StrategyConfigBase, generate_map_from_config
from hmls.nncore.persistence import create_player as create_singletank_player
from hmls.nncore.persistence import load_or_create_model
from hmls.nncore.reward import RewardFunction
from hmls.simplesquadexecutor.persistence import load_or_create_executor
from hmls.simplesquadplanner.persistence import load_or_create_planner
from hmls.simplesquadplayer.player import SimpleSquadPlayer
from hmls.simplesquadtrainer.config import SquadTeamRef, TeamRef


@dataclass
class TeamState:
    """Runtime state for a team during training.

    Attributes:
        player: The player instance for this team.
        reward_fns: Per-tank reward functions (keyed by tank_id).
        is_squad: Whether this team uses the squad architecture.
        train: Whether this team is being trained.
    """

    player: Player
    reward_fns: dict[str, RewardFunction] = field(default_factory=dict)
    is_squad: bool = False
    train: bool = True


@dataclass
class GameOutcome:
    """Result of a single squad training game.

    Attributes:
        result: The full GameResult.
        team_a: Runtime state for team A.
        team_b: Runtime state for team B.
    """

    result: GameResult
    team_a: TeamState
    team_b: TeamState


def create_team_state(
    team_ref: TeamRef,
    team_name: str,
    tanks_per_team: int,
    patch_size: int,
    map_width: int,
    map_height: int,
) -> TeamState:
    """Create the runtime state for a team.

    Args:
        team_ref: Team configuration reference.
        team_name: Team identifier (e.g. ``"A"`` or ``"B"``).
        tanks_per_team: Number of tanks.
        patch_size: Visibility patch size.
        map_width: Map width for position normalisation.
        map_height: Map height for position normalisation.

    Returns:
        Initialised :class:`TeamState`.
    """
    if isinstance(team_ref, SquadTeamRef):
        planner = load_or_create_planner(team_ref.dir / "planner")
        executor = load_or_create_executor(team_ref.dir / "executor")
        mode: Literal["play", "learn"] = "learn" if team_ref.train else "play"
        player: Player = SimpleSquadPlayer(
            team=team_name,
            planner=planner,
            executor=executor,
            mode=mode,
            map_width=map_width,
            map_height=map_height,
        )
        reward_fns = {
            f"{team_name}{i + 1}": RewardFunction(team_ref.reward) for i in range(tanks_per_team)
        }
        return TeamState(player=player, reward_fns=reward_fns, is_squad=True, train=team_ref.train)
    else:
        # Independent: N copies of a single-tank model, each acting independently
        # We use a single NNPlayer per team (it handles one tank at a time)
        model = load_or_create_model(team_ref.dir)
        ind_mode: Literal["play", "learn"] = "learn" if team_ref.train else "play"
        player = create_singletank_player(
            model_id=model.config.model_id,
            team=team_name,
            model=model,
            mode=ind_mode,
        )
        reward_fns = {
            f"{team_name}{i + 1}": RewardFunction(team_ref.reward) for i in range(tanks_per_team)
        }
        return TeamState(player=player, reward_fns=reward_fns, is_squad=False, train=team_ref.train)


def create_map(
    width: int,
    height: int,
    impassable_fraction: float,
    strategy_config: StrategyConfigBase,
    seed: int | None = None,
) -> GameMap:
    """Generate a random map.

    Args:
        width: Map width.
        height: Map height.
        impassable_fraction: Fraction of impassable cells.
        strategy_config: Generation strategy configuration.
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


def _reset_teams(team_a: TeamState, team_b: TeamState) -> None:
    """Reset episode and reward state for both teams."""
    if isinstance(team_a.player, SimpleSquadPlayer):
        team_a.player.reset_episode()
    if isinstance(team_b.player, SimpleSquadPlayer):
        team_b.player.reset_episode()
    for rf in team_a.reward_fns.values():
        rf.reset()
    for rf in team_b.reward_fns.values():
        rf.reset()


def _signal_begin_round(team_a: TeamState, team_b: TeamState) -> None:
    """Signal the start of a new planning round to squad players."""
    if isinstance(team_a.player, SimpleSquadPlayer):
        team_a.player.begin_round()
    if isinstance(team_b.player, SimpleSquadPlayer):
        team_b.player.begin_round()


def run_game(
    game_map: GameMap,
    team_a: TeamState,
    team_b: TeamState,
    *,
    max_turns: int = 200,
    patch_size: int = 9,
    tanks_per_team: int = 3,
    rng: random.Random | None = None,
) -> GameOutcome:
    """Run a single squad training game.

    Args:
        game_map: The map to play on.
        team_a: Team A state.
        team_b: Team B state.
        max_turns: Maximum turns before draw.
        patch_size: Visibility patch size.
        tanks_per_team: Tanks per team.
        rng: Random number generator.

    Returns:
        A :class:`GameOutcome` with results and player references.
    """
    _reset_teams(team_a, team_b)

    # Place tanks
    placement_seed = rng.randint(0, 2**31) if rng else None
    tanks = place_tanks(game_map, tanks_per_player=tanks_per_team, seed=placement_seed)

    engine = GameEngine(
        game_map=game_map,
        tanks=tanks,
        players={"A": team_a.player, "B": team_b.player},
        max_turns=max_turns,
        patch_size=patch_size,
    )

    # Track planning rounds: signal begin_round at start of each turn
    last_turn_number = -1

    while not engine.game_over:
        current_turn = engine.turn_number if hasattr(engine, "turn_number") else 0
        if current_turn != last_turn_number:
            _signal_begin_round(team_a, team_b)
            last_turn_number = current_turn

        entry: HistoryEntry = engine.step()

        # Assign rewards
        acting_team = entry.tank_id[0]
        tank_id = entry.tank_id
        if acting_team == "A" and team_a.train:
            _assign_step_reward(team_a, tank_id, entry)
        elif acting_team == "B" and team_b.train:
            _assign_step_reward(team_b, tank_id, entry)

    result = engine.make_result()

    # End-of-episode rewards
    _apply_end_rewards(result.winner, team_a, team_b)

    # Compute planner rewards (mean of per-tank executor rewards)
    if team_a.is_squad and team_a.train:
        _assign_planner_rewards(team_a)
    if team_b.is_squad and team_b.train:
        _assign_planner_rewards(team_b)

    return GameOutcome(result=result, team_a=team_a, team_b=team_b)


def _assign_step_reward(team: TeamState, tank_id: str, entry: HistoryEntry) -> None:
    """Compute and assign step reward for a specific tank."""
    reward_fn = team.reward_fns.get(tank_id)
    if reward_fn is None:
        return

    if isinstance(team.player, SimpleSquadPlayer):
        patch = team.player.last_patch(tank_id)
        if patch is None:
            return
        reward_fn.observe_patch(patch)
        reward = reward_fn.compute_step_reward(entry, patch, team.player.team)
        episode = team.player.episodes.get(tank_id)
        if episode and len(episode) > 0:
            episode.set_reward(len(episode) - 1, reward)


def _apply_end_rewards(
    winner: str | None,
    team_a: TeamState,
    team_b: TeamState,
) -> None:
    """Apply end-of-episode rewards to both teams."""
    for team_state, team_name in [(team_a, "A"), (team_b, "B")]:
        if not team_state.train:
            continue
        won = True if winner == team_name else (False if winner is not None else None)
        if isinstance(team_state.player, SimpleSquadPlayer):
            for tank_id, episode in team_state.player.episodes.items():
                if len(episode) == 0:
                    continue
                reward_fn = team_state.reward_fns.get(tank_id)
                if reward_fn:
                    end_reward = reward_fn.compute_episode_end_reward(won)
                    last_idx = len(episode) - 1
                    current = episode.steps[last_idx].reward
                    episode.set_reward(last_idx, current + end_reward)


def _assign_planner_rewards(team: TeamState) -> None:
    """Assign aggregated rewards to the planner's trajectory.

    The planner's reward at each step is the mean of per-tank executor
    rewards across all alive tanks at that timestep.
    """
    if not isinstance(team.player, SimpleSquadPlayer):
        return

    planner_episode = team.player.planner_episode
    if len(planner_episode) == 0:
        return

    # Aggregate per-tank rewards into per-planning-round rewards
    # Simple approach: mean of all per-tank rewards / planner steps
    all_rewards: list[float] = []
    for episode in team.player.episodes.values():
        all_rewards.extend(episode.rewards())

    if not all_rewards:
        return

    total_reward = sum(all_rewards)
    num_planner_steps = len(planner_episode)

    # Distribute evenly across planner steps
    per_step_reward = total_reward / num_planner_steps
    for i in range(num_planner_steps):
        planner_episode.set_reward(i, per_step_reward)

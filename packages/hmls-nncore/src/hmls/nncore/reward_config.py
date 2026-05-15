"""Pydantic configuration models for reward shaping.

Defines :class:`RewardConfig` and its nested section models that
parameterise the shaped reward function used during RL training.
Extracted into its own module so that both ``reward`` and
``reward_components`` can import it without circular dependencies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Nested reward config sections ────────────────────────────────────


class ActionsRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Per-action reward configuration.

    Attributes:
        move_forward: Reward for choosing to move forward.
        turn_left: Reward for choosing to turn left.
        turn_right: Reward for choosing to turn right.
        fire: Reward for choosing to fire (independent of hit/miss,
            which are in :class:`FiringRewardConfig`).
        pass_action: Reward for deliberately choosing to pass a turn
            (not applied when an invalid action is converted to pass).
        consecutive_turn: Escalating reward multiplier for consecutive
            turn actions (typically negative).  When a tank takes N
            consecutive turns (``TURN_LEFT`` or ``TURN_RIGHT``), the
            Nth turn incurs an additional reward of
            ``consecutive_turn × N``.

            The streak resets to 0 only on *meaningful* non-turn
            actions: a fire that hits or a valid move forward.  Other
            actions leave the streak unchanged but do not incur the
            escalating penalty.

            Set to ``0.0`` (the default) to disable.
        consecutive_pass: Escalating reward multiplier for consecutive
            valid pass actions (typically negative).  When a tank takes
            N consecutive deliberate passes, the Nth pass incurs an
            additional reward of ``consecutive_pass × N``.

            The streak resets to 0 on a fire that hits or a valid move
            forward.  Other actions leave the streak unchanged but do
            not incur the escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    move_forward: float = Field(
        default=0.0,
        title="Move Forward Reward",
        description="Reward for choosing to move forward.",
    )
    turn_left: float = Field(
        default=0.0,
        title="Turn Left Reward",
        description="Reward for choosing to turn left.",
    )
    turn_right: float = Field(
        default=0.0,
        title="Turn Right Reward",
        description="Reward for choosing to turn right.",
    )
    fire: float = Field(
        default=0.0,
        title="Fire Reward",
        description=("Reward for choosing to fire, independent of hit/miss outcome."),
    )
    pass_action: float = Field(
        default=-0.02,
        title="Pass Action Reward",
        description=(
            "Reward for deliberately choosing to pass a turn. "
            "Not applied when an invalid action is converted to pass."
        ),
    )
    consecutive_turn: float = Field(
        default=0.0,
        title="Consecutive Turn Penalty Multiplier",
        description=(
            "Escalating reward multiplier for consecutive turn actions. "
            "The Nth consecutive turn incurs an additional reward of this value × N. "
            "Streak resets on a hit or valid move forward. Set to 0.0 to disable."
        ),
    )
    consecutive_pass: float = Field(
        default=0.0,
        title="Consecutive Pass Penalty Multiplier",
        description=(
            "Escalating reward multiplier for consecutive deliberate passes. "
            "The Nth consecutive pass incurs an additional reward of this value × N. "
            "Streak resets on a hit or valid move forward. Set to 0.0 to disable."
        ),
    )


class FiringRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Firing-outcome reward configuration.

    Attributes:
        hit: Reward for hitting an enemy tank.
        miss: Reward (negative) for firing and missing.
        neglect: Reward (negative) for not firing when an alive enemy
            tank is directly ahead and could have been hit.
        consecutive_miss: Escalating reward multiplier for consecutive
            fire misses (typically negative).  When a tank fires and
            misses N consecutive times, the Nth miss incurs an
            additional reward of ``consecutive_miss × N``.

            The streak resets to 0 on a fire that hits or a valid move
            forward.  Other actions leave the streak unchanged but do
            not incur the escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    hit: float = Field(
        default=0.5,
        title="Hit Reward",
        description="Reward for hitting an enemy tank.",
    )
    miss: float = Field(
        default=-0.05,
        title="Miss Penalty",
        description="Penalty for firing and missing.",
    )
    neglect: float = Field(
        default=-0.1,
        title="Neglect Penalty",
        description=(
            "Penalty for not firing when an alive enemy tank is directly "
            "ahead and could have been hit."
        ),
    )
    consecutive_miss: float = Field(
        default=0.0,
        title="Consecutive Miss Penalty Multiplier",
        description=(
            "Escalating penalty multiplier for consecutive fire misses. "
            "The Nth consecutive miss incurs an additional reward of this value × N. "
            "Streak resets on a hit or valid move forward. Set to 0.0 to disable."
        ),
    )


class GameStateRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Game-state reward configuration.

    Attributes:
        win: Reward for winning the game.
        loss: Reward (negative) for losing the game.
        invalid_move: Reward (negative) for attempting an invalid action.
        step: Per-step reward (negative to encourage faster play).
        death: Reward (negative) when the player's tank dies.
    """

    win: float = Field(
        default=1.0,
        title="Win Reward",
        description="Reward for winning the game.",
    )
    loss: float = Field(
        default=-1.0,
        title="Loss Penalty",
        description="Penalty for losing the game.",
    )
    invalid_move: float = Field(
        default=-0.1,
        title="Invalid Move Penalty",
        description="Penalty for attempting an invalid action.",
    )
    step: float = Field(
        default=-0.01,
        title="Per-Step Cost",
        description="Per-step reward (typically negative to encourage faster play).",
    )
    death: float = Field(
        default=-1.0,
        title="Death Penalty",
        description="Penalty when the player's tank dies.",
    )


class ExplorationRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Exploration reward configuration.

    Attributes:
        see_cell: Reward per newly *seen* cell in the visibility patch
            (cells visible but not necessarily moved into).
        occupy_cell: Reward per newly *occupied* cell (cells the tank
            physically moves into for the first time).
    """

    see_cell: float = Field(
        default=0.02,
        title="See Cell Reward",
        description=(
            "Reward per newly seen cell in the visibility patch "
            "(visible but not necessarily moved into)."
        ),
    )
    occupy_cell: float = Field(
        default=0.0,
        title="Occupy Cell Reward",
        description=(
            "Reward per newly occupied cell (cells the tank physically "
            "moves into for the first time)."
        ),
    )


class SituationalRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Situational reward configuration.

    Attributes:
        enemy_in_cone: Per-enemy reward for each alive enemy tank
            visible in the forward cone of the egocentric patch.
        enemy_in_cone_distance_discount: Discount factor applied
            per unit of Manhattan distance from the player to the
            enemy in egocentric coordinates.  Each enemy's
            contribution is ``enemy_in_cone *
            enemy_in_cone_distance_discount ** manhattan_distance``.
            A value of ``1.0`` (the default) disables discounting.
            Values below ``1.0`` make distant enemies worth less.
    """

    enemy_in_cone: float = Field(
        default=0.01,
        title="Enemy In Cone Reward",
        description=(
            "Per-enemy reward for each alive enemy tank visible in the "
            "forward cone of the egocentric patch."
        ),
    )
    enemy_in_cone_distance_discount: float = Field(
        default=1.0,
        title="Enemy In Cone Distance Discount",
        description=(
            "Discount factor per unit of Manhattan distance to the enemy. "
            "Each enemy's contribution is enemy_in_cone × discount^distance. "
            "1.0 disables discounting; values below 1.0 reduce distant enemy reward."
        ),
    )


class RewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Top-level reward configuration with nested sections.

    All sections have sensible defaults so ``RewardConfig()`` produces
    a usable configuration out of the box.

    Attributes:
        actions: Per-action rewards (move, turn, fire, pass,
            consecutive turn penalty).
        firing: Firing-outcome rewards (hit, miss, neglect).
        game_state: Game-state rewards (win, loss, invalid move,
            per-step cost, death).
        exploration: Exploration rewards (see cell, occupy cell).
        situational: Situational rewards (enemy in cone).
    """

    actions: ActionsRewardConfig = Field(
        default_factory=ActionsRewardConfig,
        title="Action Rewards",
        description="Per-action rewards (move, turn, fire, pass, consecutive penalties).",
    )
    firing: FiringRewardConfig = Field(
        default_factory=FiringRewardConfig,
        title="Firing Rewards",
        description="Firing-outcome rewards (hit, miss, neglect, consecutive miss).",
    )
    game_state: GameStateRewardConfig = Field(
        default_factory=GameStateRewardConfig,
        title="Game State Rewards",
        description="Game-state rewards (win, loss, invalid move, per-step cost, death).",
    )
    exploration: ExplorationRewardConfig = Field(
        default_factory=ExplorationRewardConfig,
        title="Exploration Rewards",
        description="Exploration rewards (see cell, occupy cell).",
    )
    situational: SituationalRewardConfig = Field(
        default_factory=SituationalRewardConfig,
        title="Situational Rewards",
        description="Situational rewards (enemy in cone).",
    )

"""Tests for the concrete NNPlayer in hmls.nncore.player."""

from __future__ import annotations

import pytest
import torch

from hmls.core.map import CellType
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, PlayerView, TankInfo, TankPatch, VisibleCell
from hmls.nncore.constants import ACTION_INDEX_TO_ACTION
from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.player import NNPlayer

# ── Minimal stub model for testing ────────────────────────────────────


class _StubConfig(TankModelConfig, frozen=True, extra="forbid"):
    """Minimal config for the stub model."""

    model_package: str = "test.stub"


class _StubModel(TankModelBase):
    """Minimal TankModelBase that returns fixed logits."""

    def __init__(self, config: _StubConfig | None = None) -> None:
        super().__init__()
        self.config: _StubConfig = config or _StubConfig()
        self._hidden_size = 4

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed logits favouring action index 0."""
        unbatched = patch_tensor.dim() == 3
        if unbatched:
            logits = torch.tensor([2.0, 0.1, 0.1, 0.1, 0.1])
            new_hidden = hidden + 0.1
        else:
            batch = patch_tensor.size(0)
            logits = torch.tensor([2.0, 0.1, 0.1, 0.1, 0.1]).unsqueeze(0).expand(batch, -1)
            new_hidden = hidden + 0.1
        return logits, new_hidden

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return zero hidden state."""
        return torch.zeros(batch_size, self._hidden_size)

    @property
    def total_hidden_size(self) -> int:
        """Hidden size."""
        return self._hidden_size


# ── Helpers ───────────────────────────────────────────────────────────


def _make_view(patch_size: int = 9, team: str = "alpha") -> PlayerView:
    """Create a minimal PlayerView with a single all-passable visible patch."""
    grid: list[list[VisibleCell | FogCell]] = []
    for _row in range(patch_size):
        row_cells: list[VisibleCell | FogCell] = []
        for _col in range(patch_size):
            row_cells.append(VisibleCell(cell_type=CellType.PASSABLE))
        grid.append(row_cells)

    patch = TankPatch(
        tank_id="t1",
        position=Position(5, 5),
        direction=Direction.NORTH,
        grid=grid,
    )
    tank_info = TankInfo(
        tank_id="t1", position=Position(5, 5), direction=Direction.NORTH, alive=True
    )
    return PlayerView(patches=[patch], tanks=[tank_info])


# ── Tests ─────────────────────────────────────────────────────────────


def test_player_choose_action_play_mode() -> None:
    """Player returns a valid action in play mode."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="play")

    view = _make_view(patch_size=9)
    action = player.choose_action("t1", view)
    assert action in ACTION_INDEX_TO_ACTION


def test_player_choose_action_learn_mode_records_trajectory() -> None:
    """In learn mode, trajectory steps are recorded."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="learn")

    view = _make_view(patch_size=9)
    player.choose_action("t1", view)
    assert len(player.episode) == 1
    step = player.episode.steps[0]
    assert 0 <= step.action_index < 5
    assert step.log_prob < 0  # log-probability is negative


def test_player_patch_size_mismatch_raises() -> None:
    """Player raises ValueError if patch size doesn't match."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="play")

    view = _make_view(patch_size=7)
    with pytest.raises(ValueError, match="Patch size mismatch"):
        player.choose_action("t1", view)


def test_player_reset_episode() -> None:
    """reset_episode clears trajectory and resets hidden state."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="learn")

    view = _make_view(patch_size=9)
    player.choose_action("t1", view)
    assert len(player.episode) == 1

    player.reset_episode()
    assert len(player.episode) == 0
    # Hidden state should be reset to zeros
    assert torch.all(player._hidden == 0)


def test_player_no_patch_returns_pass() -> None:
    """If no patch found for tank_id, returns PASS."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="play")

    view = _make_view(patch_size=9)
    action = player.choose_action("nonexistent_tank", view)
    assert action == Action.PASS


def test_player_model_property() -> None:
    """The model property returns the underlying model."""
    model = _StubModel()
    player = NNPlayer(team="alpha", model=model, mode="play")
    assert player.model is model


def test_player_patch_size_from_model_config() -> None:
    """patch_size is derived from the model config."""
    config = _StubConfig(patch_size=7)
    model = _StubModel(config)
    player = NNPlayer(team="alpha", model=model, mode="play")
    assert player.patch_size == 7

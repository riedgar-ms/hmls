"""Encode a TankPatch into a multi-channel tensor for the CNN.

The encoding produces a ``[5, patch_size, patch_size]`` float tensor with
the following channels:

* Channel 0 — **terrain**: passable=1.0, impassable=0.0, fog=−1.0
* Channel 1 — **friendly tank**: 1.0 if an alive friendly tank occupies the cell
* Channel 2 — **enemy tank**: 1.0 if an alive enemy tank occupies the cell
* Channel 3 — **wreckage**: 1.0 if a dead tank (any team) occupies the cell
* Channel 4 — **visibility mask**: 1.0 if visible, 0.0 if fog
"""

from __future__ import annotations

import torch

from hmls.core.map import CellType
from hmls.core.visibility import FogCell, TankPatch, VisibleCell

NUM_CHANNELS: int = 5
"""Number of input channels produced by the encoder."""


def encode_patch(patch: TankPatch, team: str) -> torch.Tensor:
    """Convert a :class:`TankPatch` to a float tensor.

    The tensor has shape ``[NUM_CHANNELS, N, N]`` where *N* is the patch
    grid side length.

    Args:
        patch: The egocentric visibility patch for one tank.
        team: The team the observing player belongs to (used to
            distinguish friendly vs enemy tanks).

    Returns:
        A ``torch.float32`` tensor of shape ``[5, N, N]``.
    """
    n = len(patch.grid)
    tensor = torch.zeros(NUM_CHANNELS, n, n, dtype=torch.float32)

    for row_idx, row in enumerate(patch.grid):
        for col_idx, cell in enumerate(row):
            if isinstance(cell, FogCell):
                # Terrain channel gets -1 for fog
                tensor[0, row_idx, col_idx] = -1.0
                # Visibility mask = 0 (fog)
                tensor[4, row_idx, col_idx] = 0.0
            elif isinstance(cell, VisibleCell):
                # Terrain channel
                tensor[0, row_idx, col_idx] = 1.0 if cell.cell_type == CellType.PASSABLE else 0.0
                # Visibility mask = 1 (visible)
                tensor[4, row_idx, col_idx] = 1.0

                # Tank channels
                if cell.tank is not None:
                    if not cell.tank.alive:
                        # Wreckage
                        tensor[3, row_idx, col_idx] = 1.0
                    elif cell.tank.team == team:
                        # Friendly alive tank
                        tensor[1, row_idx, col_idx] = 1.0
                    else:
                        # Enemy alive tank
                        tensor[2, row_idx, col_idx] = 1.0

    return tensor

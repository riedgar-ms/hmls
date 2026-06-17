"""Patch encoding strategies for neural network players.

This module provides the :class:`FiveChannelPatchEncoder`, which converts a
:class:`~hmls.core.visibility.TankPatch` into a multi-channel float tensor
suitable for CNN-based models.

.. note::

    ``FiveChannelPatchEncoder`` is **one** encoding strategy.  Different
    neural-network architectures may benefit from different representations —
    for example, additional channels for ammunition counts, health levels,
    directional information, or entirely different spatial resolutions.
    New encoders can be added to this module (or to downstream packages) and
    used by any model that accepts the corresponding tensor shape.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from hmls.core.map import CellType
from hmls.core.visibility import BoundaryCell, FogCell, TankPatch, VisibleCell


class FiveChannelPatchEncoder:
    """Encode a :class:`TankPatch` into a 5-channel float tensor.

    The output tensor has shape ``[5, N, N]`` where *N* is the patch grid
    side length.  Each channel captures a different aspect of the local
    environment:

    * **Channel 0 — terrain**: passable = 1.0, impassable / boundary = 0.0,
      fog = −1.0
    * **Channel 1 — friendly tank**: 1.0 if an alive friendly tank occupies
      the cell
    * **Channel 2 — enemy tank**: 1.0 if an alive enemy tank occupies the
      cell
    * **Channel 3 — wreckage**: 1.0 if a dead tank (any team) occupies the
      cell
    * **Channel 4 — visibility mask**: 1.0 if visible or boundary, 0.0 if
      fog

    This is one concrete encoding strategy.  Models are free to use
    alternative encoders with different channel counts or feature
    representations.

    Example usage::

        encoder = FiveChannelPatchEncoder()
        tensor = encoder.encode_patch(patch, team="alpha")
        assert tensor.shape[0] == FiveChannelPatchEncoder.NUM_CHANNELS
    """

    class Channel(IntEnum):
        """Index of each channel in the encoded patch tensor.

        Attributes:
            TERRAIN: Passable = 1.0, impassable / boundary = 0.0, fog = −1.0.
            FRIENDLY: 1.0 if an alive friendly tank occupies the cell.
            ENEMY: 1.0 if an alive enemy tank occupies the cell.
            WRECKAGE: 1.0 if a dead tank (any team) occupies the cell.
            VISIBILITY: 1.0 if visible or boundary, 0.0 if fog.
        """

        TERRAIN = 0
        FRIENDLY = 1
        ENEMY = 2
        WRECKAGE = 3
        VISIBILITY = 4

    NUM_CHANNELS: int = len(Channel)
    """Number of input channels produced by this encoder."""

    @staticmethod
    def encode_patch(patch: TankPatch, team: str) -> torch.Tensor:
        """Convert a :class:`TankPatch` to a float tensor.

        Args:
            patch: The egocentric visibility patch for one tank.
            team: The team the observing player belongs to (used to
                distinguish friendly vs enemy tanks).

        Returns:
            A ``torch.float32`` tensor of shape
            ``[NUM_CHANNELS, N, N]`` where *N* is the patch grid side
            length.
        """
        n = len(patch.grid)
        tensor = torch.zeros(FiveChannelPatchEncoder.NUM_CHANNELS, n, n, dtype=torch.float32)
        Channel = FiveChannelPatchEncoder.Channel

        for row_idx, row in enumerate(patch.grid):
            for col_idx, cell in enumerate(row):
                if isinstance(cell, FogCell):
                    tensor[Channel.TERRAIN, row_idx, col_idx] = -1.0
                    tensor[Channel.VISIBILITY, row_idx, col_idx] = 0.0
                elif isinstance(cell, BoundaryCell):
                    tensor[Channel.TERRAIN, row_idx, col_idx] = 0.0
                    tensor[Channel.VISIBILITY, row_idx, col_idx] = 1.0
                elif isinstance(cell, VisibleCell):
                    tensor[Channel.TERRAIN, row_idx, col_idx] = (
                        1.0 if cell.cell_type == CellType.PASSABLE else 0.0
                    )
                    tensor[Channel.VISIBILITY, row_idx, col_idx] = 1.0

                    if cell.tank is not None:
                        if not cell.tank.alive:
                            tensor[Channel.WRECKAGE, row_idx, col_idx] = 1.0
                        elif cell.tank.team == team:
                            tensor[Channel.FRIENDLY, row_idx, col_idx] = 1.0
                        else:
                            tensor[Channel.ENEMY, row_idx, col_idx] = 1.0

        return tensor

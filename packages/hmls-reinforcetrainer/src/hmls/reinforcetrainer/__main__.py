"""Entry point for ``python -m hmls.reinforcetrainer``."""

from __future__ import annotations

from hmls.reinforcetrainer.cli import parse_args
from hmls.reinforcetrainer.training_loop import train


def main() -> None:
    """Parse CLI arguments and run the training loop."""
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()

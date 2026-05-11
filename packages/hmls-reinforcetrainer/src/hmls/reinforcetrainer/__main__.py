"""Entry point for ``python -m hmls.reinforcetrainer``."""

from __future__ import annotations

import logging

from hmls.reinforcetrainer.cli import parse_args
from hmls.reinforcetrainer.training_loop import train


def main() -> None:
    """Parse CLI arguments, configure logging, and run the training loop."""
    result = parse_args()
    logging.basicConfig(
        level=result.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(result.config)


if __name__ == "__main__":
    main()

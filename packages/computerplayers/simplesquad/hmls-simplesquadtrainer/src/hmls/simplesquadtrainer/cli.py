"""CLI entry point for the simple squad trainer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from hmls.simplesquadtrainer.config import SquadTrainerConfig
from hmls.simplesquadtrainer.training_loop import train


def main() -> None:
    """Run the simple squad trainer from the command line.

    Usage::

        hmls-simplesquadtrainer <config.json>
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: hmls-simplesquadtrainer <config.json>", file=sys.stderr)  # noqa: ANN401
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)  # noqa: ANN401
        sys.exit(1)

    config_text = config_path.read_text()
    config = SquadTrainerConfig.model_validate_json(config_text)
    train(config)


if __name__ == "__main__":
    main()

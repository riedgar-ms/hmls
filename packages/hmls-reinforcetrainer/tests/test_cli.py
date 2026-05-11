"""Tests for CLI argument parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmls.reinforcetrainer.cli import CLIResult, build_parser, load_config, parse_args


def _write_minimal_config(path: Path) -> Path:
    """Write a minimal valid config JSON and return its path."""
    config_data = {
        "model_a": {"dir": "models/a"},
        "model_b": {"dir": "models/b"},
    }
    config_file = path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return config_file


class TestBuildParser:
    """Tests for build_parser."""

    def test_log_level_default(self) -> None:
        """Default log level is INFO when not specified."""
        parser = build_parser()
        args = parser.parse_args(["some_config.json"])
        assert args.log_level == "INFO"

    def test_log_level_debug(self) -> None:
        """--log-level DEBUG is accepted."""
        parser = build_parser()
        args = parser.parse_args(["some_config.json", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_log_level_warning(self) -> None:
        """--log-level WARNING is accepted."""
        parser = build_parser()
        args = parser.parse_args(["some_config.json", "--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_log_level_error(self) -> None:
        """--log-level ERROR is accepted."""
        parser = build_parser()
        args = parser.parse_args(["some_config.json", "--log-level", "ERROR"])
        assert args.log_level == "ERROR"

    def test_log_level_invalid_rejected(self) -> None:
        """An invalid log level is rejected by argparse."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["some_config.json", "--log-level", "TRACE"])


class TestParseArgs:
    """Tests for parse_args return type."""

    def test_returns_cli_result(self, tmp_path: Path) -> None:
        """parse_args returns a CLIResult with config and log_level."""
        config_file = _write_minimal_config(tmp_path)
        result = parse_args([str(config_file)])
        assert isinstance(result, CLIResult)
        assert result.log_level == "INFO"
        assert result.config.model_a.train is True

    def test_log_level_passed_through(self, tmp_path: Path) -> None:
        """--log-level value is available on the CLIResult."""
        config_file = _write_minimal_config(tmp_path)
        result = parse_args([str(config_file), "--log-level", "DEBUG"])
        assert result.log_level == "DEBUG"


class TestLoadConfig:
    """Tests for load_config error handling."""

    def test_missing_config_file_exits(self, tmp_path: Path) -> None:
        """load_config raises SystemExit for a missing file."""
        with pytest.raises(SystemExit):
            load_config(tmp_path / "nonexistent.json")

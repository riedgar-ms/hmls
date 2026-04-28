"""Sanity checks for replay viewer app constants."""

from hmls.replayviewer.app import _DEFAULT_DELAY, _DELAY_STEP, _MAX_DELAY, _MIN_DELAY


class TestDelayConstants:
    """Verify that the auto-play delay constants are consistent."""

    def test_min_less_than_default(self) -> None:
        """Minimum delay must be less than the default."""
        assert _MIN_DELAY < _DEFAULT_DELAY

    def test_default_less_than_max(self) -> None:
        """Default delay must be less than the maximum."""
        assert _DEFAULT_DELAY < _MAX_DELAY

    def test_step_positive(self) -> None:
        """Delay step must be positive."""
        assert _DELAY_STEP > 0

    def test_step_smaller_than_range(self) -> None:
        """Delay step should be smaller than the total delay range."""
        assert _DELAY_STEP < (_MAX_DELAY - _MIN_DELAY)

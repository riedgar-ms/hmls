"""Patch encoding for the Mk-II model.

The Mk-II currently uses the same 5-channel encoding as the Mk-I.
This module re-exports the encoding function and constants from
:mod:`hmls.singlemki.encoding` for convenience and to allow future
divergence without breaking Mk-II consumers.
"""

from hmls.singlemki.encoding import NUM_CHANNELS, Channel, encode_patch

__all__ = ["NUM_CHANNELS", "Channel", "encode_patch"]

"""Python package wrapper for the compiled C++ bindings.

This module imports the compiled extension `bindings` (built from
`src/my_keypoints/bindings.cc`) and re-exports its symbols so that
`import my_keypoints` exposes the expected functions.
"""

from .bindings import *  # noqa: F401,F403 - re-export compiled symbols

__all__ = [name for name in globals() if not name.startswith("_")]

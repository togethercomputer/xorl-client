"""
Request ID type alias.

This module provides the RequestID type alias used throughout the two-phase
request pattern for identifying async operations.
"""

from typing import TypeAlias

# RequestID is a string that uniquely identifies an async request
# Format: "future_{uuid_hex}" (e.g., "future_a1b2c3d4e5f6")
RequestID: TypeAlias = str

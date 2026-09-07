"""
tests/conftest.py — shared pytest configuration and fixtures.

Adds the project root to sys.path so tests can import specbench without
installing it. Provides a marker for slow tests that require GPU and models.
"""

import sys
from pathlib import Path

# Ensure the project root is importable from any test.
sys.path.insert(0, str(Path(__file__).parent.parent))

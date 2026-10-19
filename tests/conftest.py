"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest
import torch

# Shared test helpers (e.g. ``tiny_terra``) live next to this file; with
# ``--import-mode=importlib`` pytest does not put this directory on sys.path.
sys.path.insert(0, str(Path(__file__).parent))


def _available_devices() -> list[str]:
    """Return every torch device type usable on this machine (CPU first)."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request) -> torch.device:
    """Parametrize a test over every available device (cpu / mps / cuda)."""
    return torch.device(request.param)

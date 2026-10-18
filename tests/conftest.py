"""Shared pytest fixtures."""

import pytest
import torch


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

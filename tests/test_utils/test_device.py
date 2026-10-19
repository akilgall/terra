"""Tests for ``terra.utils.device`` (device selection, autocast, pin_memory).

Availability of CUDA / MPS is monkeypatched so the selection order is
tested deterministically on any machine. Tests that need a real
accelerator are skipped when it is absent.
"""

import contextlib
import logging

import pytest
import torch

from terra.utils import device as device_utils
from terra.utils.device import autocast, get_device, should_pin_memory

_HAS_MPS = torch.backends.mps.is_available()


@pytest.fixture
def fake_backends(monkeypatch):
    """Control what ``get_device`` believes is available."""
    set_device_calls = []

    def configure(cuda: bool = False, mps: bool = False, cuda_count: int = 1):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda_count if cuda else 0)
        monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        return set_device_calls

    return configure


@pytest.fixture(autouse=True)
def _reset_autocast_support_cache():
    device_utils._autocast_supported.cache_clear()
    yield
    device_utils._autocast_supported.cache_clear()


# ---------------------------------------------------------------------------
# get_device: automatic selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested", [None, "auto"])
def test_auto_prefers_cuda_over_mps(fake_backends, requested):
    set_device_calls = fake_backends(cuda=True, mps=True)

    device = get_device(requested)

    assert device == torch.device("cuda", 0)
    assert set_device_calls == [torch.device("cuda", 0)]


def test_auto_uses_local_rank_for_cuda(fake_backends):
    fake_backends(cuda=True, cuda_count=4)

    assert get_device(None, local_rank=3) == torch.device("cuda", 3)


def test_auto_falls_back_to_mps_without_cuda(fake_backends):
    set_device_calls = fake_backends(cuda=False, mps=True)

    assert get_device() == torch.device("mps")
    assert set_device_calls == []


def test_auto_falls_back_to_cpu_without_accelerators(fake_backends):
    fake_backends(cuda=False, mps=False)

    assert get_device() == torch.device("cpu")


# ---------------------------------------------------------------------------
# get_device: explicit requests
# ---------------------------------------------------------------------------

def test_explicit_cpu_is_honoured_even_when_accelerators_exist(fake_backends):
    fake_backends(cuda=True, mps=True)

    assert get_device("cpu") == torch.device("cpu")


def test_explicit_mps_is_honoured(fake_backends):
    fake_backends(cuda=True, mps=True)

    assert get_device("mps") == torch.device("mps")


def test_explicit_cuda_without_index_defaults_to_device_zero(fake_backends):
    fake_backends(cuda=True, cuda_count=2)

    assert get_device("cuda") == torch.device("cuda", 0)


def test_explicit_cuda_without_index_gets_local_rank(fake_backends):
    set_device_calls = fake_backends(cuda=True, cuda_count=2)

    assert get_device("cuda", local_rank=1) == torch.device("cuda", 1)
    assert set_device_calls == [torch.device("cuda", 1)]


def test_explicit_torch_device_is_accepted(fake_backends):
    fake_backends()

    assert get_device(torch.device("cpu")) == torch.device("cpu")


def test_unavailable_cuda_raises(fake_backends):
    fake_backends(cuda=False, mps=True)

    with pytest.raises(ValueError, match="CUDA is not available"):
        get_device("cuda")


def test_out_of_range_cuda_index_raises(fake_backends):
    fake_backends(cuda=True, cuda_count=1)

    with pytest.raises(ValueError, match="cuda:2"):
        get_device("cuda:2")


def test_unavailable_mps_raises(fake_backends):
    fake_backends(cuda=False, mps=False)

    with pytest.raises(ValueError, match="MPS is not available"):
        get_device("mps")


@pytest.mark.parametrize("bad", ["gpu", "xla", "metal"])
def test_unsupported_device_string_raises(fake_backends, bad):
    fake_backends()

    with pytest.raises(ValueError, match="Unsupported device"):
        get_device(bad)


# ---------------------------------------------------------------------------
# autocast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [True, False])
def test_autocast_is_a_noop_on_cpu(enabled):
    ctx = autocast(torch.device("cpu"), enabled=enabled)

    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        out = torch.nn.Linear(4, 4)(torch.randn(2, 4))
    assert out.dtype == torch.float32


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
def test_autocast_runs_bfloat16_on_mps_when_enabled():
    mps = torch.device("mps")
    layer = torch.nn.Linear(4, 4).to(mps)

    with autocast(mps, enabled=True):
        out = layer(torch.randn(2, 4, device=mps))

    assert out.dtype == torch.bfloat16


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
def test_autocast_keeps_float32_on_mps_when_disabled():
    mps = torch.device("mps")
    layer = torch.nn.Linear(4, 4).to(mps)

    with autocast(mps, enabled=False):
        out = layer(torch.randn(2, 4, device=mps))

    assert out.dtype == torch.float32


def test_autocast_falls_back_to_noop_when_backend_rejects_it(monkeypatch, caplog):
    def _reject(*args, **kwargs):
        raise RuntimeError("unsupported autocast device_type 'mps'")

    monkeypatch.setattr(torch, "autocast", _reject)

    with caplog.at_level(logging.WARNING, logger="terra.utils.device"):
        first = autocast(torch.device("mps"), enabled=True)
        second = autocast(torch.device("mps"), enabled=True)

    assert isinstance(first, contextlib.nullcontext)
    assert isinstance(second, contextlib.nullcontext)
    # The warning is emitted once, not on every batch.
    assert len([r for r in caplog.records if "autocast" in r.getMessage()]) == 1


def test_autocast_errors_on_cuda_still_propagate(monkeypatch):
    # e.g. bfloat16 requested on a pre-Ampere GPU: keep failing loudly, as
    # torch.cuda.amp.autocast did, instead of silently running in float32.
    def _reject(*args, **kwargs):
        raise RuntimeError("Current CUDA Device does not support bfloat16.")

    monkeypatch.setattr(torch, "autocast", _reject)

    with pytest.raises(RuntimeError, match="does not support bfloat16"):
        autocast(torch.device("cuda", 0), enabled=True)


# ---------------------------------------------------------------------------
# should_pin_memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("device_type", "requested", "expected"),
    [
        ("cuda", True, True),
        ("cuda", False, False),
        ("mps", True, False),
        ("cpu", True, False),
        ("cpu", False, False),
    ],
)
def test_should_pin_memory_only_on_cuda(device_type, requested, expected):
    assert should_pin_memory(torch.device(device_type), requested) is expected

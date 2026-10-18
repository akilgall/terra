"""Device handling in ``terra.training.decode``.

``decode.py`` historically took an integer GPU id (``-1`` meaning CPU).
``_resolve_device`` must keep accepting those legacy values while also
taking device strings (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda:N"``),
and the count decoder must train on every available device.
"""

import pytest
import torch

from terra.training.decode import CountDecoder, _resolve_device, compute_loss


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)


@pytest.mark.parametrize("value", [-1, "-1"])
def test_legacy_minus_one_means_cpu(fake_cuda, value):
    assert _resolve_device(value) == torch.device("cpu")


@pytest.mark.parametrize("value", [1, "1"])
def test_legacy_gpu_id_selects_that_cuda_device(fake_cuda, value):
    assert _resolve_device(value) == torch.device("cuda", 1)


@pytest.mark.parametrize("mps_available", [True, False])
def test_legacy_gpu_id_without_cuda_falls_back_to_auto(no_cuda, monkeypatch, mps_available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    expected = torch.device("mps") if mps_available else torch.device("cpu")
    assert _resolve_device(0) == expected
    assert _resolve_device(None) == expected
    assert _resolve_device("auto") == expected


def test_device_strings_pass_through(no_cuda):
    assert _resolve_device("cpu") == torch.device("cpu")


def test_unavailable_device_string_raises(no_cuda):
    with pytest.raises(ValueError, match="CUDA is not available"):
        _resolve_device("cuda:0")


@pytest.mark.parametrize("value", ["--1", "1.5", "gpu"])
def test_malformed_device_string_raises_value_error(no_cuda, value):
    with pytest.raises(ValueError, match="Unsupported device"):
        _resolve_device(value)


@pytest.mark.parametrize("loss_type", ["nb_softplus", "nb_libsize"])
def test_count_decoder_trains_on_device(device, loss_type):
    torch.manual_seed(0)
    n_genes = 7
    mean_activation = "softmax" if loss_type == "nb_libsize" else "softplus"
    decoder = CountDecoder(
        embed_dim=8, hidden_dim=16, n_genes=n_genes, depth=2,
        mean_activation=mean_activation,
    ).to(device)
    emb = torch.randn(5, 8, device=device)
    target = torch.randint(0, 20, (5, n_genes), device=device).float()
    library_size = target.sum(dim=1, keepdim=True).clamp(min=1.0)

    mu, theta = decoder(emb)
    loss = compute_loss(mu, theta, target, loss_type, library_size=library_size)
    loss.backward()

    assert loss.device.type == device.type
    assert torch.isfinite(loss)
    grads = [p.grad for p in decoder.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

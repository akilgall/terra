"""Device portability of the TERRA encoder (CPU / Apple MPS / CUDA).

These tests run a tiny real ``GeneTransformerRankEncoder`` on every
available device via the ``device`` fixture (see ``tests/conftest.py``).
They use ``use_flash_attention=True`` because every shipped model config
enables it, so the ``sdpa_kernel`` backend-selection path is exercised.

Run with ``PYTORCH_ENABLE_MPS_FALLBACK`` unset so unsupported MPS ops fail
instead of silently running on CPU.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from terra.models.gene_transformers import GeneTransformerRankEncoder, _eigh
from terra.models.modules import ClassificationModel

SEQ_LEN = 12
N_SPECIAL = 2
N_SEGMENTS = 2


def _make_encoder(cell_pos_enc: str) -> GeneTransformerRankEncoder:
    return GeneTransformerRankEncoder(
        vocab_size=16,
        seq_len=SEQ_LEN,
        n_special_tokens=N_SPECIAL,
        n_segments=N_SEGMENTS,
        cell_pos_enc=cell_pos_enc,
        embed_dim=32,
        depth=1,
        num_heads=4,
        use_flash_attention=True,
    )


def _make_batch(encoder, batch_size: int, device: torch.device) -> dict:
    """Two cells laid out as [special][cell 1][cell 2], with 2D positions."""
    seq_len_cell = encoder.seq_len_cell
    segments = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    rel_x = torch.full((batch_size, SEQ_LEN), float("-inf"))
    rel_y = torch.full((batch_size, SEQ_LEN), float("-inf"))
    cell_positions = torch.randn(batch_size, N_SEGMENTS, 2)
    for c in range(N_SEGMENTS):
        start = N_SPECIAL + c * seq_len_cell
        end = start + seq_len_cell
        segments[:, start:end] = c + 1
        rel_x[:, start:end] = cell_positions[:, c, 0:1]
        rel_y[:, start:end] = cell_positions[:, c, 1:2]
    batch = {
        "positions": torch.arange(SEQ_LEN).repeat(batch_size, 1),
        "segments": segments,
        "tokens": torch.randint(1, 16, (batch_size, SEQ_LEN)),
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
    }
    return {key: value.to(device) for key, value in batch.items()}


@pytest.mark.parametrize("cell_pos_enc", ["segment", "laplacian"])
def test_encoder_forward_backward_on_device(device, cell_pos_enc):
    torch.manual_seed(0)
    encoder = _make_encoder(cell_pos_enc).to(device)
    batch = _make_batch(encoder, batch_size=3, device=device)

    out, _ = encoder(batch)
    out.float().pow(2).mean().backward()

    assert out.device.type == device.type
    assert out.shape == (3, SEQ_LEN, 32)
    assert torch.isfinite(out).all()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads, "no gradients reached the encoder parameters"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("cell_pos_enc", ["segment", "laplacian"])
def test_accelerator_output_matches_cpu(device, cell_pos_enc):
    if device.type == "cpu":
        pytest.skip("reference device")
    torch.manual_seed(0)
    encoder = _make_encoder(cell_pos_enc).eval()
    batch = _make_batch(encoder, batch_size=3, device=torch.device("cpu"))

    with torch.no_grad():
        expected, _ = encoder(batch)
        actual, _ = encoder.to(device)(
            {key: value.to(device) for key, value in batch.items()})

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_classification_head_trains_on_device(device):
    class _Backbone:
        embed_dim = 32

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()
            self.encoder = _make_encoder("segment")

        def forward(self, batch, masks_attention=None):
            return self.encoder(batch, masks_attention=masks_attention)

    torch.manual_seed(0)
    model = ClassificationModel(base_model=_Base(), num_classes=3, use_mlp=True).to(device)
    batch = _make_batch(model.base_model.encoder, batch_size=4, device=device)
    selection_mask = torch.zeros(4, SEQ_LEN, dtype=torch.bool, device=device)
    selection_mask[:, N_SPECIAL:] = True
    labels = torch.tensor([0, 1, 2, 1], device=device)

    logits = model(batch, masks_attention=None, selection_mask=selection_mask)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (4, 3)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# _eigh: CPU fallback for backends without linalg.eigh (MPS on torch < 2.10)
# ---------------------------------------------------------------------------

def _symmetric(n: int = 5, batch: int = 2) -> torch.Tensor:
    a = torch.randn(batch, n, n)
    return a + a.transpose(-1, -2)


def test_eigh_falls_back_to_cpu_when_backend_lacks_it(device, monkeypatch):
    if device.type == "cpu":
        pytest.skip("fallback only applies to accelerator tensors")
    real_eigh = torch.linalg.eigh

    def eigh_without_accelerator_support(matrix):
        if matrix.device.type != "cpu":
            raise NotImplementedError("aten::_linalg_eigh.eigenvalues is not implemented")
        return real_eigh(matrix)

    monkeypatch.setattr(torch.linalg, "eigh", eigh_without_accelerator_support)
    matrix = _symmetric()

    eigvals, eigvecs = _eigh(matrix.to(device))

    expected_vals, _ = real_eigh(matrix)
    assert eigvals.device.type == device.type
    assert eigvecs.device.type == device.type
    torch.testing.assert_close(eigvals.cpu(), expected_vals, rtol=1e-4, atol=1e-4)


def test_eigh_reraises_on_cpu(monkeypatch):
    def broken_eigh(matrix):
        raise NotImplementedError("broken")

    monkeypatch.setattr(torch.linalg, "eigh", broken_eigh)

    with pytest.raises(NotImplementedError, match="broken"):
        _eigh(_symmetric())


def test_eigh_matches_torch_on_cpu():
    matrix = _symmetric()

    eigvals, eigvecs = _eigh(matrix)

    expected_vals, expected_vecs = torch.linalg.eigh(matrix)
    torch.testing.assert_close(eigvals, expected_vals)
    torch.testing.assert_close(eigvecs, expected_vecs)


# ---------------------------------------------------------------------------
# _sdpa: batch-chunked attention for backends with 32-bit indexing (MPS)
# ---------------------------------------------------------------------------

def _attention_inputs(device, batch=5, heads=2, length=6, dim=4):
    torch.manual_seed(0)
    q, k, v = (torch.randn(batch, heads, length, dim, device=device, requires_grad=True)
               for _ in range(3))
    return q, k, v


def _masks(device, batch=5, heads=2, length=6):
    bool_mask = torch.rand(batch, 1, length, length, device=device) > 0.3
    bool_mask[..., 0] = True  # keep every row attendable
    return {
        "none": None,
        "bool_per_sample": bool_mask,
        "float_per_head": torch.randn(batch, heads, length, length, device=device),
        "bool_broadcast": bool_mask[:1],
    }


@pytest.mark.parametrize("mask_kind", ["none", "bool_per_sample", "float_per_head", "bool_broadcast"])
def test_sdpa_chunks_over_batch_when_scores_exceed_backend_limit(device, monkeypatch, mask_kind):
    from terra.models import modules

    q, k, v = _attention_inputs(device)
    mask = _masks(device)[mask_kind]
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.5)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))

    # One sample's scores are 2 * 6 * 6 = 72 elements: force chunks of 2.
    calls = []
    real_sdpa = F.scaled_dot_product_attention

    def counting_sdpa(*args, **kwargs):
        calls.append(args[0].size(0))
        return real_sdpa(*args, **kwargs)

    monkeypatch.setitem(modules._MAX_ATTN_SCORES, device.type, 150)
    monkeypatch.setattr(modules.F, "scaled_dot_product_attention", counting_sdpa)
    actual = modules._sdpa(q, k, v, mask, 0.5)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))

    assert calls == [2, 2, 1]
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)


def test_sdpa_does_not_chunk_below_limit_or_without_limit(device, monkeypatch):
    from terra.models import modules

    q, k, v = _attention_inputs(device)
    calls = []
    real_sdpa = F.scaled_dot_product_attention

    def counting_sdpa(*args, **kwargs):
        calls.append(args[0].size(0))
        return real_sdpa(*args, **kwargs)

    monkeypatch.setattr(modules.F, "scaled_dot_product_attention", counting_sdpa)
    modules._sdpa(q, k, v, None, 0.5)

    assert calls == [5]

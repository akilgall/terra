"""Per-block gradient checkpointing for long (asymmetric) sequences.

Checkpointing trades compute for memory: block activations are recomputed in
the backward pass. It must give exactly the same outputs and gradients, run
only in training with gradients enabled, and refuse blocks with dropout on
MPS, where torch does not replay the random masks during recomputation (the
gradients would silently be wrong).
"""

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.masks.block_masking import BlockMaskCollator  # noqa: E402
from terra.models import gene_transformers  # noqa: E402
from terra.models.gene_transformers import GeneTransformerRankEncoder  # noqa: E402
from terra.utils.helper import init_model  # noqa: E402

ROWS = {
    "gene_tokens": [[5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0],
                    [11, 12, 13, 14, 0, 0, 0, 0, 15, 16, 0, 0]],
    "gene_expr": [[3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                  [1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 2.2, 1.1, 0.0, 0.0]],
    "rel_x_coord": [[0.0, 1.5, -2.0], [0.0, 1.0, 2.0]],
    "rel_y_coord": [[0.0, 0.5, 3.0], [0.0, -1.0, 1.0]],
    "n_nonzero_tokens": [6, 6],
}


def _jepa_batch():
    ds = datasets.Dataset.from_dict(ROWS).with_format("torch")
    cell_ds = CellGraphDataset(gt_type="combined", cell_pos_enc="coord", dataset=ds,
                               vocab_size=20, seq_len_cell=4, seq_len_neighborhood=8,
                               special_tokens=[], sampling_strategy=None)
    collator = BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=4,
                                 seq_len_neighborhood=8, n_special_tokens=0,
                                 per_block_mask_ratio=0.5)
    torch.manual_seed(7)
    udata, masks_enc, masks_pred, _, _ = collator([cell_ds[0], cell_ds[1]])
    return udata, list(masks_enc), list(masks_pred)


def _models(grad_checkpointing: bool):
    torch.manual_seed(0)
    return init_model(
        gt_type="combined", count_encoding="mlp", n_value_bins=100, cell_pos_enc="coord",
        device="cpu", vocab_size=20, seq_len=12, n_special_tokens=0, n_segments=3,
        n_special_values=0, enc_emb_dim=32, enc_depth=2, pred_emb_dim=16, pred_depth=2,
        num_heads=4, mlp_ratio=2.0, use_flash_attention=True, api_version="v3",
        predict_gene=True, mlp_bias=True, grad_checkpointing=grad_checkpointing)


def _jepa_step(encoder, predictor, udata, masks_enc, masks_pred):
    z, token_emb = encoder(batch=udata, masks=masks_enc, masks_attention=None)
    preds = predictor(z=z, token_emb=token_emb, batch=udata, masks_enc=masks_enc,
                      masks_pred=masks_pred, masks_attention=None)
    loss = sum(p.float().pow(2).mean() for p in preds)
    loss.backward()
    return loss.detach()


def test_checkpointing_gives_identical_loss_and_gradients():
    udata, masks_enc, masks_pred = _jepa_batch()
    plain = _models(grad_checkpointing=False)
    ckpt = _models(grad_checkpointing=True)

    loss_plain = _jepa_step(*plain, udata, masks_enc, masks_pred)
    loss_ckpt = _jepa_step(*ckpt, udata, masks_enc, masks_pred)

    torch.testing.assert_close(loss_ckpt, loss_plain, rtol=0, atol=1e-6)
    for model_plain, model_ckpt in zip(plain, ckpt, strict=True):
        for (name, p1), (_, p2) in zip(model_plain.named_parameters(),
                                       model_ckpt.named_parameters(), strict=True):
            if p1.grad is None:
                assert p2.grad is None, name
                continue
            torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-5, atol=1e-6, msg=name)


def _encoder(**overrides) -> GeneTransformerRankEncoder:
    kwargs = {"vocab_size": 20, "seq_len": 12, "n_special_tokens": 0, "n_segments": 3,
              "cell_pos_enc": "segment", "embed_dim": 16, "depth": 2, "num_heads": 2,
              "use_flash_attention": True, "grad_checkpointing": True}
    kwargs.update(overrides)
    return GeneTransformerRankEncoder(**kwargs)


def _rank_batch(device="cpu") -> dict:
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], device=device)
    return {"tokens": tokens,
            "positions": torch.tensor([[1, 2, 3, 4] * 3], device=device),
            "segments": torch.tensor([[1] * 4 + [2] * 4 + [3] * 4], device=device)}


@pytest.fixture
def checkpoint_calls(monkeypatch):
    calls = []
    real = gene_transformers.checkpoint

    def spy(*args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real(*args, **kwargs)

    monkeypatch.setattr(gene_transformers, "checkpoint", spy)
    return calls


def test_checkpointing_only_in_training_with_grad(checkpoint_calls):
    encoder = _encoder()

    encoder.eval()
    encoder(_rank_batch())
    encoder.train()
    with torch.no_grad():
        encoder(_rank_batch())
    assert checkpoint_calls == []

    encoder(_rank_batch())
    assert checkpoint_calls == [False, False]  # one per block, non-reentrant


def test_checkpointing_off_by_default(checkpoint_calls):
    _encoder(grad_checkpointing=False)(_rank_batch())

    assert checkpoint_calls == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_refuses_checkpointing_blocks_with_dropout():
    encoder = _encoder(drop_rate=0.1).to("mps").train()

    with pytest.raises(RuntimeError, match="dropout"):
        encoder(_rank_batch("mps"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_allows_checkpointing_without_dropout():
    encoder = _encoder().to("mps").train()

    out, _ = encoder(_rank_batch("mps"))
    out.float().pow(2).mean().backward()

    assert all(p.grad is not None for p in encoder.blocks.parameters())

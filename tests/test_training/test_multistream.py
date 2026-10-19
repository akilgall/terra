"""Stream specs, the stream-homogeneous sampler and per-stream batch views."""

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.masks.block_masking import BlockMaskCollator  # noqa: E402
from terra.models.gene_transformers import GeneTransformerRankEncoder  # noqa: E402
from terra.training.multistream import (  # noqa: E402
    MultiStreamCollator,
    StreamBatchSampler,
    StreamTaggedDataset,
    parse_streams,
    trim_trailing_pad,
)


def test_parse_streams_absent_means_legacy():
    assert parse_streams({"data": {"finetune_dataset": ["a"]}, "finetune": {}}) is None


def test_parse_streams_reads_specs():
    specs = parse_streams({"data": {"streams": {
        "spatial": {"kind": "spatial_full", "paths": ["s"]},
        "tahoe": {"kind": "cell_only", "paths": ["t"], "weight": 2.0, "batch_size": 8,
                  "tokenized_seq_len_cell": 2048}}}})

    assert [s.name for s in specs] == ["spatial", "tahoe"]
    assert specs[1].weight == 2.0 and specs[1].batch_size == 8
    assert specs[1].tokenized_seq_len_cell == 2048
    assert specs[0].paths == ("s",)


@pytest.mark.parametrize("bad", [
    {"kind": "nope", "paths": ["s"]},
    {"kind": "cell_only", "paths": []},
    {"kind": "cell_only", "paths": ["s"], "weight": 0},
])
def test_parse_streams_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_streams({"data": {"streams": {"x": bad}}})


def _sampler(**kwargs):
    defaults = {"lengths": [10, 4], "batch_sizes": [2, 2], "weights": [1.0, 1.0],
                "seed": 0, "sort_keys": None, "epoch_batches": None}
    defaults.update(kwargs)
    return StreamBatchSampler(**defaults)


def test_sampler_batches_are_stream_homogeneous():
    sampler = _sampler()
    batches = list(sampler)

    for batch in batches:
        assert all(i < 10 for i in batch) or all(i >= 10 for i in batch)
    assert len(batches) == len(sampler)


def test_sampler_splits_steps_by_weight():
    batches = list(_sampler(weights=[1.0, 3.0], epoch_batches=8))

    from_second = sum(1 for b in batches if b[0] >= 10)
    assert len(batches) == 8
    assert from_second == 6


def test_sampler_is_deterministic_and_reshuffles_per_epoch():
    a, b = _sampler(seed=3), _sampler(seed=3)
    assert list(a) == list(b)

    a.set_epoch(1)
    assert list(a) != list(b)


def test_sampler_groups_lengths_within_stream():
    keys = [[10, 1, 9, 2, 8, 3, 7, 4, 6, 5], [1, 2, 3, 4]]
    batches = [b for b in _sampler(sort_keys=keys, mega_batch=5) if b[0] < 10]

    spreads = [abs(keys[0][b[0]] - keys[0][b[1]]) for b in batches]
    assert max(spreads) <= 1


ROWS = {
    "gene_tokens": [[5, 6, 7, 0, 8, 9, 10, 0], [11, 12, 0, 0, 13, 0, 14, 15]],
    "gene_expr": [[3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 4.0, 0.0], [1.0, 0.5, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0]],
    "rel_x_coord": [[0.0, 1.5, -2.0], [0.0, 1.0, 2.0]],
    "rel_y_coord": [[0.0, 0.5, 3.0], [0.0, -1.0, 1.0]],
    "n_nonzero_tokens": [6, 6],
    "lineage_value": [0, -1],
}


def _cell_dataset():
    ds = datasets.Dataset.from_dict(ROWS).with_format("torch")
    return CellGraphDataset(gt_type="combined", cell_pos_enc="coord", dataset=ds, vocab_size=20,
                            seq_len_cell=4, seq_len_neighborhood=4, seq_len_neigh=2,
                            special_tokens=[], sampling_strategy=None)


def _collator(kinds, trim=True):
    masker = BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=4,
                               seq_len_neighborhood=4, seq_len_neigh=2, n_special_tokens=0,
                               per_block_mask_ratio=0.5)
    return MultiStreamCollator(masker, kinds=kinds, trim_trailing_pad=trim)


def _items(stream_id):
    tagged = StreamTaggedDataset(_cell_dataset(), stream_id)
    return [tagged[0], tagged[1]]


def test_spatial_full_view_keeps_neighbours():
    torch.manual_seed(0)
    batch = _collator(["spatial_full"])(_items(0))

    assert batch.kind == "spatial_full" and batch.full is None
    assert batch.jepa.udata["tokens"].shape == (2, 8)
    assert batch.jepa.udata["lineage_value"].tolist() == [0, -1]
    assert batch.jepa.masks_enc[0].max() < 8


def test_cell_only_view_hides_neighbours_and_trims():
    torch.manual_seed(0)
    batch = _collator(["spatial_full", "spatial_cell_only"])(_items(1))

    view = batch.jepa.udata
    assert batch.kind == "spatial_cell_only"
    assert view["tokens"].tolist() == [[5, 6, 7], [11, 12, 0]]  # trimmed to T=3
    assert torch.isinf(view["rel_x_coords"][1, 2])
    for masks in (batch.jepa.masks_enc, batch.jepa.masks_pred):
        assert all(m.max() < 3 for m in masks)
    assert batch.full.udata["tokens"].shape == (2, 8)  # full view untouched
    assert batch.full.udata["tokens"][0, 4:].tolist() == [8, 9, 10, 0]


def test_trim_can_be_disabled():
    batch = _collator(["spatial_cell_only"], trim=False)(_items(0))

    assert batch.jepa.udata["tokens"].shape == (2, 8)
    assert batch.jepa.udata["tokens"][0, 4:].tolist() == [0, 0, 0, 0]


def test_mixed_batch_raises():
    items = _items(0)[:1] + _items(1)[:1]

    with pytest.raises(ValueError, match="stream"):
        _collator(["spatial_full", "cell_only"])(items)


def test_trim_trailing_pad_is_exact_for_encoder(device):
    torch.manual_seed(0)
    encoder = GeneTransformerRankEncoder(
        vocab_size=20, seq_len=8, n_special_tokens=0, n_segments=3, cell_pos_enc="segment",
        embed_dim=16, depth=2, num_heads=2, use_flash_attention=True, seq_len_cell=4).eval()
    tokens = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0], [11, 12, 0, 0, 0, 0, 0, 0]])
    batch = {"tokens": tokens,
             "positions": torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0]]),
             "segments": (tokens != 0).long(),
             "lineage_value": torch.tensor([0, -1])}
    trimmed = trim_trailing_pad(batch)
    encoder = encoder.to(device)

    def run(b):
        b = {k: v.to(device) for k, v in b.items()}
        attn = (b["tokens"] != 0)[:, None, None, :]
        out, _ = encoder(b, masks_attention=attn)
        return out[:, :2].cpu()

    with torch.no_grad():
        torch.testing.assert_close(run(trimmed), run(batch), rtol=1e-5, atol=1e-5)
    assert trimmed["tokens"].shape == (2, 3)
    assert trimmed["lineage_value"].tolist() == [0, -1]

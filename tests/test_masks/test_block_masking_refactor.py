"""``BlockMaskCollator``: legacy outputs pinned, extracted helpers, asymmetric cutoff.

``GOLDEN`` holds the collator outputs of the unmodified base code (commit
4bf0ad6) under ``torch.manual_seed(1234)`` for three sampling configurations,
so the helper extraction is proven behaviour-preserving.
"""

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.masks.block_masking import BlockMaskCollator  # noqa: E402

INF = float("-inf")

LEGACY_ROWS = {
    "gene_tokens": [
        [5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0],
        [11, 12, 13, 14, 0, 0, 0, 0, 15, 16, 0, 0],
    ],
    "gene_expr": [
        [3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 2.2, 1.1, 0.0, 0.0],
    ],
    "rel_x_coord": [[0.0, 1.5, -2.0], [0.0, 1.0, 2.0]],
    "rel_y_coord": [[0.0, 0.5, 3.0], [0.0, -1.0, 1.0]],
    "n_nonzero_tokens": [6, 6],
    "cell_id": ["c0", "c1"],
}

GOLDEN = {
    "plain": {
        "kwargs": {"sample_segments": False},
        "tokens": [[5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0], [11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0]],
        "ctx": [[[0], [2]]],
        "tgt": [[[4, 2, 5, 8, 1], [4, 3, 1, 0, 5]]],
    },
    "segments_ratio1": {
        "kwargs": {"sample_segments": True, "cell_segment_sampling_ratio": 1.0},
        "tokens": [[5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0]],
        "ctx": [[[], []]],
        "tgt": [[[0, 1, 2], [1, 3, 2]]],
    },
    "segments_ratio0": {
        "kwargs": {"sample_segments": True, "cell_segment_sampling_ratio": 0.0},
        "tokens": [[5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0], [11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0]],
        "ctx": [[[8], [4]]],
        "tgt": [[[0, 1, 2, 5, 4], [3, 1, 0, 2, 5]]],
    },
}


def _batch() -> list[dict]:
    ds = datasets.Dataset.from_dict(LEGACY_ROWS).with_format(
        "torch", columns=[c for c in LEGACY_ROWS if c != "cell_id"], output_all_columns=True)
    cell_ds = CellGraphDataset(
        gt_type="combined", cell_pos_enc="coord", dataset=ds, vocab_size=20, seq_len_cell=4,
        seq_len_neighborhood=8, special_tokens=[], sampling_strategy=None, include_cell_id=True)
    return [cell_ds[0], cell_ds[1]]


def _collator(**kwargs) -> BlockMaskCollator:
    return BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=4,
                             seq_len_neighborhood=8, n_special_tokens=0,
                             per_block_mask_ratio=0.8, sample_gene_masks=True, **kwargs)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_call_matches_pinned_legacy_outputs(name):
    golden = GOLDEN[name]
    torch.manual_seed(1234)

    collated, ctx, tgt, attn, pad = _collator(**golden["kwargs"])(_batch())

    assert collated["tokens"].tolist() == golden["tokens"]
    assert ctx.tolist() == golden["ctx"]
    assert tgt.tolist() == golden["tgt"]
    assert torch.equal(attn.squeeze(1).squeeze(1), collated["tokens"] != 0)
    assert pad is False


def test_sample_masks_for_tokens_equals_call():
    coll = _collator(sample_segments=False)
    torch.manual_seed(1234)
    collated, ctx_call, tgt_call, _, _ = coll(_batch())

    torch.manual_seed(1234)
    ctx, tgt = coll.sample_masks_for_tokens(collated["tokens"])

    assert torch.equal(ctx, ctx_call)
    assert torch.equal(tgt, tgt_call)


def test_stack_min_trimmed_trims_to_shortest_row():
    tgt_list = [[torch.tensor([4, 2, 9])], [torch.tensor([1, 3])]]
    ctx_list = [[torch.tensor([7])], [torch.tensor([5, 6, 8])]]

    tgt, ctx = BlockMaskCollator.stack_min_trimmed(tgt_list, ctx_list, seq_len_genes=12)

    assert tgt.shape == (1, 2, 2)
    assert tgt.tolist() == [[[4, 2], [1, 3]]]
    assert ctx.tolist() == [[[7], [5]]]
    assert tgt.dtype == ctx.dtype == torch.long


def test_zero_beyond_zeroes_tail_only():
    collated = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "segments": torch.tensor([[1, 1, 2, 2]]),
        "positions": torch.tensor([[1, 2, 1, 2]]),
        "values": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "rel_x_coords": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "rel_y_coords": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
    }

    BlockMaskCollator.zero_beyond(collated, cutoff=2)

    assert collated["tokens"].tolist() == [[1, 2, 0, 0]]
    assert collated["segments"].tolist() == [[1, 1, 0, 0]]
    assert collated["positions"].tolist() == [[1, 2, 0, 0]]
    assert collated["values"].tolist() == [[1.0, 2.0, 0.0, 0.0]]
    assert collated["rel_x_coords"].tolist() == [[0.0, 0.0, INF, INF]]
    assert collated["rel_y_coords"].tolist() == [[0.0, 0.0, INF, INF]]


def test_segment_cutoff_legacy_and_asymmetric():
    legacy = _collator()
    asym = BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=8,
                             seq_len_neighborhood=6, seq_len_neigh=3, n_special_tokens=1)

    assert [legacy.segment_cutoff(k) for k in (1, 2, 3)] == [4, 8, 12]
    assert [asym.segment_cutoff(k) for k in (1, 2, 3)] == [9, 12, 15]


def test_asymmetric_cell_only_sampling_zeroes_neighbours():
    coll = BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=8,
                             seq_len_neighborhood=6, seq_len_neigh=3, n_special_tokens=0,
                             sample_segments=True, cell_segment_sampling_ratio=1.0,
                             sample_gene_masks=False)
    tokens = torch.arange(1, 15).unsqueeze(0)
    batch = [{"tokens": tokens[0].clone(), "segments": torch.tensor([1] * 8 + [2] * 3 + [3] * 3),
              "positions": torch.tensor(list(range(1, 9)) + [1, 2, 3] * 2)}]

    collated, ctx, tgt, attn, _ = coll(batch)

    assert collated["tokens"][0, :8].tolist() == list(range(1, 9))
    assert collated["tokens"][0, 8:].tolist() == [0] * 6
    assert collated["segments"][0, 8:].tolist() == [0] * 6
    assert ctx is None and tgt is None
    assert attn[0, 0, 0].tolist() == [True] * 8 + [False] * 6

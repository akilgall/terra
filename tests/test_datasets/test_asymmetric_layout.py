"""Asymmetric gene budgets in ``CellGraphDataset``.

The pretrained layout gives every segment ``seq_len_cell`` slots. The
asymmetric layout (``seq_len_neigh`` set) gives the centre cell
``seq_len_cell`` slots and each neighbour ``seq_len_neigh`` slots, so a
whole-transcriptome centre cell can keep e.g. 2048 genes while neighbours keep
256. Cell-only records (Tahoe-100M cells, which have no neighbours) are the
same layout with every neighbour segment empty.

The ``GOLDEN_*`` literals were produced by the unmodified base code (commit
4bf0ad6) and pin legacy behaviour exactly: when ``seq_len_neigh`` is absent,
items must stay byte-identical.
"""

import hashlib
import os

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset, init_cell_dataset  # noqa: E402

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

GOLDEN_COMBINED_COORD = [
    {
        "tokens": [5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0],
        "values": [3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        "rel_x_coords": [0.0, 0.0, 0.0, INF, 1.5, 1.5, INF, INF, -2.0, INF, INF, INF],
        "rel_y_coords": [0.0, 0.0, 0.0, INF, 0.5, 0.5, INF, INF, 3.0, INF, INF, INF],
        "positions": [1, 2, 3, 0, 1, 2, 0, 0, 1, 0, 0, 0],
        "segments": [1, 1, 1, 0, 2, 2, 0, 0, 3, 0, 0, 0],
        "cell_id": "c0",
    },
    {
        "tokens": [11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0],
        "values": [1.0, 0.9, 0.8, 0.7, 2.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rel_x_coords": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, INF, INF, INF, INF, INF, INF],
        "rel_y_coords": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, INF, INF, INF, INF, INF, INF],
        "positions": [1, 2, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0],
        "segments": [1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0],
        "cell_id": "c1",
    },
]

GOLDEN_RANK_SEGMENT = [
    {"tokens": [5, 6, 7, 0, 8, 9, 0, 0, 10, 0, 0, 0],
     "positions": [1, 2, 3, 0, 1, 2, 0, 0, 1, 0, 0, 0],
     "segments": [1, 1, 1, 0, 2, 2, 0, 0, 3, 0, 0, 0], "cell_id": "c0"},
    {"tokens": [11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0],
     "positions": [1, 2, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0],
     "segments": [1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0], "cell_id": "c1"},
]

# sha256 over the first 64 CosMx NSCLC items with the TERRA-96M config, base code.
GOLDEN_COSMX_FIRST64_SHA256 = "f2bd652116fcdd078298aeff6f968464841792bcfa4da99a9afedc266171e51f"


def _hf(rows: dict) -> "datasets.Dataset":
    tensor_cols = [c for c in rows if c != "cell_id"]
    return datasets.Dataset.from_dict(rows).with_format(
        "torch", columns=tensor_cols, output_all_columns=True)


def _dataset(rows, gt_type="combined", cell_pos_enc="coord", seq_len_cell=4,
             seq_len_neighborhood=8, **kwargs) -> CellGraphDataset:
    return CellGraphDataset(
        gt_type=gt_type, cell_pos_enc=cell_pos_enc, dataset=_hf(rows), vocab_size=20,
        seq_len_cell=seq_len_cell, seq_len_neighborhood=seq_len_neighborhood,
        special_tokens=[], sampling_strategy=None, include_cell_id=True, **kwargs)


def _assert_item(item: dict, expected: dict) -> None:
    assert set(item) == set(expected)
    for key, want in expected.items():
        got = item[key]
        if key == "cell_id":
            assert got == want
            continue
        want_t = torch.tensor(want, dtype=got.dtype)
        assert torch.equal(got, want_t), f"{key}: {got.tolist()} != {want}"


# ---------------------------------------------------------------------------
# Legacy layout stays byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idx", [0, 1])
def test_legacy_combined_coord_item_matches_golden(idx):
    _assert_item(_dataset(LEGACY_ROWS)[idx], GOLDEN_COMBINED_COORD[idx])


@pytest.mark.parametrize("idx", [0, 1])
def test_legacy_rank_segment_item_matches_golden(idx):
    ds = _dataset(LEGACY_ROWS, gt_type="rank", cell_pos_enc="segment")
    _assert_item(ds[idx], GOLDEN_RANK_SEGMENT[idx])


@pytest.mark.parametrize("idx", [0, 1])
def test_explicit_uniform_layout_matches_legacy(idx):
    """Setting seq_len_neigh == seq_len_cell takes the layout code path and
    must reproduce the legacy items exactly."""
    ds = _dataset(LEGACY_ROWS, seq_len_neigh=4)
    _assert_item(ds[idx], GOLDEN_COMBINED_COORD[idx])


def test_legacy_n_segments_attribute_unchanged():
    assert _dataset(LEGACY_ROWS).n_segments == 3.0


# ---------------------------------------------------------------------------
# Asymmetric layout
# ---------------------------------------------------------------------------

ASYM_ROWS = {
    "gene_tokens": [[5, 6, 7, 0, 8, 9, 10, 0], [11, 12, 13, 14, 0, 0, 15, 16]],
    "gene_expr": [[3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 4.0, 0.0],
                  [1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 2.0, 1.0]],
    "rel_x_coord": [[0.0, 1.5, -2.0], [0.0, 1.0, 2.0]],
    "rel_y_coord": [[0.0, 0.5, 3.0], [0.0, -1.0, 1.0]],
    "n_nonzero_tokens": [6, 6],
    "cell_id": ["a0", "a1"],
}


def _asym(rows=ASYM_ROWS, **kwargs) -> CellGraphDataset:
    """Centre 4 slots, two neighbours of 2 slots: [4][2][2]."""
    return _dataset(rows, seq_len_cell=4, seq_len_neighborhood=4, seq_len_neigh=2, **kwargs)


def test_asymmetric_round_trip():
    _assert_item(_asym()[0], {
        "tokens": [5, 6, 7, 0, 8, 9, 10, 0],
        "values": [3.0, 2.0, 1.5, 0.0, 2.5, 1.0, 4.0, 0.0],
        "rel_x_coords": [0.0, 0.0, 0.0, INF, 1.5, 1.5, -2.0, INF],
        "rel_y_coords": [0.0, 0.0, 0.0, INF, 0.5, 0.5, 3.0, INF],
        "positions": [1, 2, 3, 0, 1, 2, 1, 0],
        "segments": [1, 1, 1, 0, 2, 2, 3, 0],
        "cell_id": "a0",
    })


def test_asymmetric_empty_neighbour_packs_like_legacy():
    _assert_item(_asym()[1], {
        "tokens": [11, 12, 13, 14, 15, 16, 0, 0],
        "values": [1.0, 0.5, 0.25, 0.125, 2.0, 1.0, 0.0, 0.0],
        "rel_x_coords": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, INF, INF],
        "rel_y_coords": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, INF, INF],
        "positions": [1, 2, 3, 4, 1, 2, 0, 0],
        "segments": [1, 1, 1, 1, 3, 3, 0, 0],
        "cell_id": "a1",
    })


def test_asymmetric_n_segments_is_int():
    assert _asym().n_segments == 3


CELL_ONLY_EXPECTED = {
    "tokens": [5, 6, 7, 0, 0, 0, 0, 0],
    "values": [3.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    "rel_x_coords": [0.0, 0.0, 0.0, INF, INF, INF, INF, INF],
    "rel_y_coords": [0.0, 0.0, 0.0, INF, INF, INF, INF, INF],
    "positions": [1, 2, 3, 0, 0, 0, 0, 0],
    "segments": [1, 1, 1, 0, 0, 0, 0, 0],
    "cell_id": "t0",
}


@pytest.mark.parametrize("stored_tokens,stored_expr", [
    ([5, 6, 7, 0, 0, 0, 0, 0], [3.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),  # full length
    ([5, 6, 7, 0], [3.0, 2.0, 1.5, 0.0]),  # centre only
])
def test_cell_only_record_with_single_coordinate(stored_tokens, stored_expr):
    rows = {"gene_tokens": [stored_tokens], "gene_expr": [stored_expr],
            "rel_x_coord": [[0.0]], "rel_y_coord": [[0.0]],
            "n_nonzero_tokens": [3], "cell_id": ["t0"]}

    _assert_item(_asym(rows)[0], CELL_ONLY_EXPECTED)


def _sentinel_rows(lengths: list[int]) -> dict:
    """Token at (segment s, rank r) = 100*s + r + 1 (segments 0-indexed)."""
    tokens = [100 * s + r + 1 for s, n in enumerate(lengths) for r in range(n)]
    return {"gene_tokens": [tokens], "gene_expr": [[float(t) + 1000 for t in tokens]],
            "rel_x_coord": [[float(s) for s in range(len(lengths))]],
            "rel_y_coord": [[0.0] * len(lengths)],
            "n_nonzero_tokens": [len(tokens)], "cell_id": ["s0"]}


def test_per_segment_load_time_truncation():
    """Stored [10][4][4] -> keep [8][3][3]."""
    ds = _dataset(_sentinel_rows([10, 4, 4]), seq_len_cell=8, seq_len_neighborhood=6,
                  seq_len_neigh=3, tokenized_seq_len_cell=10, tokenized_seq_len_neigh=4)

    item = ds[0]

    expected = list(range(1, 9)) + [101, 102, 103] + [201, 202, 203]
    assert item["tokens"].tolist() == expected
    assert item["values"].tolist() == [float(t) + 1000 for t in expected]
    assert item["segments"].tolist() == [1] * 8 + [2] * 3 + [3] * 3
    assert item["positions"].tolist() == list(range(1, 9)) + [1, 2, 3] * 2
    assert item["rel_x_coords"].tolist() == [0.0] * 8 + [1.0] * 3 + [2.0] * 3


def test_neighbour_truncation_with_asymmetric_layout():
    """Stored [4][2][2][2] -> keep centre + first 2 neighbours."""
    ds = _dataset(_sentinel_rows([4, 2, 2, 2]), seq_len_cell=4, seq_len_neighborhood=4,
                  seq_len_neigh=2, truncate_neighbors=True)

    assert ds[0]["tokens"].tolist() == [1, 2, 3, 4, 101, 102, 201, 202]


def test_truncation_rejects_growth():
    ds = _dataset(_sentinel_rows([4, 2, 2]), seq_len_cell=4, seq_len_neighborhood=6,
                  seq_len_neigh=3, tokenized_seq_len_neigh=2)

    with pytest.raises(ValueError, match="seq_len_neigh"):
        ds[0]


def test_panel_columns_give_index_aligned_view():
    rows = dict(ASYM_ROWS)
    rows = {k: [v[0]] for k, v in rows.items()}
    rows["gene_tokens_panel"] = [[6, 8]]
    rows["gene_expr_panel"] = [[2.0, 2.5]]
    rows["n_nonzero_tokens_panel"] = [2]
    rows["rel_x_coord"] = [[0.0]]
    rows["rel_y_coord"] = [[0.0]]

    ds = _asym(rows, token_column="gene_tokens_panel", expr_column="gene_expr_panel",
               nnz_column="n_nonzero_tokens_panel")
    item = ds[0]

    assert item["tokens"].tolist() == [6, 8, 0, 0, 0, 0, 0, 0]
    assert item["values"].tolist() == [2.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert item["segments"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert ds.n_nz_tokens == [2]


def test_centre_position_cap_clamps_centre_only():
    item = _asym(centre_position_cap=2)[1]

    assert item["positions"].tolist() == [1, 2, 2, 2, 1, 2, 0, 0]


def test_init_cell_dataset_forwards_layout_kwargs():
    ds = init_cell_dataset(
        tokenizer_type="cell_graph", gt_type="combined", cell_pos_enc="coord",
        dataset=_hf(ASYM_ROWS), vocab_size=20, seq_len_cell=4, seq_len_neighborhood=4,
        seq_len_neigh=2, special_tokens=[], sampling_strategy=None, include_cell_id=True)

    assert ds.layout.total == 8
    assert ds[0]["tokens"].tolist() == [5, 6, 7, 0, 8, 9, 10, 0]


# ---------------------------------------------------------------------------
# Opt-in regression on the real CosMx NSCLC dataset (TERRA-96M config)
# ---------------------------------------------------------------------------

def _hash_items(cell_dataset, n: int) -> str:
    h = hashlib.sha256()
    for i in range(n):
        item = cell_dataset[i]
        for key in sorted(item):
            value = item[key]
            h.update(key.encode())
            h.update(value.numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
    return h.hexdigest()


@pytest.mark.skipif(
    not (os.environ.get("TERRA_GOLDEN_DATASET") and os.environ.get("TERRA_BUNDLE_DIR")),
    reason="set TERRA_GOLDEN_DATASET and TERRA_BUNDLE_DIR to run the CosMx golden check")
def test_cosmx_first_64_items_unchanged():
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(os.environ["TERRA_BUNDLE_DIR"], "model_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    data, meta = cfg["data"], cfg["meta"]
    ds = datasets.load_from_disk(os.environ["TERRA_GOLDEN_DATASET"]).select(range(64))
    cell_dataset = init_cell_dataset(
        dataset=ds, vocab_size=23407, seq_len_cell=data["seq_len_cell"],
        seq_len_neighborhood=data["seq_len_neighborhood"], tokenizer_type=data["tokenizer_type"],
        gt_type=meta["gt_type"], cell_pos_enc=meta["cell_pos_enc"],
        special_tokens=meta["special_tokens"], sampling_strategy=None, n_nonzero_tokens_list=[],
        include_cell_id=True, sep_gene_tokens_neb=data["sep_gene_tokens_neb"],
        pad_special_tokens=True)

    assert _hash_items(cell_dataset, 64) == GOLDEN_COSMX_FIRST64_SHA256

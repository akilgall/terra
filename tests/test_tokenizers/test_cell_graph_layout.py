"""``CellGraphTokenizer`` with asymmetric gene budgets and a cell-only mode.

Legacy records give every segment ``model_input_size / (1 + non-empty
neighbours)`` slots. The asymmetric layout gives the centre cell
``seq_len_cell`` slots and each neighbour ``seq_len_neigh`` fixed slots (an
empty neighbour becomes an all-padding slot, later slots do not move).
Cell-only mode (Tahoe-100M cells, which have no coordinates) skips the spatial
graph and stores only the centre segment.

``LEGACY_SHA256`` was produced by the unmodified upstream tokenizer (main, 6cb235f).
"""

import hashlib
import pickle

import anndata as ad
import numpy as np
import pytest
import yaml
from tiny_terra import skip_without_squidpy

from terra.inference.tokenize import tokenize_adata
from terra.tokenizers.cell_tokenizers import CellGraphTokenizer

N_CELLS, N_GENES = 10, 8
LEGACY_SHA256 = "d253542d601d0f1813819b156c4f6ae616baac545cb6d7c7186b457867752c57"
NORMS = {"rank_cell_norm_method": None, "rank_gene_norm_method": None,
         "rank_count_norm_method": None, "count_cell_norm_method": None,
         "count_gene_norm_method": None, "count_count_norm_method": "shifted_log"}


def _adata() -> ad.AnnData:
    # Distinct counts within each cell: the tokenizer orders genes with an
    # unstable argsort, whose order for tied counts differs between platforms
    # (e.g. x86 Linux and Apple Silicon), so ties would make the digest
    # platform-dependent.
    rng = np.random.default_rng(42)
    counts = np.stack([rng.permutation(N_GENES) for _ in range(N_CELLS)]).astype(np.float32)
    counts[3, :] = 0.0
    counts[3, 0] = 1.0  # sparse cell
    genes = [f"ENSG{idx:011d}" for idx in range(1, N_GENES + 1)]
    adata = ad.AnnData(X=counts)
    adata.var_names = genes
    adata.var["ensembl_id"] = genes
    adata.obs_names = [f"cell{i}" for i in range(N_CELLS)]
    adata.obs["cell_id"] = adata.obs_names.to_list()
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(N_CELLS, 2)).astype(np.float32)
    return adata


@pytest.fixture
def token_path(tmp_path) -> str:
    token_dict = {"<pad>": 0, "<mask>": 1}
    token_dict.update({f"ENSG{idx:011d}": idx + 1 for idx in range(1, N_GENES + 1)})
    path = tmp_path / "token_dictionary.pkl"
    with open(path, "wb") as f:
        pickle.dump(token_dict, f)
    return str(path)


def _tokenize(adata, token_path, model_input_size=12, n_neighs=2, **kwargs):
    if not kwargs.get("cell_only"):
        skip_without_squidpy()
    tk = CellGraphTokenizer(
        nproc=1, processing_mode="sequential", model_input_size=model_input_size,
        n_neighs=n_neighs, token_dictionary_file_path=token_path,
        include_special_tokens=False, **NORMS, **kwargs)
    return tk._create_dataset(tk._tokenize_adata(adata=adata), use_generator=False)


def _digest(ds) -> str:
    # Expression values are float32 log1p of the counts 1..7, whose last bits
    # depend on the platform's log1p; at 3 decimals each lies >5e-5 from a
    # rounding boundary.
    h = hashlib.sha256()
    for row in ds:
        h.update(np.asarray(row["gene_tokens"], dtype=np.int64).tobytes())
        h.update(np.round(np.asarray(row["gene_expr"], dtype=np.float64), 3).tobytes())
        h.update(np.asarray(row["rel_x_coord"], dtype=np.float64).tobytes())
        h.update(np.asarray(row["rel_y_coord"], dtype=np.float64).tobytes())
        h.update(str(row["n_nonzero_tokens"]).encode())
    return h.hexdigest()


def _nearest(adata, i: int, k: int) -> list[int]:
    coords = adata.obsm["spatial"].astype(np.float64)
    dist = np.linalg.norm(coords - coords[i], axis=1)
    dist[i] = np.inf
    return list(np.argsort(dist, kind="stable")[:k])


# ---------------------------------------------------------------------------
# Legacy layout is unchanged
# ---------------------------------------------------------------------------

def test_legacy_output_matches_golden(token_path):
    assert _digest(_tokenize(_adata(), token_path)) == LEGACY_SHA256


def test_degenerate_asymmetric_equals_legacy(token_path):
    """Centre and neighbour budgets both 4 with all neighbours non-empty is the
    legacy 3 x 4 layout."""
    ds = _tokenize(_adata(), token_path, seq_len_cell=4, seq_len_neigh=4)

    assert _digest(ds) == LEGACY_SHA256


# ---------------------------------------------------------------------------
# Asymmetric layout
# ---------------------------------------------------------------------------

def _cell_only_tokens(token_path, adata, budget):
    ds = _tokenize(adata, token_path, model_input_size=budget, n_neighs=2,
                   seq_len_cell=budget, seq_len_neigh=3, cell_only=True)
    return [list(row["gene_tokens"]) for row in ds]


def test_asymmetric_centre_and_neighbour_blocks(token_path):
    adata = _adata()
    ds = _tokenize(adata, token_path, model_input_size=14, seq_len_cell=8, seq_len_neigh=3)
    centre = _cell_only_tokens(token_path, adata, budget=8)

    for i, row in enumerate(ds):
        tokens = list(row["gene_tokens"])
        assert len(tokens) == 14
        assert tokens[:8] == centre[i]
        for slot, neighbour in enumerate(_nearest(adata, i, 2)):
            start = 8 + 3 * slot
            assert tokens[start:start + 3] == centre[neighbour][:3]
        assert row["n_nonzero_tokens"] == sum(t != 0 for t in tokens)


def test_empty_neighbour_keeps_fixed_slot(token_path):
    adata = _adata()
    host = 0
    empty, other = _nearest(adata, host, 2)
    adata.X[empty, :] = 0.0  # the host's nearest neighbour now expresses nothing
    ds = _tokenize(adata, token_path, model_input_size=14, seq_len_cell=8, seq_len_neigh=3)
    centre = _cell_only_tokens(token_path, adata, budget=8)
    slot, other_slot = 0, 1

    tokens = list(ds[host]["gene_tokens"])

    assert tokens[8 + 3 * slot: 11 + 3 * slot] == [0, 0, 0]
    assert tokens[8 + 3 * other_slot: 11 + 3 * other_slot] == centre[other][:3]


def test_cell_only_without_coordinates(token_path):
    adata = _adata()
    reference = _cell_only_tokens(token_path, adata, budget=8)
    del adata.obsm["spatial"]

    ds = _tokenize(adata, token_path, model_input_size=14, seq_len_cell=8,
                   seq_len_neigh=3, cell_only=True)

    for i, row in enumerate(ds):
        assert list(row["gene_tokens"]) == reference[i]
        assert row["rel_x_coord"] == [0.0]
        assert row["rel_y_coord"] == [0.0]
        assert row["n_nonzero_tokens"] == min(int((adata.X[i] > 0).sum()), 8)


def test_cell_only_records_neighbour_free_cell_ids(token_path):
    adata = _adata()
    del adata.obsm["spatial"]
    tk = CellGraphTokenizer(
        nproc=1, model_input_size=14, n_neighs=2, token_dictionary_file_path=token_path,
        include_special_tokens=False, seq_len_cell=8, seq_len_neigh=3, cell_only=True,
        add_neigh_cell_ids=True, **NORMS)

    records = tk._tokenize_adata(adata=adata)

    assert records["cell_ids"][0] == ["cell0"] * 8


@pytest.mark.parametrize("kwargs, match", [
    ({"model_input_size": 15, "seq_len_cell": 8, "seq_len_neigh": 3}, "model_input_size"),
    ({"seq_len_neigh": 3}, "seq_len_cell"),
    ({"n_neighs": None, "seq_len_cell": 8, "seq_len_neigh": 3}, "n_neighs"),
    ({"model_input_size": 6, "seq_len_cell": 8, "seq_len_neigh": 3, "cell_only": True},
     "model_input_size"),
])
def test_invalid_budgets_raise(token_path, kwargs, match):
    params = {"model_input_size": 14, "n_neighs": 2}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CellGraphTokenizer(nproc=1, token_dictionary_file_path=token_path,
                           include_special_tokens=False, **NORMS, **params)


# ---------------------------------------------------------------------------
# terra.inference.tokenize_adata plumbing
# ---------------------------------------------------------------------------

def _model_folder(tmp_path, token_path, **data_overrides) -> str:
    data = {"tokenizer_type": "cell_graph", "model_input_size": 14, "n_neighs": 2,
            "seq_len_cell": 8, "seq_len_neigh": 3, "seq_len_neighborhood": 6, **NORMS}
    data.update(data_overrides)
    folder = tmp_path / "model"
    folder.mkdir()
    with open(folder / "model_config.yaml", "w") as f:
        yaml.safe_dump({"data": data}, f)
    with open(token_path, "rb") as src, open(folder / "token_dictionary.pkl", "wb") as dst:
        dst.write(src.read())
    return str(folder)


def test_tokenize_adata_uses_config_budgets(tmp_path, token_path):
    skip_without_squidpy()
    folder = _model_folder(tmp_path, token_path)

    ds = tokenize_adata(_adata(), folder, str(tmp_path / "cache"), nproc=1,
                        processing_mode="sequential", use_generator=False)

    assert ds[0]["gene_tokens"].shape[0] == 14


def test_tokenize_adata_auto_detects_cell_only(tmp_path, token_path):
    folder = _model_folder(tmp_path, token_path)
    adata = _adata()
    del adata.obsm["spatial"]

    ds = tokenize_adata(adata, folder, str(tmp_path / "cache"), nproc=1,
                        processing_mode="sequential", use_generator=False)

    assert ds[0]["gene_tokens"].shape[0] == 8
    assert ds[0]["rel_x_coord"].tolist() == [0.0]


def test_tokenize_adata_explicit_cell_only_overrides_coordinates(tmp_path, token_path):
    folder = _model_folder(tmp_path, token_path)

    ds = tokenize_adata(_adata(), folder, str(tmp_path / "cache"), nproc=1,
                        processing_mode="sequential", use_generator=False, cell_only=True)

    assert ds[0]["gene_tokens"].shape[0] == 8


# ---------------------------------------------------------------------------
# Records load into the layout-aware dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell_only", [False, True])
def test_records_load_into_asymmetric_dataset(token_path, cell_only):
    from terra.datasets.cell_datasets import CellGraphDataset

    adata = _adata()
    ds = _tokenize(adata, token_path, model_input_size=14, seq_len_cell=8,
                   seq_len_neigh=3, cell_only=cell_only)
    cell_ds = CellGraphDataset(
        gt_type="combined", cell_pos_enc="coord", dataset=ds.with_format("torch"),
        vocab_size=N_GENES + 2, seq_len_cell=8, seq_len_neighborhood=6, seq_len_neigh=3,
        special_tokens=[], sampling_strategy=None)

    item = cell_ds[0]
    record = list(ds[0]["gene_tokens"])

    assert item["tokens"].shape[0] == 14
    assert item["tokens"][:8].tolist() == record[:8]
    if cell_only:
        assert item["tokens"][8:].tolist() == [0] * 6
        assert set(item["segments"].tolist()) <= {0, 1}
    else:
        assert set(item["segments"][8:].tolist()) - {0} == {2, 3}

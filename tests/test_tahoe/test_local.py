"""``terra.tahoe.local``: a local Tahoe-100M download (offline).

The fixtures write a tiny copy in both layouts a full download can have:
parquet shards (``data/train-*.parquet`` + ``metadata/*.parquet``, as from
``snapshot_download``) and the Arrow cache that ``datasets.load_dataset``
builds from them.
"""

import itertools

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from terra.tahoe.convert import cell_line_table, gene_lookup, sample_table, table_to_anndata
from terra.tahoe.local import (
    cell_counts,
    cells_filter,
    find_expression_files,
    metadata_file,
    open_expression_data,
    read_cells,
    read_obs,
    sample_per_group,
    select_cell_lines,
    select_samples,
)
from terra.tahoe.stream import REPO_ID

GENE_METADATA = pd.DataFrame({"gene_symbol": ["G1", "G2", "G3"],
                              "ensembl_id": ["ENSG01", "ENSG02", "ENSG03"],
                              "token_id": [3, 4, 5]})
SAMPLE_METADATA = pd.DataFrame({
    "sample": ["smp_1", "smp_2", "smp_3", "smp_4", "smp_5", "smp_6"],
    "plate": ["plate1"] * 3 + ["plate2"] * 3,
    "drug": ["Trametinib", "Paclitaxel", "DMSO_TF", "Trametinib", "Erdafitinib ", "DMSO_TF"],
    "drugname_drugconc": ["[('Trametinib', 5.0, 'uM')]", "[('Paclitaxel', 5.0, 'uM')]",
                          "[('DMSO_TF', 0.0, 'uM')]", "[('Trametinib', 0.5, 'uM')]",
                          "[('Erdafitinib ', 0.5, 'uM')]", "[('DMSO_TF', 0.0, 'uM')]"],
})
CELL_LINE_METADATA = pd.DataFrame({
    "cell_name": ["A549", "HT-29", "NCI-H358", "NCI-H358"],
    "Cell_ID_Cellosaur": ["CVCL_0023", "CVCL_0320", "CVCL_1559", "CVCL_1559"],
    "Cell_ID_DepMap": ["ACH-000681", "ACH-000552", "ACH-000684", "ACH-000684"],
    "Organ": ["Lung", "Bowel", "Lung", "Lung"],
})
CELL_LINES = ["CVCL_0023", "CVCL_0320", "CVCL_1559"]
CELLS_PER_GROUP = 3


def _records(plate: str) -> list[dict]:
    rows = SAMPLE_METADATA[SAMPLE_METADATA["plate"] == plate]
    records = []
    for (sample, drug), line, k in itertools.product(zip(rows["sample"], rows["drug"], strict=True),
                                                      CELL_LINES, range(CELLS_PER_GROUP)):
        records.append({"genes": [1, 3, 5], "expressions": [-2.0, float(k + 1), 1.0],
                        "drug": drug, "sample": sample,
                        "BARCODE_SUB_LIB_ID": f"{sample}_{line}_{k}",
                        "cell_line_id": line, "moa-fine": "x", "canonical_smiles": "C",
                        "pubchem_cid": "1.0", "plate": plate})
    return records


N_CELLS = len(SAMPLE_METADATA) * len(CELL_LINES) * CELLS_PER_GROUP


def _write_snapshot(root):
    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()
    for i, plate in enumerate(["plate1", "plate2"]):
        pq.write_table(pa.Table.from_pylist(_records(plate)),
                       root / "data" / f"train-0000{i}-of-00002.parquet", row_group_size=9)
    for name, table in [("gene_metadata", GENE_METADATA), ("sample_metadata", SAMPLE_METADATA),
                        ("cell_line_metadata", CELL_LINE_METADATA)]:
        table.to_parquet(root / "metadata" / f"{name}.parquet")
    return root


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return _write_snapshot(tmp_path_factory.mktemp("snapshot"))


@pytest.fixture(scope="module")
def arrow_cache(tmp_path_factory, snapshot):
    datasets = pytest.importorskip("datasets")
    cache = tmp_path_factory.mktemp("hf_cache")
    datasets.load_dataset("parquet", data_files=sorted(map(str, (snapshot / "data").iterdir())),
                          split="train", cache_dir=str(cache))
    return cache


@pytest.fixture(params=["parquet", "arrow"])
def layout(request, snapshot):
    """(root, expected kind) for each layout."""
    root = snapshot if request.param == "parquet" else request.getfixturevalue("arrow_cache")
    return root, request.param


# ---------------------------------------------------------------------------
# Locating and opening the data
# ---------------------------------------------------------------------------

def test_find_expression_files_in_both_layouts(layout):
    root, kind = layout

    found_kind, files = find_expression_files(root)

    assert found_kind == kind
    assert all(f.suffix == f".{kind}" for f in files)
    if kind == "parquet":
        assert [f.name for f in files] == ["train-00000-of-00002.parquet",
                                           "train-00001-of-00002.parquet"]


def test_find_expression_files_rejects_two_copies(tmp_path, snapshot):
    for copy in ["a", "b"]:
        (tmp_path / copy).mkdir()
        pq.write_table(pq.read_table(snapshot / "data" / "train-00000-of-00002.parquet"),
                       tmp_path / copy / "train.parquet")

    with pytest.raises(ValueError, match="several directories"):
        find_expression_files(tmp_path)


def test_find_expression_files_without_data_raises(tmp_path):
    GENE_METADATA.to_parquet(tmp_path / "gene_metadata.parquet")

    with pytest.raises(FileNotFoundError, match="No Tahoe-100M expression data"):
        find_expression_files(tmp_path)


def test_layouts_open_to_the_same_cells(layout):
    dataset = open_expression_data(layout[0])

    barcodes = dataset.to_table(columns=["BARCODE_SUB_LIB_ID"])["BARCODE_SUB_LIB_ID"]

    assert dataset.count_rows() == N_CELLS
    assert sorted(barcodes.to_pylist()) == sorted(
        r["BARCODE_SUB_LIB_ID"] for r in _records("plate1") + _records("plate2"))


def test_metadata_file_finds_local_copies(tmp_path, snapshot):
    hub = tmp_path / "datasets--tahoebio--Tahoe-100M" / "snapshots" / "abc" / "metadata"
    hub.mkdir(parents=True)
    GENE_METADATA.to_parquet(hub / "gene_metadata.parquet")

    assert metadata_file("sample_metadata", snapshot) == \
        snapshot / "metadata" / "sample_metadata.parquet"
    assert metadata_file("gene_metadata", tmp_path) == hub / "gene_metadata.parquet"


def test_metadata_file_downloads_missing_tables(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    path = metadata_file("drug_metadata", tmp_path)

    assert path == tmp_path / "metadata" / "drug_metadata.parquet"
    assert calls == [(REPO_ID, "metadata/drug_metadata.parquet",
                      {"repo_type": "dataset", "cache_dir": str(tmp_path)})]


# ---------------------------------------------------------------------------
# Selecting wells and cell lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"drugs": ["Trametinib"], "doses_uM": [5.0]}, ["smp_1", "smp_3"]),
    ({"drugs": ["Trametinib"], "doses_uM": [5.0], "include_controls": False}, ["smp_1"]),
    ({"drugs": ["Trametinib"]}, ["smp_1", "smp_3", "smp_4", "smp_6"]),
    ({"drugs": ["  Erdafitinib"]}, ["smp_5", "smp_6"]),
    ({"plates": ["plate2"], "include_controls": False}, ["smp_4", "smp_5", "smp_6"]),
    ({"doses_uM": [0.5], "plates": ["plate2"]}, ["smp_4", "smp_5", "smp_6"]),
])
def test_select_samples(kwargs, expected):
    assert select_samples(SAMPLE_METADATA, **kwargs) == expected


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"drugs": ["Trametinb"]}, "Unknown drugs.*Trametinb"),
    ({"plates": ["plate9"]}, "Unknown plates"),
    ({"drugs": ["Paclitaxel"], "doses_uM": [0.5]}, "No samples match"),
])
def test_select_samples_rejects_impossible_selections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_samples(SAMPLE_METADATA, **kwargs)


@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"organs": ["Lung"]}, ["CVCL_0023", "CVCL_1559"]),
    ({"cell_names": ["HT-29"]}, ["CVCL_0320"]),
    ({"organs": ["Lung"], "cell_names": ["A549", "HT-29"]}, ["CVCL_0023"]),
    ({}, CELL_LINES),
])
def test_select_cell_lines(kwargs, expected):
    assert select_cell_lines(CELL_LINE_METADATA, **kwargs) == expected


def test_select_cell_lines_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown organs.*Lungs"):
        select_cell_lines(CELL_LINE_METADATA, organs=["Lungs"])


def test_cells_filter_is_none_without_criteria():
    assert cells_filter() is None


# ---------------------------------------------------------------------------
# Counting, reading metadata and reading cells
# ---------------------------------------------------------------------------

def _expected_counts(records, by):
    frame = pd.DataFrame(records)
    return frame.groupby(list(by)).size().rename("n_cells").reset_index()


def test_cell_counts_match_the_records(layout):
    by = ("plate", "sample", "cell_line_id")
    dataset = open_expression_data(layout[0])

    counts = cell_counts(dataset, by=by, batch_size=7)

    expected = _expected_counts(_records("plate1") + _records("plate2"), by)
    pd.testing.assert_frame_equal(counts, expected, check_dtype=False)


def test_cell_counts_with_filter(snapshot):
    dataset = open_expression_data(snapshot)

    counts = cell_counts(dataset, by=("sample",),
                         filter=cells_filter(samples=["smp_2"], cell_lines=["CVCL_0023"]))

    assert counts.to_dict("records") == [{"sample": "smp_2", "n_cells": CELLS_PER_GROUP}]


def test_read_obs_returns_categorical_metadata(layout):
    dataset = open_expression_data(layout[0])

    obs = read_obs(dataset, cells_filter(samples=["smp_5"]))

    assert len(obs) == len(CELL_LINES) * CELLS_PER_GROUP
    assert set(obs["drug"]) == {"Erdafitinib"}  # surrounding whitespace removed
    assert isinstance(obs["cell_line_id"].dtype, pd.CategoricalDtype)
    assert obs["BARCODE_SUB_LIB_ID"].is_unique
    assert not isinstance(obs["BARCODE_SUB_LIB_ID"].dtype, pd.CategoricalDtype)


def test_sample_per_group_caps_groups_and_keeps_order():
    obs = pd.DataFrame({"line": ["a"] * 5 + ["b"] * 2, "value": range(7)})

    picked = sample_per_group(obs, by=["line"], n=3, seed=1)

    assert picked["line"].value_counts().to_dict() == {"a": 3, "b": 2}
    assert picked.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(picked, sample_per_group(obs, by=["line"], n=3, seed=1))
    assert not picked.equals(sample_per_group(obs, by=["line"], n=3, seed=2))


def test_read_cells_reads_requested_cells_ready_for_anndata(layout):
    dataset = open_expression_data(layout[0])
    wanted = ["smp_1_CVCL_0023_0", "smp_6_CVCL_1559_2"]

    table = read_cells(dataset, wanted, filter=cells_filter(samples=["smp_1", "smp_6"]))
    lookup, var = gene_lookup(GENE_METADATA)
    adata = table_to_anndata(table, lookup, var, sample_table(SAMPLE_METADATA),
                             cell_line_table(CELL_LINE_METADATA))

    assert sorted(adata.obs["cell_id"]) == wanted
    assert adata.to_df().loc[wanted[1]].tolist() == [3, 0, 1]
    assert adata.obs.loc[wanted[0], "dose_uM"] == 5.0


def test_read_cells_warns_about_missing_cells(snapshot, caplog):
    dataset = open_expression_data(snapshot)

    with caplog.at_level("WARNING"):
        table = read_cells(dataset, ["smp_1_CVCL_0023_0", "no_such_cell"])

    assert table.num_rows == 1
    assert "1 of 2 requested cells were not found" in caplog.text


def test_sample_filters_skip_parquet_row_groups(snapshot):
    """Row-group statistics let a sample filter skip most of the data."""
    dataset = open_expression_data(snapshot)
    expression = cells_filter(samples=["smp_1"])

    kept = {f.path: f.split_by_row_group(filter=expression) for f in dataset.get_fragments()}

    plate1, plate2 = sorted(kept)
    assert kept[plate2] == []
    rows = sum(group.row_groups[0].num_rows for group in kept[plate1])
    first_shard = pq.read_table(plate1)
    n_smp_1 = int(pc.sum(pc.equal(first_shard["sample"], "smp_1")).as_py())
    assert n_smp_1 <= rows < first_shard.num_rows

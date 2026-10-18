"""``terra.tahoe.labels``: shared label codes and per-cell ``*_value`` columns."""

import pandas as pd
import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.tahoe.labels import (  # noqa: E402
    DOMAIN_SPATIAL,
    DOMAIN_TAHOE,
    UNMAPPED,
    LabelCodes,
    add_value_columns,
    lineage_of,
    load_lineage_map,
)

LINEAGE_MAP = {"lineages": {"lung": {"tahoe_organs": ["Lung"],
                                     "spatial_cell_types": ["tumor", "tumour"]}}}

TAHOE_OBS = pd.DataFrame({
    "cell_id": ["t0", "t1", "t2"],
    "organ": ["Lung", "Bowel", "Lung"],
    "cell_line_id": ["CVCL_0023", "CVCL_0320", "CVCL_0023"],
    "drug": ["Trametinib", "DMSO_TF", "DMSO_TF"],
    "dose_uM": [0.5, 0.0, 0.0],
    "is_control": [False, True, True],
})
SPATIAL_OBS = pd.DataFrame({"cell_id": ["s0", "s1"], "cell_type": ["tumor", "fibroblast"]})


def _codes() -> LabelCodes:
    return LabelCodes.from_values(LINEAGE_MAP, cell_lines=TAHOE_OBS["cell_line_id"],
                                  drugs=TAHOE_OBS["drug"])


def test_lineage_lookup():
    assert lineage_of("Lung", "tahoe_organs", LINEAGE_MAP) == "lung"
    assert lineage_of("tumour", "spatial_cell_types", LINEAGE_MAP) == "lung"
    assert lineage_of("fibroblast", "spatial_cell_types", LINEAGE_MAP) is None


def test_codes_are_deterministic_and_round_trip(tmp_path):
    codes = _codes()
    path = tmp_path / "label_codes.json"

    codes.save(path)

    assert codes.lineage == {"lung": 0}
    assert codes.cell_line == {"CVCL_0023": 0, "CVCL_0320": 1}
    assert codes.drug == {"DMSO_TF": 0, "Trametinib": 1}
    assert LabelCodes.load(path) == codes


def test_load_lineage_map(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("lineages:\n  lung:\n    tahoe_organs: [Lung]\n"
                    "    spatial_cell_types: [tumor]\n")

    assert load_lineage_map(path)["lineages"]["lung"]["tahoe_organs"] == ["Lung"]


def _dataset(cell_ids):
    return datasets.Dataset.from_dict({
        "cell_id": cell_ids,
        "gene_tokens": [[2, 3, 0, 0]] * len(cell_ids),
        "gene_expr": [[1.0, 0.5, 0.0, 0.0]] * len(cell_ids),
        "rel_x_coord": [[0.0]] * len(cell_ids),
        "rel_y_coord": [[0.0]] * len(cell_ids),
        "n_nonzero_tokens": [2] * len(cell_ids),
    })


def test_tahoe_value_columns_follow_dataset_order():
    ds = _dataset(["t2", "t0", "t1"])  # dataset order differs from obs order

    out = add_value_columns(ds, TAHOE_OBS, _codes(), LINEAGE_MAP, domain=DOMAIN_TAHOE,
                            lineage_column="organ", lineage_source="tahoe_organs")

    assert out["domain_value"] == [DOMAIN_TAHOE] * 3
    assert out["lineage_value"] == [0, 0, UNMAPPED]
    assert out["cell_line_value"] == [0, 0, 1]
    assert out["drug_value"] == [0, 1, 0]
    assert out["dose_value"] == [0, 500, 0]
    assert out["is_control_value"] == [1, 0, 1]
    assert "domain_value" not in ds.column_names  # input untouched


def test_spatial_value_columns_use_defaults():
    out = add_value_columns(_dataset(["s0", "s1"]), SPATIAL_OBS, _codes(), LINEAGE_MAP,
                            domain=DOMAIN_SPATIAL, lineage_column="cell_type",
                            lineage_source="spatial_cell_types")

    assert out["domain_value"] == [DOMAIN_SPATIAL] * 2
    assert out["lineage_value"] == [0, UNMAPPED]
    assert out["drug_value"] == [UNMAPPED] * 2
    assert out["is_control_value"] == [0, 0]


def test_missing_cells_raise():
    with pytest.raises(ValueError, match="cell_id"):
        add_value_columns(_dataset(["t0", "zz"]), TAHOE_OBS, _codes(), LINEAGE_MAP,
                          domain=DOMAIN_TAHOE, lineage_column="organ",
                          lineage_source="tahoe_organs")


def test_value_columns_reach_dataset_items_as_long():
    out = add_value_columns(_dataset(["t0", "t1"]), TAHOE_OBS, _codes(), LINEAGE_MAP,
                            domain=DOMAIN_TAHOE, lineage_column="organ",
                            lineage_source="tahoe_organs")
    cell_ds = CellGraphDataset(
        gt_type="combined", cell_pos_enc="coord", dataset=out.with_format("torch"),
        vocab_size=10, seq_len_cell=4, seq_len_neighborhood=4, seq_len_neigh=2,
        special_tokens=[], sampling_strategy=None)

    batch = torch.utils.data.default_collate([cell_ds[0], cell_ds[1]])

    assert batch["lineage_value"].tolist() == [0, UNMAPPED]
    assert batch["is_control_value"].dtype == torch.long

"""Pure helpers of the ``terra.tahoe.prepare`` CLI (offline)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from terra.tahoe.labels import LabelCodes
from terra.tahoe.prepare import balanced_subset, control_samples, parse_args, write_tahoe_dataset

SAMPLES = pd.DataFrame({
    "sample": ["smp_1", "smp_2", "smp_95", "smp_96", "smp_97"],
    "plate": ["plate1", "plate1", "plate1", "plate1", "plate2"],
    "drug": ["A", "B", "DMSO_TF", "DMSO_TF", "DMSO_TF"],
})


def _adata(n_lung_ctrl=4, n_lung_trt=6, n_bowel=5) -> ad.AnnData:
    organs = ["Lung"] * (n_lung_ctrl + n_lung_trt) + ["Bowel"] * n_bowel
    controls = [True] * n_lung_ctrl + [False] * n_lung_trt + [True] * n_bowel
    obs = pd.DataFrame({"organ": organs, "is_control": controls},
                       index=[f"c{i}" for i in range(len(organs))])
    return ad.AnnData(X=np.zeros((len(organs), 2), dtype=np.int32), obs=obs)


def test_control_samples_of_plate():
    assert control_samples(SAMPLES, "plate1") == {95, 96}


def test_balanced_subset_halves_controls_and_treated():
    subset = balanced_subset(_adata(), n_cells=6, organs=["Lung"], seed=0)

    assert subset.n_obs == 6
    assert set(subset.obs["organ"]) == {"Lung"}
    assert int(subset.obs["is_control"].sum()) == 3


def test_balanced_subset_fills_from_treated_when_controls_short():
    subset = balanced_subset(_adata(n_lung_ctrl=1), n_cells=6, organs=["Lung"], seed=0)

    assert int(subset.obs["is_control"].sum()) == 1
    assert subset.n_obs == 6


def test_balanced_subset_is_deterministic_and_keeps_order():
    a = balanced_subset(_adata(), n_cells=6, organs=["Lung"], seed=3)
    b = balanced_subset(_adata(), n_cells=6, organs=["Lung"], seed=3)

    assert a.obs_names.tolist() == b.obs_names.tolist()
    order = [int(name[1:]) for name in a.obs_names]
    assert order == sorted(order)


def test_parse_args_defaults():
    args = parse_args(["tahoe", "--out", "x", "--model-dir", "m", "--lineage-map", "l.yaml"])

    assert args.command == "tahoe"
    assert args.n_cells == 200
    assert args.organs == "Lung"


def test_write_tahoe_dataset_tokenizes_and_labels(tmp_path):
    datasets = pytest.importorskip("datasets")
    from tiny_terra import LINEAGE_MAP, make_adata, make_bundle

    bundle = make_bundle(tmp_path / "bundle")
    lineage_map = tmp_path / "lineage_map.yaml"
    lineage_map.write_text(yaml.safe_dump(LINEAGE_MAP))
    adata = make_adata(8, spatial=False, seed=0)

    path = write_tahoe_dataset(adata, bundle, lineage_map, tmp_path / "out",
                               {"seq_len_cell": 6, "seq_len_neigh": 3, "model_input_size": 12})

    dataset = datasets.load_from_disk(str(path)).with_format(None)
    assert path == tmp_path / "out" / "tahoe_ds"
    assert dataset.num_rows == 8
    assert len(dataset[0]["gene_tokens"]) == 6  # cell-only records: centre segment only
    assert list(dataset["is_control_value"]) == [1, 0] * 4
    assert set(dataset["lineage_value"]) == {0}
    assert LabelCodes.load(tmp_path / "out" / "label_codes.json").drug == {"DMSO_TF": 0, "Drug": 1}

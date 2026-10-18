"""Pure helpers of the ``terra.tahoe.prepare`` CLI (offline)."""

import anndata as ad
import numpy as np
import pandas as pd

from terra.tahoe.prepare import balanced_subset, control_samples, parse_args

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

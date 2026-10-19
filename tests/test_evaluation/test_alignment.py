"""Alignment metrics (``terra.evaluation.alignment``) on synthetic embeddings
with known answers."""

import numpy as np
import pandas as pd
import pytest

from terra.evaluation.alignment import (
    collapse_stats,
    domain_mixing,
    evaluate_alignment,
    label_retention,
    paired_view_similarity,
    perturbation_separability,
    perturbation_vector_consistency,
)

RNG = np.random.default_rng(0)
D = 8


def _gauss(n, shift=0.0, scale=1.0):
    return RNG.normal(loc=shift, scale=scale, size=(n, D))


def test_import_does_not_need_eval_extra():
    import terra.evaluation

    assert callable(terra.evaluation.alignment.evaluate_alignment)


def test_domain_mixing_high_when_mixed_low_when_separated():
    domain = np.repeat([0, 1], 100)
    mixed = domain_mixing(_gauss(200), domain, k=10)
    separated = domain_mixing(np.vstack([_gauss(100), _gauss(100, shift=10.0)]), domain, k=10)

    assert mixed["ratio"] > 0.8
    assert separated["ratio"] < 0.05
    assert mixed["expected"] == pytest.approx(100 / 199)


def test_domain_mixing_needs_two_domains():
    with pytest.raises(ValueError, match="two domains"):
        domain_mixing(_gauss(10), np.zeros(10, dtype=int))


def test_paired_view_similarity():
    full = _gauss(50)

    same = paired_view_similarity(full, full + 1e-3 * _gauss(50))
    unrelated = paired_view_similarity(full, _gauss(50))

    assert same["mean_cosine"] > 0.99 and same["top1_match"] == 1.0
    assert abs(unrelated["mean_cosine"]) < 0.2 and unrelated["top1_match"] < 0.2


def test_perturbation_separability_detects_shifted_drug():
    emb = np.vstack([_gauss(40), _gauss(40, shift=3.0), _gauss(40)])
    drug = np.array(["DMSO"] * 40 + ["A"] * 40 + ["B"] * 40)
    is_control = drug == "DMSO"

    auc = perturbation_separability(emb, drug, is_control, min_cells=10)

    assert auc["A"] > 0.95
    assert 0.3 < auc["B"] < 0.7


def test_perturbation_separability_skips_small_drugs():
    emb = _gauss(45)
    drug = np.array(["DMSO"] * 40 + ["rare"] * 5)

    assert perturbation_separability(emb, drug, drug == "DMSO", min_cells=10) == {}


def _lines_with_shift(direction_for_line):
    blocks, drug, line = [], [], []
    for name, direction in direction_for_line.items():
        blocks += [_gauss(30, scale=0.1), _gauss(30, scale=0.1) + direction]
        drug += ["DMSO"] * 30 + ["A"] * 30
        line += [name] * 60
    return np.vstack(blocks), np.array(drug), np.array(line)


def test_perturbation_vector_consistency():
    shift = np.full(D, 2.0)
    emb, drug, line = _lines_with_shift({"L1": shift, "L2": shift, "L3": shift})
    consistent = perturbation_vector_consistency(emb, drug, drug == "DMSO", line)

    opposite = {"L1": shift, "L2": -shift}
    emb, drug, line = _lines_with_shift(opposite)
    inconsistent = perturbation_vector_consistency(emb, drug, drug == "DMSO", line)

    assert consistent["A"] > 0.99
    assert inconsistent["A"] < -0.99


def test_label_retention_separable_vs_random():
    labels = np.repeat([0, 1, 2], 40)
    clustered = np.vstack([_gauss(40, shift=s) for s in (0.0, 8.0, 16.0)])

    assert label_retention(clustered, labels, k=5) > 0.95
    assert label_retention(_gauss(120), labels, k=5) < 0.6


def test_collapse_stats():
    isotropic = collapse_stats(_gauss(500))
    collapsed = collapse_stats(np.outer(RNG.normal(size=500), np.ones(D)))

    assert isotropic["effective_rank"] > 0.9 * D
    assert collapsed["effective_rank"] < 1.1
    assert collapsed["top_pc_variance"] > 0.99


def test_evaluate_alignment_reports_every_metric():
    n = 60
    obs = pd.DataFrame({
        "domain": [0] * 30 + [1] * 30,
        "lineage": [0] * n,
        "cell_type": ["tumor", "fibroblast"] * 15 + [None] * 30,
        "drug": [None] * 30 + ["DMSO"] * 15 + ["A"] * 15,
        "is_control": [False] * 30 + [True] * 15 + [False] * 15,
        "cell_line": [None] * 30 + ["L1", "L2"] * 15,
    })
    emb = _gauss(n)

    report = evaluate_alignment(emb, obs, context_emb=emb, min_cells=5, k=5)

    assert {"domain_mixing", "domain_mixing_controls", "paired_view", "separability",
            "vector_consistency", "label_retention", "collapse"} <= set(report)
    assert report["paired_view"]["top1_match"] == 1.0


def test_evaluate_alignment_validates_inputs():
    obs = pd.DataFrame({"domain_value": [0, 1]})

    with pytest.raises(ValueError, match="domain column"):
        evaluate_alignment(_gauss(2), obs)
    with pytest.raises(ValueError, match="rows"):
        evaluate_alignment(_gauss(3), obs, columns={"domain": "domain_value"})


def test_evaluate_alignment_skips_metrics_without_columns():
    obs = pd.DataFrame({"domain_value": [0] * 20 + [1] * 20})

    report = evaluate_alignment(_gauss(40), obs, columns={"domain": "domain_value"}, k=5)

    assert report["domain_mixing"]["ratio"] > 0
    assert report["separability"] is None and report["label_retention"] is None

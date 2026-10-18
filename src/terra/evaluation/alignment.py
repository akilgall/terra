"""Metrics for spatial / Tahoe-100M embedding alignment.

All functions take numpy embeddings (cells x dims) and per-cell label arrays,
e.g. the outputs of :func:`terra.embed_dataset` with the ``*_value`` label
columns of the tokenized datasets (see ``terra.tahoe.labels``).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from terra.tahoe.labels import DOMAIN_SPATIAL, DOMAIN_TAHOE

__all__ = [
    "domain_mixing",
    "paired_view_similarity",
    "perturbation_separability",
    "perturbation_vector_consistency",
    "label_retention",
    "collapse_stats",
    "evaluate_alignment",
]

MAX_FOLDS = 5
DEFAULT_COLUMNS = {"domain": "domain", "lineage": "lineage", "cell_type": "cell_type",
                   "drug": "drug", "is_control": "is_control", "cell_line": "cell_line"}


def _neighbours(emb: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest other cells of every cell."""
    index = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    return index.kneighbors(emb, return_distance=False)[:, 1:]


def _unit(emb: np.ndarray) -> np.ndarray:
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


def _folds(labels: np.ndarray) -> StratifiedKFold:
    n_splits = min(MAX_FOLDS, int(pd.Series(labels).value_counts().min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)


def domain_mixing(emb: np.ndarray, domain: np.ndarray, k: int = 15) -> dict[str, float]:
    """
    How well domains interleave in the embedding.

    Returns
    -------
    dict
        `observed`: mean fraction of a cell's k nearest neighbours from another
        domain; `expected`: the same under perfect mixing (domain proportions);
        `ratio`: observed / expected (1 = fully mixed, 0 = separated).
    """
    domain = np.asarray(domain)
    values, counts = np.unique(domain, return_counts=True)
    if len(values) < 2:
        raise ValueError("domain_mixing needs at least two domains.")
    neighbours = _neighbours(emb, min(k, len(emb) - 1))
    observed = float((domain[neighbours] != domain[:, None]).mean())
    own = counts[np.searchsorted(values, domain)]
    expected = float(((len(domain) - own) / (len(domain) - 1)).mean())
    return {"observed": observed, "expected": expected, "ratio": observed / expected}


def paired_view_similarity(full: np.ndarray, cell_only: np.ndarray) -> dict[str, float]:
    """
    Agreement of two embeddings of the same cells (row i = cell i).

    Returns
    -------
    dict
        `mean_cosine` of paired rows and `top1_match`, the fraction of cells
        whose nearest cell-only embedding (cosine) is their own.
    """
    full, cell_only = _unit(np.asarray(full)), _unit(np.asarray(cell_only))
    nearest = NearestNeighbors(n_neighbors=1).fit(cell_only).kneighbors(
        full, return_distance=False)[:, 0]
    return {"mean_cosine": float((full * cell_only).sum(axis=1).mean()),
            "top1_match": float((nearest == np.arange(len(full))).mean())}


def perturbation_separability(emb: np.ndarray,
                              drug: np.ndarray,
                              is_control: np.ndarray,
                              min_cells: int = 10,
                              cell_line: np.ndarray | None = None,
                              ) -> dict[str, float]:
    """
    Cross-validated ROC AUC of treated vs control cells, per drug.

    0.5 = indistinguishable from controls. With `cell_line`, controls are
    restricted to the lines the drug's cells come from. Drugs with fewer than
    `min_cells` cells (or controls) are skipped.
    """
    drug, is_control = np.asarray(drug, dtype=object), np.asarray(is_control, dtype=bool)
    aucs = {}
    for name in pd.unique(drug[~is_control & pd.notna(drug)]):
        treated = (drug == name) & ~is_control
        controls = is_control.copy()
        if cell_line is not None:
            controls &= np.isin(cell_line, np.asarray(cell_line)[treated])
        if treated.sum() < min_cells or controls.sum() < min_cells:
            continue
        rows = treated | controls
        scores = cross_val_score(LogisticRegression(max_iter=1000), emb[rows], treated[rows],
                                 cv=_folds(treated[rows]), scoring="roc_auc")
        aucs[name] = float(scores.mean())
    return aucs


def perturbation_vector_consistency(emb: np.ndarray,
                                    drug: np.ndarray,
                                    is_control: np.ndarray,
                                    cell_line: np.ndarray,
                                    min_cells: int = 5,
                                    ) -> dict[str, float]:
    """
    Whether a drug moves cells in the same direction in every cell line.

    Per line the shift is mean(treated) - mean(controls); the score is the
    mean pairwise cosine of these shifts (1 = same direction everywhere).
    Drugs measured in fewer than two lines are skipped.
    """
    drug, is_control = np.asarray(drug, dtype=object), np.asarray(is_control, dtype=bool)
    cell_line = np.asarray(cell_line, dtype=object)
    scores = {}
    for name in pd.unique(drug[~is_control & pd.notna(drug)]):
        shifts = []
        for line in pd.unique(cell_line[(drug == name) & ~is_control]):
            treated = (drug == name) & ~is_control & (cell_line == line)
            controls = is_control & (cell_line == line)
            if treated.sum() >= min_cells and controls.sum() >= min_cells:
                shifts.append(emb[treated].mean(axis=0) - emb[controls].mean(axis=0))
        if len(shifts) >= 2:
            unit = _unit(np.stack(shifts))
            scores[name] = float(np.mean([unit[i] @ unit[j]
                                          for i, j in combinations(range(len(unit)), 2)]))
    return scores


def label_retention(emb: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    """Cross-validated balanced accuracy of a kNN classifier for `labels`."""
    labels = np.asarray(labels)
    folds = _folds(labels)
    k = min(k, len(labels) - len(labels) // folds.get_n_splits() - 1)
    predicted = cross_val_predict(KNeighborsClassifier(n_neighbors=k), emb, labels, cv=folds)
    return float(balanced_accuracy_score(labels, predicted))


def collapse_stats(emb: np.ndarray) -> dict[str, float]:
    """
    Signs of representation collapse.

    Returns
    -------
    dict
        `std_mean` (mean per-dimension std), `effective_rank` (exp of the
        entropy of the PCA variance spectrum) and `top_pc_variance` (fraction
        of variance on the first principal component).
    """
    centred = emb - emb.mean(axis=0)
    variance = np.linalg.svd(centred, compute_uv=False) ** 2
    variance = variance / variance.sum()
    nonzero = variance[variance > 0]
    return {"std_mean": float(emb.std(axis=0).mean()),
            "effective_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
            "top_pc_variance": float(variance[0])}


def _column(obs: pd.DataFrame, columns: dict, role: str) -> np.ndarray | None:
    name = columns[role]
    return obs[name].to_numpy() if name in obs.columns else None


def _if_two_classes(fn, labels: np.ndarray, *args, **kwargs):
    return fn(*args, **kwargs) if len(pd.unique(labels)) >= 2 else None


def evaluate_alignment(emb: np.ndarray,
                       obs: pd.DataFrame,
                       context_emb: np.ndarray | None = None,
                       columns: dict | None = None,
                       k: int = 15,
                       min_cells: int = 10,
                       ) -> dict:
    """
    All alignment metrics for one set of embeddings.

    Parameters
    ----------
    emb:
        Cell embeddings without neighbour context (`cell_emb` of
        `embed_dataset`) for spatial and Tahoe cells, rows matching `obs`.
    obs:
        Per-cell labels. Roles and their default columns: `domain` (0 spatial,
        1 Tahoe), `lineage` (shared lineage code, < 0 unmapped), `cell_type`,
        `drug`, `is_control`, `cell_line`. Missing columns skip their metrics.
    context_emb:
        Embeddings with neighbour context (`spatial_cell_emb`); enables the
        paired-view metric on spatial cells.
    columns:
        Overrides of the role -> column mapping (e.g. `{'domain':
        'domain_value'}`).
    k, min_cells:
        Neighbours for kNN metrics; minimum cells per drug / line.

    Returns
    -------
    dict
        Metric name -> result (`None` when not computable).
    """
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    domain = _column(obs, columns, "domain")
    if domain is None:
        raise ValueError(f"obs needs a domain column ({columns['domain']!r}); pass "
                         "columns={'domain': ...} to name it.")
    if len(emb) != len(obs):
        raise ValueError(f"emb has {len(emb)} rows but obs has {len(obs)}.")
    spatial, tahoe = domain == DOMAIN_SPATIAL, domain == DOMAIN_TAHOE
    lineage = _column(obs, columns, "lineage")
    shared = np.ones(len(obs), dtype=bool) if lineage is None else lineage >= 0
    control_col = _column(obs, columns, "is_control")
    control = (np.zeros(len(obs), dtype=bool) if control_col is None
               else pd.Series(control_col).fillna(False).astype(bool).to_numpy())
    controls_view = shared & (spatial | (tahoe & control))
    drug, line = _column(obs, columns, "drug"), _column(obs, columns, "cell_line")
    cell_type = _column(obs, columns, "cell_type")
    labelled = spatial & (pd.notna(cell_type) if cell_type is not None else False)

    return {
        "collapse": collapse_stats(emb),
        "domain_mixing": _if_two_classes(domain_mixing, domain[shared], emb[shared],
                                         domain[shared], k),
        "domain_mixing_controls": _if_two_classes(domain_mixing, domain[controls_view],
                                                  emb[controls_view], domain[controls_view], k),
        "paired_view": (paired_view_similarity(context_emb[spatial], emb[spatial])
                        if context_emb is not None and spatial.any() else None),
        "separability": (perturbation_separability(emb[tahoe], drug[tahoe], control[tahoe],
                                                   min_cells, None if line is None
                                                   else line[tahoe])
                         if drug is not None else None),
        "vector_consistency": (perturbation_vector_consistency(
            emb[tahoe], drug[tahoe], control[tahoe], line[tahoe], min_cells)
            if drug is not None and line is not None else None),
        "label_retention": (_if_two_classes(label_retention, cell_type[labelled],
                                            emb[labelled], cell_type[labelled], k)
                            if cell_type is not None else None),
    }

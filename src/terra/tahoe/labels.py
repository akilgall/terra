"""Shared label codes and per-cell ``*_value`` columns for both domains.

Spatial and Tahoe-100M datasets get the same integer columns so the
fine-tuning loop can route alignment by lineage and evaluation can group by
drug: ``domain_value`` (0 spatial, 1 Tahoe), ``lineage_value``,
``cell_line_value``, ``drug_value``, ``dose_value`` (nM) and
``is_control_value``. ``CellBaseDataset`` exposes every numeric ``*_value``
column on each item automatically. Unmapped or not-applicable values are -1.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml

__all__ = [
    "DOMAIN_SPATIAL",
    "DOMAIN_TAHOE",
    "UNMAPPED",
    "LabelCodes",
    "load_lineage_map",
    "lineage_of",
    "add_value_columns",
]

DOMAIN_SPATIAL = 0
DOMAIN_TAHOE = 1
UNMAPPED = -1


def load_lineage_map(path: str | Path) -> dict:
    """Read a lineage map: `{lineages: {name: {tahoe_organs, spatial_cell_types}}}`."""
    with open(path) as f:
        lineage_map = yaml.safe_load(f)
    if not isinstance(lineage_map, dict) or "lineages" not in lineage_map:
        raise ValueError(f"{path}: expected a top-level 'lineages' mapping.")
    return lineage_map


def lineage_of(value: str, source: str, lineage_map: dict) -> str | None:
    """Shared lineage for a Tahoe organ or a spatial cell type (`None` if unmapped)."""
    for name, members in lineage_map["lineages"].items():
        if value in (members.get(source) or []):
            return name
    return None


@dataclass(frozen=True)
class LabelCodes:
    """Integer codes shared by spatial and Tahoe datasets."""

    lineage: dict[str, int]
    cell_line: dict[str, int]
    drug: dict[str, int]

    @classmethod
    def from_values(cls,
                    lineage_map: dict,
                    cell_lines: Iterable[str],
                    drugs: Iterable[str],
                    ) -> "LabelCodes":
        """Deterministic (sorted) codes."""
        def _codes(values: Iterable[str]) -> dict[str, int]:
            return {v: i for i, v in enumerate(sorted({str(v) for v in values}))}
        return cls(lineage=_codes(lineage_map["lineages"]),
                   cell_line=_codes(cell_lines), drug=_codes(drugs))

    def save(self, path: str | Path) -> None:
        """Write the codes as JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "LabelCodes":
        """Read codes written by `save`."""
        with open(path) as f:
            return cls(**json.load(f))


def _code_column(values: pd.Series | None, codes: dict[str, int]) -> np.ndarray:
    if values is None:
        return np.full(0, UNMAPPED, dtype=np.int64)
    return np.array([codes.get(str(v), UNMAPPED) for v in values], dtype=np.int64)


def _dose_nm(doses: pd.Series | None, n: int) -> np.ndarray:
    if doses is None:
        return np.full(n, UNMAPPED, dtype=np.int64)
    values = doses.to_numpy(dtype=float)
    return np.where(np.isnan(values), UNMAPPED, np.rint(values * 1000.0)).astype(np.int64)


def _value_columns(obs: pd.DataFrame,
                   codes: LabelCodes,
                   lineage_map: dict,
                   domain: int,
                   lineage_column: str,
                   lineage_source: str,
                   ) -> dict[str, np.ndarray]:
    n = len(obs)
    lineages = [lineage_of(str(v), lineage_source, lineage_map) for v in obs[lineage_column]]

    def optional(column: str, column_codes: dict[str, int]) -> np.ndarray:
        if column not in obs:
            return np.full(n, UNMAPPED, dtype=np.int64)
        return _code_column(obs[column], column_codes)

    is_control = (obs["is_control"].to_numpy(dtype=bool) if "is_control" in obs
                  else np.zeros(n, dtype=bool))
    return {
        "domain_value": np.full(n, domain, dtype=np.int64),
        "lineage_value": np.array([codes.lineage.get(name, UNMAPPED) if name else UNMAPPED
                                   for name in lineages], dtype=np.int64),
        "cell_line_value": optional("cell_line_id", codes.cell_line),
        "drug_value": optional("drug", codes.drug),
        "dose_value": _dose_nm(obs["dose_uM"] if "dose_uM" in obs else None, n),
        "is_control_value": is_control.astype(np.int64),
    }


def add_value_columns(dataset: datasets.Dataset,
                      obs: pd.DataFrame,
                      codes: LabelCodes,
                      lineage_map: dict,
                      domain: int,
                      lineage_column: str,
                      lineage_source: str,
                      ) -> datasets.Dataset:
    """
    Return a copy of a tokenized dataset with the `*_value` label columns.

    Rows are matched to `obs` by `cell_id`, so the dataset order may differ
    from `obs`.

    Parameters
    ----------
    dataset:
        Tokenized dataset with a `cell_id` column.
    obs:
        Cell metadata with `cell_id` and `lineage_column`; Tahoe obs also
        has `cell_line_id`, `drug`, `dose_uM` and `is_control`.
    codes:
        Shared label codes.
    lineage_map:
        Mapping of shared lineages to Tahoe organs / spatial cell types.
    domain:
        `DOMAIN_SPATIAL` or `DOMAIN_TAHOE`.
    lineage_column:
        `obs` column holding the organ (Tahoe) or cell type (spatial).
    lineage_source:
        `'tahoe_organs'` or `'spatial_cell_types'`.
    """
    by_id = obs.set_index("cell_id", drop=False)
    order = [str(c) for c in dataset["cell_id"]]
    missing = [c for c in order if c not in by_id.index]
    if missing:
        raise ValueError(
            f"{len(missing)} dataset cell_id values are absent from obs "
            f"(first: {missing[:3]}).")
    columns = _value_columns(by_id.loc[order], codes, lineage_map, domain,
                             lineage_column, lineage_source)
    out = dataset
    for name, values in columns.items():
        if name in out.column_names:
            out = out.remove_columns(name)
        out = out.add_column(name, values.tolist())
    return out

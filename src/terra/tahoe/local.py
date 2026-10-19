"""Read a local copy of the Tahoe-100M ``expression_data`` (the full download).

Two layouts are recognised:

- parquet shards (``data/train-*.parquet``), as written by
  ``huggingface_hub.snapshot_download(REPO_ID, repo_type="dataset", ...)`` or
  kept in the Hugging Face hub cache. Read lazily; filters on ``sample`` or
  ``plate`` skip row groups using their statistics.
- the Arrow files of a ``datasets`` cache, as written by
  ``datasets.load_dataset(REPO_ID, cache_dir=DIR)``. Memory-mapped.

Either opens as a ``pyarrow.dataset.Dataset``. Cells are picked in two steps,
so that expression lists are read only for the cells kept: `read_obs` reads
the metadata of the cells matching a filter, a subset of barcodes is chosen
from it (e.g. `sample_per_group`), and `read_cells` reads those cells for
`terra.tahoe.convert.table_to_anndata`.
"""

import functools
import logging
import operator
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .convert import CONTROL_DRUG, cell_line_table, sample_table
from .stream import REPO_ID, sample_number

logger = logging.getLogger(__name__)

__all__ = [
    "BARCODE",
    "OBS_COLUMNS",
    "find_expression_files",
    "open_expression_data",
    "metadata_file",
    "select_samples",
    "select_cell_lines",
    "cells_filter",
    "cell_counts",
    "read_obs",
    "sample_per_group",
    "read_cells",
]

BARCODE = "BARCODE_SUB_LIB_ID"
OBS_COLUMNS = (BARCODE, "plate", "sample", "cell_line_id", "drug")
_EXPRESSION_COLUMNS = ("genes", "expressions")


def _arrow_schema(path: Path) -> pa.Schema:
    with pa.memory_map(str(path)) as source:
        return pa.ipc.open_stream(source).schema


def _expression_dirs(files: Iterable[Path],
                     read_schema: Callable[[Path], pa.Schema],
                     ) -> dict[Path, list[Path]]:
    """Files grouped by directory, for directories whose files hold expression data."""
    by_dir: dict[Path, list[Path]] = {}
    for path in sorted(files):
        by_dir.setdefault(path.parent, []).append(path)
    return {folder: paths for folder, paths in by_dir.items()
            if all(c in read_schema(paths[0]).names for c in _EXPRESSION_COLUMNS)}


def find_expression_files(root: str | Path) -> tuple[str, list[Path]]:
    """
    Locate the expression-data files of a local download under `root`.

    Returns `("parquet", files)` or `("arrow", files)`; parquet shards are
    preferred when both exist. Raises `FileNotFoundError` without any, and
    `ValueError` when they lie in several directories (e.g. two dataset
    caches), which would duplicate cells: pass the intended directory then.
    """
    root = Path(root)
    arrow_files = (p for p in root.rglob("*.arrow") if not p.name.startswith("cache-"))
    for kind, dirs in [("parquet", _expression_dirs(root.rglob("*.parquet"), pq.read_schema)),
                       ("arrow", _expression_dirs(arrow_files, _arrow_schema))]:
        if len(dirs) > 1:
            raise ValueError(f"Tahoe-100M expression {kind} files in several directories "
                             f"under {root}: {sorted(map(str, dirs))}. Pass one of them.")
        if dirs:
            return kind, next(iter(dirs.values()))
    raise FileNotFoundError(
        f"No Tahoe-100M expression data under {root}: expected parquet or Arrow files "
        f"with columns {list(_EXPRESSION_COLUMNS)}.")


def open_expression_data(root: str | Path) -> pads.Dataset:
    """Open a local download (see `find_expression_files`) as one dataset."""
    kind, files = find_expression_files(root)
    logger.info(f"Opening {len(files)} {kind} file(s) of Tahoe-100M expression data.")
    if kind == "parquet":
        return pads.dataset([str(f) for f in files], format="parquet")
    tables = [pa.ipc.open_stream(pa.memory_map(str(f))).read_all() for f in files]
    return pads.dataset(pa.concat_tables(tables))


def metadata_file(name: str, root: str | Path) -> Path:
    """
    Path of the metadata table `name` (e.g. `"sample_metadata"`).

    Looks for `metadata/<name>.parquet` anywhere under `root` (a snapshot
    or a hub cache); otherwise downloads it into the hub cache under `root`.
    """
    local = sorted(Path(root).rglob(f"metadata/{name}.parquet"))
    if local:
        return local[0]
    import huggingface_hub

    return Path(huggingface_hub.hf_hub_download(
        REPO_ID, f"metadata/{name}.parquet", repo_type="dataset", cache_dir=str(root)))


def _check_known(wanted: Iterable[str], known: pd.Series, what: str) -> None:
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise ValueError(f"Unknown {what}: {unknown}.")


def select_samples(sample_metadata: pd.DataFrame,
                   drugs: Sequence[str] | None = None,
                   doses_uM: Sequence[float] | None = None,
                   plates: Sequence[str] | None = None,
                   include_controls: bool = True,
                   ) -> list[str]:
    """
    Sample (well) ids matching every given criterion, in sample order.

    Drug names match regardless of surrounding whitespace. With
    `include_controls`, the DMSO wells of the selected wells' plates are
    added: controls from the same plate share its batch.
    """
    meta = sample_metadata.assign(drug=sample_metadata["drug"].str.strip())
    meta = meta.join(sample_table(sample_metadata)["dose_uM"], on="sample")
    keep = np.ones(len(meta), dtype=bool)
    if drugs is not None:
        wanted = [d.strip() for d in drugs]
        _check_known(wanted, meta["drug"], "drugs")
        keep &= meta["drug"].isin(wanted).to_numpy()
    if doses_uM is not None:
        doses = meta["dose_uM"].to_numpy()[:, None]
        keep &= np.isclose(doses, np.asarray(doses_uM, dtype=float)[None, :]).any(axis=1)
    if plates is not None:
        _check_known(plates, meta["plate"], "plates")
        keep &= meta["plate"].isin(plates).to_numpy()
    if not keep.any():
        raise ValueError("No samples match the selection.")
    selected = meta[keep]
    if include_controls:
        controls = meta[meta["drug"].eq(CONTROL_DRUG) & meta["plate"].isin(selected["plate"])]
        selected = pd.concat([selected, controls])
    return sorted(set(selected["sample"]), key=sample_number)


def select_cell_lines(cell_line_metadata: pd.DataFrame,
                      organs: Sequence[str] | None = None,
                      cell_names: Sequence[str] | None = None,
                      ) -> list[str]:
    """Cellosaurus ids (`cell_line_id` in the data) matching every given criterion."""
    lines = cell_line_table(cell_line_metadata)
    keep = np.ones(len(lines), dtype=bool)
    if organs is not None:
        _check_known(organs, lines["organ"], "organs")
        keep &= lines["organ"].isin(organs).to_numpy()
    if cell_names is not None:
        _check_known(cell_names, lines["cell_name"], "cell names")
        keep &= lines["cell_name"].isin(cell_names).to_numpy()
    return sorted(lines.index[keep])


def cells_filter(samples: Iterable[str] | None = None,
                 cell_lines: Iterable[str] | None = None,
                 ) -> pc.Expression | None:
    """Filter for cells of the given samples and cell lines (`None`: no criteria)."""
    terms = []
    if samples is not None:
        terms.append(pc.field("sample").isin(list(samples)))
    if cell_lines is not None:
        terms.append(pc.field("cell_line_id").isin(list(cell_lines)))
    return functools.reduce(operator.and_, terms) if terms else None


def cell_counts(dataset: pads.Dataset,
                by: Sequence[str] = ("plate", "sample", "cell_line_id", "drug"),
                filter: pc.Expression | None = None,
                batch_size: int = 1_000_000,
                ) -> pd.DataFrame:
    """
    Number of cells (`n_cells`) per combination of the `by` columns.

    One pass over those columns only, batch by batch, so memory stays small
    even for all ~96M cells.
    """
    by = list(by)
    count = (by[0], "count", pc.CountOptions(mode="all"))
    parts = [pa.Table.from_batches([batch]).group_by(by).aggregate([count])
             for batch in dataset.to_batches(columns=by, filter=filter, batch_size=batch_size)
             if batch.num_rows]
    if not parts:
        return pd.DataFrame({**{c: [] for c in by}, "n_cells": []})
    totals = pa.concat_tables(parts).group_by(by).aggregate([(f"{by[0]}_count", "sum")])
    return (totals.to_pandas()
                  .rename(columns={f"{by[0]}_count_sum": "n_cells"})[by + ["n_cells"]]
                  .sort_values(by).reset_index(drop=True))


def read_obs(dataset: pads.Dataset,
             filter: pc.Expression | None = None,
             columns: Sequence[str] = OBS_COLUMNS,
             ) -> pd.DataFrame:
    """
    Metadata columns of the cells matching `filter`, without expression.

    String columns other than the barcode become categoricals (small in
    memory); surrounding whitespace is removed from drug names.
    """
    table = dataset.to_table(columns=list(columns), filter=filter)
    arrays = {}
    for name in table.column_names:
        column = pc.utf8_trim_whitespace(table[name]) if name == "drug" else table[name]
        arrays[name] = column if name == BARCODE else column.dictionary_encode()
    return pa.table(arrays).to_pandas()


def sample_per_group(obs: pd.DataFrame,
                     by: Sequence[str],
                     n: int,
                     seed: int = 0,
                     ) -> pd.DataFrame:
    """
    Up to `n` random rows per group of `by`, in their original order.

    Smaller groups are kept whole. `obs` needs a unique index, as from
    `read_obs`.
    """
    shuffled = obs.sample(frac=1.0, random_state=seed)
    picked = shuffled.groupby(list(by), observed=True, sort=False, dropna=False).head(n)
    return obs[obs.index.isin(picked.index)]


def read_cells(dataset: pads.Dataset,
               barcodes: Iterable[str],
               filter: pc.Expression | None = None,
               columns: Sequence[str] | None = None,
               ) -> pa.Table:
    """
    The rows (with expression) of the given cell barcodes.

    Passing the `filter` the barcodes were chosen with lets parquet shards
    skip row groups.
    """
    wanted = sorted({str(b) for b in barcodes})
    expression = pc.field(BARCODE).isin(wanted)
    if filter is not None:
        expression = filter & expression
    table = dataset.to_table(columns=None if columns is None else list(columns),
                             filter=expression)
    if table.num_rows < len(wanted):
        logger.warning(f"{len(wanted) - table.num_rows} of {len(wanted)} requested cells "
                       "were not found.")
    return table

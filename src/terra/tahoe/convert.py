"""Convert Tahoe-100M ``expression_data`` rows into an AnnData.

Each record holds parallel ``genes`` (Tahoe vocabulary token ids) and
``expressions`` (raw UMI counts) lists, led by a CLS marker
(``genes[0] == 1``, ``expressions[0] == -2.0``). Token ids map to Ensembl ids
through the ``gene_metadata`` table.
"""

import ast
import logging
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_DRUG",
    "gene_lookup",
    "sample_table",
    "cell_line_table",
    "table_to_anndata",
    "records_to_anndata",
    "restrict_to_vocab",
]

# Vehicle control; matched exactly ('Trametinib (DMSO_TF solvate)' is a drug).
CONTROL_DRUG = "DMSO_TF"
_CATEGORICAL_OBS = ("drug", "sample", "plate", "cell_line_id", "cell_name",
                    "organ", "moa_fine")
_TABLE_COLUMNS = ("genes", "expressions", "BARCODE_SUB_LIB_ID", "drug", "sample", "plate",
                  "cell_line_id", "moa-fine", "pubchem_cid")


def gene_lookup(gene_metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the token-id -> column lookup and the AnnData `var` table.

    Returns
    -------
    lookup:
        Array mapping a Tahoe token id to its column (-1 if unknown).
    var:
        Genes in token-id order, indexed by Ensembl id, with columns
        `ensembl_id` (read by TERRA's tokenizer), `gene_symbol` and
        `tahoe_token_id`.
    """
    genes = gene_metadata.sort_values("token_id").reset_index(drop=True)
    if not genes["ensembl_id"].is_unique:
        raise ValueError("gene_metadata ensembl_id values are not unique.")
    lookup = np.full(int(genes["token_id"].max()) + 1, -1, dtype=np.int64)
    lookup[genes["token_id"].to_numpy()] = np.arange(len(genes))
    var = pd.DataFrame(
        {"ensembl_id": genes["ensembl_id"].to_numpy(),
         "gene_symbol": genes["gene_symbol"].to_numpy(),
         "tahoe_token_id": genes["token_id"].to_numpy()},
        index=pd.Index(genes["ensembl_id"].to_numpy(), name="ensembl_id"))
    return lookup, var


def _parse_dose(drugname_drugconc: str) -> tuple[float, str]:
    """`"[('Drug', 0.5, 'uM')]"` -> (0.5, 'uM'); unparsable -> (nan, '')."""
    try:
        entries = ast.literal_eval(drugname_drugconc)
        return float(entries[0][1]), str(entries[0][2])
    except (ValueError, SyntaxError, IndexError, TypeError):
        return float("nan"), ""


def sample_table(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-sample dose table indexed by sample id."""
    doses = sample_metadata["drugname_drugconc"].map(_parse_dose)
    return pd.DataFrame(
        {"drugname_drugconc": sample_metadata["drugname_drugconc"].to_numpy(),
         "dose_uM": [dose for dose, _ in doses],
         "dose_unit": [unit for _, unit in doses]},
        index=pd.Index(sample_metadata["sample"].to_numpy(), name="sample"))


def cell_line_table(cell_line_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell-line table, one row per Cellosaurus id.

    The metadata has one row per driver mutation, so lines are de-duplicated.
    """
    lines = cell_line_metadata.drop_duplicates("Cell_ID_Cellosaur")
    return (lines.set_index("Cell_ID_Cellosaur")[["cell_name", "Organ", "Cell_ID_DepMap"]]
                 .rename(columns={"Organ": "organ", "Cell_ID_DepMap": "depmap_id"}))


def _pubchem_cids(values: pd.Series) -> pd.Series:
    """`'5288382.0'` -> 5288382; missing or unparsable ids -> NA."""
    numbers = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Nullable integers: some compounds have no PubChem id, and h5ad cannot
    # store a mixed int/NA object column.
    return np.trunc(numbers).astype("Int64")


def _require_columns(table: pa.Table) -> None:
    missing = [c for c in _TABLE_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"Tahoe-100M table lacks columns {missing}.")


def _table_counts(table: pa.Table,
                  lookup: np.ndarray,
                  n_genes: int,
                  ) -> tuple[sp.csr_matrix, int]:
    """
    Cells x genes counts of the `genes` / `expressions` list columns.

    Drops each cell's CLS marker (a negative first expression) and gene
    tokens absent from `lookup`; returns the matrix and the number dropped.
    """
    lengths = pc.list_value_length(table["genes"]).to_numpy(zero_copy_only=False)
    expr_lengths = pc.list_value_length(table["expressions"]).to_numpy(zero_copy_only=False)
    barcodes = table["BARCODE_SUB_LIB_ID"]
    mismatched = np.flatnonzero(lengths != expr_lengths)
    if mismatched.size:
        raise ValueError(
            f"genes/expressions length mismatch for {barcodes[int(mismatched[0])]}.")
    genes = pc.list_flatten(table["genes"]).to_numpy().astype(np.int64, copy=False)
    counts = pc.list_flatten(table["expressions"]).to_numpy().astype(np.float32, copy=False)
    ends = np.cumsum(lengths)
    starts = ends - lengths

    keep = np.ones(len(genes), dtype=bool)
    nonempty = starts[lengths > 0]
    keep[nonempty[counts[nonempty] < 0]] = False  # CLS marker
    columns = np.full(len(genes), -1, dtype=np.int64)
    in_table = (genes >= 0) & (genes < len(lookup))
    columns[in_table] = lookup[genes[in_table]]
    n_dropped = int((keep & (columns < 0)).sum())
    keep &= columns >= 0

    fractional = np.flatnonzero(keep & (np.mod(counts, 1) != 0))
    if fractional.size:
        row = int(np.searchsorted(starts, fractional[0], side="right")) - 1
        raise ValueError(f"Non-integer expression values in {barcodes[row]}; "
                         "expected raw counts.")
    kept_before = np.concatenate([[0], np.cumsum(keep)])
    indptr = np.concatenate([[0], kept_before[ends]])
    X = sp.csr_matrix((counts[keep].astype(np.int32), columns[keep], indptr),
                      shape=(table.num_rows, n_genes))
    return X, n_dropped


def _table_obs(table: pa.Table) -> pd.DataFrame:
    meta = table.select(list(_TABLE_COLUMNS[2:])).to_pandas()
    return pd.DataFrame({"cell_id": meta["BARCODE_SUB_LIB_ID"].astype(str),
                         "drug": meta["drug"].str.strip(),
                         "sample": meta["sample"],
                         "plate": meta["plate"],
                         "cell_line_id": meta["cell_line_id"],
                         "moa_fine": meta["moa-fine"],
                         "pubchem_cid": _pubchem_cids(meta["pubchem_cid"])})


def table_to_anndata(table: pa.Table,
                     lookup: np.ndarray,
                     var: pd.DataFrame,
                     samples: pd.DataFrame,
                     cell_lines: pd.DataFrame,
                     ) -> ad.AnnData:
    """
    Convert an Arrow table of `expression_data` rows to an AnnData of raw counts.

    Works column-wise, without a per-cell Python loop. `obs` holds `cell_id`
    (the record barcode), `drug`, `sample`, `plate`, `cell_line_id`,
    `moa_fine`, `pubchem_cid`, the dose columns from `samples`, `cell_name` /
    `organ` / `depmap_id` from `cell_lines`, and `is_control`
    (`drug == 'DMSO_TF'`).
    """
    _require_columns(table)
    if table.num_rows == 0:
        raise ValueError("No cells to convert.")
    X, n_dropped = _table_counts(table, lookup, var.shape[0])
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} gene entries absent from gene_metadata.")
    obs = _table_obs(table)
    obs = obs.join(samples, on="sample").join(cell_lines, on="cell_line_id")
    obs["is_control"] = obs["drug"].eq(CONTROL_DRUG)
    obs.index = pd.Index(obs["cell_id"].to_numpy(), name="barcode")
    for col in _CATEGORICAL_OBS:
        obs[col] = obs[col].astype("category")
    return ad.AnnData(X=X, obs=obs, var=var.copy())


def records_to_anndata(records: Iterable[dict],
                       lookup: np.ndarray,
                       var: pd.DataFrame,
                       samples: pd.DataFrame,
                       cell_lines: pd.DataFrame,
                       ) -> ad.AnnData:
    """Convert expression records (row dicts) to an AnnData; see `table_to_anndata`."""
    records = list(records)
    if not records:
        raise ValueError("No records to convert.")
    return table_to_anndata(pa.Table.from_pylist(records), lookup, var, samples, cell_lines)


def restrict_to_vocab(adata: ad.AnnData, token_dict: dict) -> ad.AnnData:
    """Copy of `adata` keeping only genes present in a TERRA token dict."""
    keep = adata.var["ensembl_id"].isin(set(token_dict)).to_numpy()
    return adata[:, keep].copy()

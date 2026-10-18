"""Convert Tahoe-100M ``expression_data`` records into an AnnData.

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
import scipy.sparse as sp

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_DRUG",
    "gene_lookup",
    "sample_table",
    "cell_line_table",
    "records_to_anndata",
    "restrict_to_vocab",
]

# Vehicle control; matched exactly ('Trametinib (DMSO_TF solvate)' is a drug).
CONTROL_DRUG = "DMSO_TF"
_CATEGORICAL_OBS = ("drug", "sample", "plate", "cell_line_id", "cell_name",
                    "organ", "moa_fine")


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


def _pubchem_cid(value) -> object:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return pd.NA


def _record_counts(record: dict, lookup: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Columns and integer counts of one record (CLS marker dropped)."""
    genes = np.asarray(record["genes"], dtype=np.int64)
    counts = np.asarray(record["expressions"], dtype=np.float32)
    if len(genes) != len(counts):
        raise ValueError(
            f"genes/expressions length mismatch for {record['BARCODE_SUB_LIB_ID']}.")
    if len(counts) and counts[0] < 0:
        genes, counts = genes[1:], counts[1:]
    in_table = genes < len(lookup)
    columns = np.full(len(genes), -1, dtype=np.int64)
    columns[in_table] = lookup[genes[in_table]]
    keep = columns >= 0
    if not np.all(np.mod(counts[keep], 1) == 0):
        raise ValueError(
            f"Non-integer expression values in {record['BARCODE_SUB_LIB_ID']}; "
            "expected raw counts.")
    return columns[keep], counts[keep].astype(np.int32), int((~keep).sum())


def _obs_row(record: dict) -> dict:
    return {"cell_id": record["BARCODE_SUB_LIB_ID"],
            "drug": record["drug"].strip(),
            "sample": record["sample"],
            "plate": record["plate"],
            "cell_line_id": record["cell_line_id"],
            "moa_fine": record["moa-fine"],
            "pubchem_cid": _pubchem_cid(record["pubchem_cid"])}


def records_to_anndata(records: Iterable[dict],
                       lookup: np.ndarray,
                       var: pd.DataFrame,
                       samples: pd.DataFrame,
                       cell_lines: pd.DataFrame,
                       ) -> ad.AnnData:
    """
    Convert expression records to an AnnData of raw integer counts.

    `obs` holds `cell_id` (the record barcode), `drug`, `sample`, `plate`,
    `cell_line_id`, `moa_fine`, `pubchem_cid`, the dose columns from
    `samples`, `cell_name` / `organ` / `depmap_id` from `cell_lines`, and
    `is_control` (`drug == 'DMSO_TF'`).
    """
    data, indices, indptr, obs_rows = [], [], [0], []
    n_dropped = 0
    for record in records:
        columns, counts, dropped = _record_counts(record, lookup)
        data.append(counts)
        indices.append(columns)
        indptr.append(indptr[-1] + len(columns))
        obs_rows.append(_obs_row(record))
        n_dropped += dropped
    if not obs_rows:
        raise ValueError("No records to convert.")
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} gene entries absent from gene_metadata.")
    X = sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(len(obs_rows), var.shape[0]), dtype=np.int32)
    obs = pd.DataFrame(obs_rows)
    obs = obs.join(samples, on="sample").join(cell_lines, on="cell_line_id")
    obs["is_control"] = obs["drug"].eq(CONTROL_DRUG)
    # Nullable integers: some compounds have no PubChem id, and h5ad cannot
    # store a mixed int/NA object column.
    obs["pubchem_cid"] = obs["pubchem_cid"].astype("Int64")
    obs.index = pd.Index(obs["cell_id"].to_numpy(), name="barcode")
    for col in _CATEGORICAL_OBS:
        obs[col] = obs[col].astype("category")
    return ad.AnnData(X=X, obs=obs, var=var.copy())


def restrict_to_vocab(adata: ad.AnnData, token_dict: dict) -> ad.AnnData:
    """Copy of `adata` keeping only genes present in a TERRA token dict."""
    keep = adata.var["ensembl_id"].isin(set(token_dict)).to_numpy()
    return adata[:, keep].copy()

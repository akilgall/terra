"""``terra.tahoe.convert``: Tahoe-100M records -> AnnData (offline)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from terra.tahoe.convert import (
    CONTROL_DRUG,
    cell_line_table,
    gene_lookup,
    records_to_anndata,
    restrict_to_vocab,
    sample_table,
    table_to_anndata,
)

GENE_METADATA = pd.DataFrame({
    "gene_symbol": ["G3", "G1", "G2", "G4"],
    "ensembl_id": ["ENSG00000000003", "ENSG00000000001", "ENSG00000000002", "ENSG00000000004"],
    "token_id": [5, 3, 4, 6],
})
SAMPLE_METADATA = pd.DataFrame({
    "sample": ["smp_1", "smp_2", "smp_3"],
    "plate": ["plate1"] * 3,
    "drug": ["Trametinib", "Trametinib (DMSO_TF solvate)", "DMSO_TF"],
    "drugname_drugconc": ["[('Trametinib', 0.5, 'uM')]",
                          "[('Trametinib (DMSO_TF solvate)', 5.0, 'uM')]",
                          "[('DMSO_TF', 0.0, 'uM')]"],
})
CELL_LINE_METADATA = pd.DataFrame({
    "cell_name": ["A549", "A549", "HT-29"],
    "Cell_ID_Cellosaur": ["CVCL_0023", "CVCL_0023", "CVCL_0320"],
    "Cell_ID_DepMap": ["ACH-000681", "ACH-000681", "ACH-000552"],
    "Organ": ["Lung", "Lung", "Bowel"],
})


def _record(barcode, genes, expressions, sample="smp_1", drug="Trametinib",
            cell_line="CVCL_0023", pubchem="5288382.0"):
    return {"genes": [1] + genes, "expressions": [-2.0] + expressions, "drug": drug,
            "sample": sample, "BARCODE_SUB_LIB_ID": barcode, "cell_line_id": cell_line,
            "moa-fine": "MEK inhibitor", "canonical_smiles": "C", "pubchem_cid": pubchem,
            "plate": "plate1"}


RECORDS = [
    _record("b0", [3, 5], [2.0, 1.0]),
    _record("b1", [4], [7.0], sample="smp_2", drug="Trametinib (DMSO_TF solvate) "),
    _record("b2", [3, 4, 6], [1.0, 1.0, 3.0], sample="smp_3", drug="DMSO_TF",
            cell_line="CVCL_0320", pubchem="nan"),
]


def _adata():
    lut, var = gene_lookup(GENE_METADATA)
    return records_to_anndata(RECORDS, lut, var, sample_table(SAMPLE_METADATA),
                              cell_line_table(CELL_LINE_METADATA))


def test_gene_lookup_orders_by_token_id():
    lut, var = gene_lookup(GENE_METADATA)

    assert var["ensembl_id"].tolist() == ["ENSG00000000001", "ENSG00000000002",
                                          "ENSG00000000003", "ENSG00000000004"]
    assert lut[3] == 0 and lut[6] == 3 and lut[0] == -1


def test_counts_drop_cls_marker_and_stay_integer():
    adata = _adata()

    assert adata.shape == (3, 4)
    assert adata.X.dtype == np.int32
    assert adata.X.toarray().tolist() == [[2, 0, 1, 0], [0, 7, 0, 0], [1, 1, 0, 3]]


def test_obs_metadata():
    obs = _adata().obs

    assert obs["cell_id"].tolist() == ["b0", "b1", "b2"]
    assert obs["drug"].tolist() == ["Trametinib", "Trametinib (DMSO_TF solvate)", CONTROL_DRUG]
    assert obs["is_control"].tolist() == [False, False, True]
    assert obs["dose_uM"].tolist() == [0.5, 5.0, 0.0]
    assert obs["organ"].tolist() == ["Lung", "Lung", "Bowel"]
    assert obs["cell_name"].tolist() == ["A549", "A549", "HT-29"]
    assert obs["pubchem_cid"].tolist()[:2] == [5288382, 5288382]
    assert pd.isna(obs["pubchem_cid"].iloc[2])


def test_anndata_round_trips_through_h5ad(tmp_path):
    import anndata as ad

    path = tmp_path / "tahoe.h5ad"
    _adata().write_h5ad(path)
    back = ad.read_h5ad(path)

    assert back.X.toarray().tolist() == _adata().X.toarray().tolist()
    assert back.obs["is_control"].tolist() == [False, False, True]
    assert back.obs["pubchem_cid"].tolist()[:2] == [5288382, 5288382]


def test_var_carries_ensembl_column_for_tokenizer():
    var = _adata().var

    assert var["ensembl_id"].tolist() == list(var.index)
    assert var["gene_symbol"].tolist() == ["G1", "G2", "G3", "G4"]


def test_non_integer_counts_raise():
    lut, var = gene_lookup(GENE_METADATA)
    bad = [_record("b", [3], [1.5])]

    with pytest.raises(ValueError, match="raw counts"):
        records_to_anndata(bad, lut, var, sample_table(SAMPLE_METADATA),
                           cell_line_table(CELL_LINE_METADATA))


def test_restrict_to_vocab_keeps_token_genes_only():
    adata = _adata()
    token_dict = {"<pad>": 0, "ENSG00000000001": 2, "ENSG00000000004": 3}

    restricted = restrict_to_vocab(adata, token_dict)

    assert restricted.var_names.tolist() == ["ENSG00000000001", "ENSG00000000004"]
    assert adata.n_vars == 4  # input untouched


# ---------------------------------------------------------------------------
# table_to_anndata: Arrow tables (columns of a parquet / Arrow download)
# ---------------------------------------------------------------------------

def _convert_table(table):
    lut, var = gene_lookup(GENE_METADATA)
    return table_to_anndata(table, lut, var, sample_table(SAMPLE_METADATA),
                            cell_line_table(CELL_LINE_METADATA))


def test_table_to_anndata_matches_records():
    adata = _convert_table(pa.Table.from_pylist(RECORDS))

    assert adata.X.toarray().tolist() == _adata().X.toarray().tolist()
    pd.testing.assert_frame_equal(adata.obs, _adata().obs)


def test_table_to_anndata_reads_chunked_and_sliced_columns():
    table = pa.concat_tables([pa.Table.from_pylist(RECORDS[:2]),
                              pa.Table.from_pylist(RECORDS[2:])])

    whole = _convert_table(table)
    tail = _convert_table(table.slice(1))

    assert whole.X.toarray().tolist() == [[2, 0, 1, 0], [0, 7, 0, 0], [1, 1, 0, 3]]
    assert tail.X.toarray().tolist() == [[0, 7, 0, 0], [1, 1, 0, 3]]
    assert tail.obs["cell_id"].tolist() == ["b1", "b2"]


def test_table_to_anndata_keeps_first_gene_without_cls_marker():
    record = {**_record("b", [], []), "genes": [3, 5], "expressions": [4.0, 1.0]}

    adata = _convert_table(pa.Table.from_pylist([record]))

    assert adata.X.toarray().tolist() == [[4, 0, 1, 0]]


def test_table_to_anndata_handles_cells_without_genes():
    records = [_record("b0", [], []), {**_record("b1", [], []), "genes": [], "expressions": []},
               _record("b2", [4], [2.0])]

    adata = _convert_table(pa.Table.from_pylist(records))

    assert adata.X.toarray().tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]]


def test_table_to_anndata_drops_unknown_tokens(caplog):
    with caplog.at_level("WARNING"):
        adata = _convert_table(pa.Table.from_pylist([_record("b", [3, 99], [2.0, 5.0])]))

    assert adata.X.toarray().tolist() == [[2, 0, 0, 0]]
    assert "Dropped 1 gene entries" in caplog.text


@pytest.mark.parametrize(("record", "match"), [
    ({**_record("bad", [3], [1.0]), "expressions": [-2.0]}, "length mismatch for bad"),
    (_record("bad", [3], [1.5]), "Non-integer expression values in bad"),
])
def test_table_to_anndata_rejects_malformed_cells(record, match):
    table = pa.Table.from_pylist([_record("ok", [3], [1.0]), record])

    with pytest.raises(ValueError, match=match):
        _convert_table(table)


def test_table_to_anndata_names_missing_columns():
    table = pa.Table.from_pylist(RECORDS).drop_columns(["plate"])

    with pytest.raises(ValueError, match="plate"):
        _convert_table(table)


def test_table_to_anndata_rejects_empty_tables():
    with pytest.raises(ValueError, match="No cells"):
        _convert_table(pa.Table.from_pylist(RECORDS).slice(0, 0))

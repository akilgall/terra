"""Tahoe-100M (dissociated cancer cell lines + drug perturbations) data prep.

``stream`` locates and reads row groups of the Hub parquet shards, ``local``
reads a full local download, ``convert`` turns rows into an AnnData of raw
counts, ``labels`` adds the shared ``*_value`` label columns, and ``prepare``
is the command-line pipeline (``python -m terra.tahoe.prepare``).
"""

from .convert import (
    CONTROL_DRUG,
    cell_line_table,
    gene_lookup,
    records_to_anndata,
    restrict_to_vocab,
    sample_table,
    table_to_anndata,
)
from .labels import DOMAIN_SPATIAL, DOMAIN_TAHOE, UNMAPPED, LabelCodes, add_value_columns, lineage_of, load_lineage_map
from .local import (
    cell_counts,
    cells_filter,
    find_expression_files,
    metadata_file,
    open_expression_data,
    read_cells,
    read_obs,
    sample_per_group,
    select_cell_lines,
    select_samples,
)
from .stream import REPO_ID, RowGroupInfo, find_row_groups, list_shards, read_footer, read_row_groups, sample_number

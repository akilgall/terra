"""Tahoe-100M (dissociated cancer cell lines + drug perturbations) data prep.

``stream`` locates and reads row groups of the Hub parquet shards, ``convert``
turns records into an AnnData of raw counts, ``labels`` adds the shared
``*_value`` label columns, and ``prepare`` is the command-line pipeline
(``python -m terra.tahoe.prepare``).
"""

from .convert import CONTROL_DRUG, cell_line_table, gene_lookup, records_to_anndata, restrict_to_vocab, sample_table
from .labels import DOMAIN_SPATIAL, DOMAIN_TAHOE, UNMAPPED, LabelCodes, add_value_columns, lineage_of, load_lineage_map
from .stream import REPO_ID, RowGroupInfo, find_row_groups, list_shards, read_footer, read_row_groups, sample_number

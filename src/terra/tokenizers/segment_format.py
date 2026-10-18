"""Record formatting for the asymmetric cell-graph layout.

Legacy records split ``model_input_size`` evenly over the centre cell and its
non-empty neighbours. The asymmetric layout gives the centre cell
``seq_len_cell`` slots and each of ``n_neighs`` neighbours ``seq_len_neigh``
fixed slots (an empty neighbour is an all-padding slot, so later neighbours do
not move). Cell-only records hold just the centre segment.
"""

import numpy as np

from .tokenize import process_gene_expr, process_gene_tokens

__all__ = [
    "resolve_segment_budgets",
    "format_asymmetric_example",
]

_INPUT_KEYS = ('gene_tokens_cell', 'gene_expr_cell', 'gene_tokens_neighborhood',
               'gene_expr_neighborhood', 'seg_tokens_neighborhood')


def resolve_segment_budgets(model_input_size: int,
                            n_neighs: int | None,
                            seq_len_cell: int | None,
                            seq_len_neigh: int | None,
                            cell_only: bool,
                            ) -> tuple[int, int]:
    """
    Validate and return the (centre, neighbour) slot budgets.

    Parameters
    ----------
    model_input_size:
        Record length of a spatial record (centre + all neighbours).
    n_neighs:
        Number of neighbour segments.
    seq_len_cell:
        Centre-cell slots. Optional for cell-only records (defaults to
        `model_input_size`).
    seq_len_neigh:
        Slots per neighbour (defaults to `seq_len_cell`).
    cell_only:
        Whether records hold only the centre segment.

    Raises
    ------
    ValueError
        If the budgets are missing or do not add up to `model_input_size`.
    """
    if n_neighs is None or int(n_neighs) != n_neighs or n_neighs < 0:
        raise ValueError(
            "The asymmetric layout needs an integer n_neighs >= 0, got "
            f"{n_neighs!r}.")
    if seq_len_cell is None and not cell_only:
        raise ValueError(
            "seq_len_cell is required for the asymmetric layout (only "
            "cell-only records may default it to model_input_size).")
    centre = model_input_size if seq_len_cell is None else seq_len_cell
    neigh = centre if seq_len_neigh is None else seq_len_neigh
    if centre <= 0 or neigh <= 0:
        raise ValueError(
            f"Segment budgets must be positive, got seq_len_cell={centre}, "
            f"seq_len_neigh={neigh}.")
    if cell_only:
        if centre > model_input_size:
            raise ValueError(
                f"seq_len_cell {centre} exceeds model_input_size "
                f"{model_input_size}.")
    elif centre + int(n_neighs) * neigh != model_input_size:
        raise ValueError(
            f"model_input_size {model_input_size} must equal seq_len_cell "
            f"{centre} + n_neighs {n_neighs} x seq_len_neigh {neigh}.")
    return centre, neigh


def _neighbour_blocks(example: dict,
                      n_neighs: int,
                      seq_len_neigh: int,
                      token_dict: dict,
                      ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """One fixed-width block per neighbour slot (empty slot -> padding)."""
    genes = np.asarray(example['gene_tokens_neighborhood'])
    segments = np.asarray(example['seg_tokens_neighborhood'])
    expr = np.asarray(example['gene_expr_neighborhood'])
    token_blocks, expr_blocks, n_nonzero = [], [], 0
    for segment in range(2, n_neighs + 2):
        mask = segments == segment
        tokens, nnz = process_gene_tokens(
            genes[mask], seq_len_neigh, token_dict)
        token_blocks.append(tokens)
        expr_blocks.append(process_gene_expr(expr[mask], seq_len_neigh))
        n_nonzero += nnz
    return token_blocks, expr_blocks, n_nonzero


def format_asymmetric_example(example: dict,
                              seq_len_cell: int,
                              seq_len_neigh: int,
                              n_neighs: int,
                              cell_only: bool,
                              token_dict: dict,
                              include_special_tokens: bool,
                              ) -> dict:
    """
    Format one tokenized cell into a fixed-slot record.

    Follows the Hugging Face ``map`` contract of the legacy formatter: the
    intermediate per-segment columns are removed from `example` and the
    record columns (`gene_tokens`, `gene_expr`, `n_nonzero_tokens`) are added.
    """
    centre_tokens, n_nonzero = process_gene_tokens(
        example['gene_tokens_cell'], seq_len_cell, token_dict)
    token_blocks = [centre_tokens]
    expr_blocks = [process_gene_expr(example['gene_expr_cell'], seq_len_cell)]
    if not cell_only:
        neigh_tokens, neigh_expr, neigh_nonzero = _neighbour_blocks(
            example, n_neighs, seq_len_neigh, token_dict)
        token_blocks += neigh_tokens
        expr_blocks += neigh_expr
        n_nonzero += neigh_nonzero

    for key in _INPUT_KEYS:
        del example[key]
    example['gene_tokens'] = np.concatenate(token_blocks).astype(int)
    example['gene_expr'] = np.concatenate(expr_blocks).astype(float)
    example['n_nonzero_tokens'] = n_nonzero
    if include_special_tokens:
        for key in ('assay_value', 'species_value', 'tissue_value',
                    'gene_panel_value', 'batch_value'):
            example[key] = [example[key]]
    return example

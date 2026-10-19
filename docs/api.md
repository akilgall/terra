# API

The main user-facing API is exposed at the top level of the `terra` package. The
typical workflow is to harmonize an `AnnData`, tokenize it against a trained
model, and embed it:

```python
from terra import download_pretrained, harmonize_tokenize_embed_pipeline
```

## Inference

```{eval-rst}
.. currentmodule:: terra

.. autosummary::
    :toctree: generated

    harmonize_tokenize_embed_pipeline
    harmonize_adata
    tokenize_adata
    embed_dataset
    gene_embed_dataset
    infer
    get_gene_embed
    get_average_gene_embed
    perturb_dataset
    get_emd_distance
    get_spatial_score
```

## Supervised Finetuning

Adapt a pretrained model to a classification task via selective weight
unfreezing, with LoRA available as an alternative. See the
{doc}`supervised finetuning tutorial <notebooks/supervised_finetuning>`.

```{eval-rst}
.. currentmodule:: terra.training.finetune

.. autosummary::
    :toctree: generated

    finetune
```

```{eval-rst}
.. currentmodule:: terra.utils.nested_stratified_group_split

.. autosummary::
    :toctree: generated

    filter_dataset_by_cell_ids
```

## Self-supervised finetuning and Tahoe-100M alignment

Continue the JEPA objective on new data with LoRA, optionally across several
data streams (spatial, spatial without neighbours, Tahoe-100M) with alignment
losses. See {doc}`tahoe_alignment`.

```{eval-rst}
.. currentmodule:: terra.training.finetune_self_supervised

.. autosummary::
    :toctree: generated

    finetune_self_supervised
    prepare_finetuned_model
```

```{eval-rst}
.. currentmodule:: terra.evaluation.alignment

.. autosummary::
    :toctree: generated

    evaluate_alignment
    domain_mixing
    paired_view_similarity
    perturbation_separability
    perturbation_vector_consistency
    label_retention
    collapse_stats
```

### Tahoe-100M data

Read a full local Tahoe-100M download (parquet shards or a `datasets` cache),
select cells by well and cell line, and convert them to AnnData. See the
{doc}`Tahoe-100M exploration notebook <notebooks/tahoe100m_exploration>`.

```{eval-rst}
.. currentmodule:: terra.tahoe

.. autosummary::
    :toctree: generated

    open_expression_data
    metadata_file
    select_samples
    select_cell_lines
    cells_filter
    cell_counts
    read_obs
    sample_per_group
    read_cells
    table_to_anndata
```

## In-silico perturbation scoring

Quantify a perturbation's effect on the embeddings — per cell, or summarized
across cell populations (niches / cell types). These live in `terra.inference`
and require the optional `perturb` extra (`pip install "terra-st[perturb]"`).

```{eval-rst}
.. currentmodule:: terra.inference

.. autosummary::
    :toctree: generated

    infer_token_distance
    summarize_w1_by_label
    summarize_w2_by_label
    summarize_energy_by_label
    summarize_mmd_by_label
    summarize_cosine_sim_by_label
```

## Hugging Face Hub

```{eval-rst}
.. currentmodule:: terra

.. autosummary::
    :toctree: generated

    download_pretrained
    push_model_to_hub
```

## Data

Convenience reader used by the tutorials to load public 10x Genomics Xenium
samples.

```{eval-rst}
.. currentmodule:: terra.datasets

.. autosummary::
    :toctree: generated

    read_xenium_10x
```

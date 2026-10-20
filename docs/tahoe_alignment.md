# Aligning spatial data with Tahoe-100M

This guide fine-tunes a pretrained TERRA model so that three kinds of input share one
embedding space:

1. **spatial cells with their neighbours** (your whole-transcriptome spatial data),
2. **the same cells with the neighbours hidden**, and
3. **dissociated cells from [Tahoe-100M]**, the 100-million-cell drug-perturbation atlas
   (50 cancer cell lines, 379 drugs).

Once aligned, Tahoe drug responses can be compared with the cells in your tissue: a
Tahoe DMSO control of a lung cancer line should land near the tumour cells of a lung
section, and a drug's shift away from its controls can be read in the same space.

[Tahoe-100M]: https://huggingface.co/datasets/tahoebio/Tahoe-100M

## How it works

Every optimizer step draws a batch from one of three **streams** (by default a third of
the steps each):

| Stream kind | Input | Loss terms |
|---|---|---|
| `spatial_full` | centre cell + 10 neighbours | JEPA (masked latent prediction, as in pretraining) |
| `spatial_cell_only` | same cells, neighbours hidden | JEPA, **view consistency** (cell-only embedding → target encoder's full-view embedding), variance |
| `cell_only` | Tahoe cells | JEPA, **alignment** (mean matching + CORAL to spatial cells of the same lineage), variance |

The spatial cell-only stream fills a per-lineage **bank** with its current (online)
embeddings; Tahoe batches are
aligned to the bank entries of their lineage (e.g. lung lines ↔ `tumor` cells), so the two
domains never need to share a batch. By default only **DMSO controls** are aligned, so drug
effects are not trained away. Each term can start late and ramp up (`start_epoch`,
`ramp_epochs`).

JEPA runs on the masked view as in pretraining. The consistency, alignment and variance terms
use whole-cell embeddings (mean over the centre cell's genes, no masking), the same
representation `embed_dataset` returns, so they train what you use afterwards. At TERRA's
embedding scale the CORAL part of the alignment is small (about 1e-4), so in practice the
alignment matches lineage means.

### Sequence layout

TERRA-96M was pretrained with 11 segments × 256 genes. A whole-transcriptome centre cell
usually expresses more than 256 genes (a Tahoe cell has a median of about 1,300), so the
fine-tune can switch to an **asymmetric layout**: `seq_len_cell` genes for the centre cell
and `seq_len_neigh` per neighbour, e.g. 2048 + 10 × 256 = 4,608 tokens. Gene positions
restart at 1 in every segment, so the pretrained position table (2,816 rows) covers any
centre budget up to 2,816 without new parameters.

Genes that never occurred in pretraining have randomly initialised token embeddings.
With `extra_trainable: [token_embed]` and `token_embed_rows: unseen`, those rows are
trained while the pretrained rows stay fixed.

## 1. Tokenize the spatial data

Harmonize as usual, but keep genes that were rare or absent in pretraining
(`gene_occurrence_count_filter_value=0`) so their embeddings can be learned. Then tokenize
with the asymmetric layout:

```python
from terra import harmonize_adata, tokenize_adata

adata = harmonize_adata(adata,
                        gene_mapping_dict_file_path=f"{model_dir}/ensembl_dictionary.pkl",
                        gene_occurrence_count_file_path=f"{model_dir}/gene_count_dictionary.pkl",
                        gene_occurrence_count_filter_value=0)
dataset = tokenize_adata(adata, model_dir, "cache",
                         data_overrides={"seq_len_cell": 2048, "seq_len_neigh": 256,
                                         "model_input_size": 4608})
dataset.save_to_disk("spatial_ds")
```

## 2. Prepare a Tahoe-100M slice

`terra.tahoe.prepare tahoe` streams a balanced slice of control and treated cells for the
chosen organs from Hugging Face, converts it to AnnData, tokenizes it as centre-only
records and adds the label columns (`domain_value`, `lineage_value`, `cell_line_value`,
`drug_value`, `dose_value`, `is_control_value`):

```bash
python -m terra.tahoe.prepare tahoe --out tahoe_slice --model-dir $MODEL_DIR \
    --lineage-map configs/finetune/lineage_map.yaml --organs Lung --n-cells 200 \
    --seq-len-cell 2048 --seq-len-neigh 256 --model-input-size 4608
```

It writes `tahoe.h5ad`, `selection.json`, `tahoe_ds` and `label_codes.json`.

With a full local download, the
{doc}`Tahoe-100M exploration notebook <notebooks/tahoe100m_exploration>` selects
cells by organ, cell line, drug, dose and plate instead. It writes the same
`tahoe_ds` and `label_codes.json` with `terra.tahoe.prepare.write_tahoe_dataset`.

## 3. Label the spatial dataset

Map your cell types to the shared lineages in `configs/finetune/lineage_map.yaml` (add
the tumour labels of your own annotation) and add the same label columns, reusing the
Tahoe label codes:

```bash
python -m terra.tahoe.prepare spatial --dataset spatial_ds --adata spatial.h5ad \
    --cell-type-column cell_type --lineage-map configs/finetune/lineage_map.yaml \
    --label-codes tahoe_slice/label_codes.json --out spatial_labeled
```

The labelled dataset is written to `spatial_labeled/spatial_ds`.

## 4. Fine-tune

`configs/finetune/tahoe_alignment.yaml` documents every option. Fill in the paths and run:

```python
import yaml
from terra.training.finetune_self_supervised import finetune_self_supervised

with open("configs/finetune/tahoe_alignment.yaml") as f:
    args = yaml.safe_load(f)
run_dir = finetune_self_supervised(args, run_name="tahoe_alignment")
```

Checkpoints (`checkpoint_epoch_<N>.pt`) include the embedding bank, and
`finetune.resume_from` continues an interrupted run. The run directory gets its own
`model_config.yaml` describing the new layout.

### Watching for collapse

A falling alignment loss is not enough: it also falls if every cell is mapped to the same
point. Each stream therefore logs `<stream>/emb_std`, the mean per-dimension spread of the
cell embeddings it trains (about 0.2–0.3 for TERRA-96M at the start). It should stay in that
range; a drop towards 0 means the encoder is collapsing.

In tests on the CosMx stand-in, `lr: 1e-3` with LoRA alpha 256 / rank 16 (the setting of
the Xenium fine-tuning notebook) collapsed the encoder within 60 steps, even with JEPA
alone and with warmup. `lr: 1e-4` (the default here) or `peft_alpha: 32` did not, and gave
a lower JEPA loss. The variance floor `gamma` sits near the pretrained spread (0.2); VICReg's
usual 1.0 kept inflating the Tahoe embeddings.

## 5. Export and embed

```python
from datasets import load_from_disk
from terra import embed_dataset
from terra.training.finetune_self_supervised import prepare_finetuned_model

model = prepare_finetuned_model(run_dir, model_dir, "tahoe_aligned_model", use_peft=True)
spatial = embed_dataset(load_from_disk("spatial_labeled/spatial_ds"), model)
tahoe = embed_dataset(load_from_disk("tahoe_slice/tahoe_ds"), model, trim_padding=True)
```

The exported folder carries the asymmetric layout, so `tokenize_adata` and
`embed_dataset` use it automatically. `trim_padding=True` skips the empty neighbour slots
of centre-only records: on an M4 Max it embedded 200 Tahoe cells in 7 s instead of 24 s,
with identical embeddings.

## 6. Evaluate

`terra.evaluation.alignment.evaluate_alignment` reports, for one set of embeddings:

- `domain_mixing` / `domain_mixing_controls`: how well spatial and Tahoe cells of the shared
  lineages interleave (ratio 1 = fully mixed, 0 = separated);
- `paired_view`: agreement of each spatial cell's cell-only and full-context embeddings;
- `separability`: per-drug AUC of treated vs control Tahoe cells (drug effects preserved);
- `vector_consistency`: whether a drug shifts cells in the same direction in every line;
- `label_retention`: kNN balanced accuracy of spatial cell types;
- `collapse`: effective rank and per-dimension spread.

```python
import numpy as np
from terra.evaluation.alignment import evaluate_alignment

report = evaluate_alignment(
    np.vstack([spatial["cell_emb"], tahoe["cell_emb"]]), obs,
    context_emb=np.vstack([spatial["spatial_cell_emb"], tahoe["spatial_cell_emb"]]),
    columns={"domain": "domain_value", "lineage": "lineage_value",
             "is_control": "is_control_value", "drug": "drug_value",
             "cell_line": "cell_line_value"})
```

where `obs` holds those label columns (and `cell_type` for the spatial rows) in the same
row order. Compare the report with one computed from the pretrained model under the same
layout to see what the fine-tune changed.

`embed_dataset` uses the target encoder, an exponential moving average of the trained
encoder (`ema_momentum` 0.9995 → 1). It follows the trained encoder over a few thousand
steps, so after a short run the exported embeddings barely differ from the pretrained
ones. Judge short runs by `emb_std` and the training losses, not by the exported model.

## Hardware notes

- **Apple Silicon (MPS)**: use `use_bfloat16: true`, `grad_checkpointing: true` and
  `peft_dropout: 0` (gradient checkpointing on MPS cannot replay dropout; a non-zero value
  is set to 0 with a warning). Keep `num_workers` small. On an M4 Max with the 2048 + 10 ×
  256 layout: tokenizing 9,320 CosMx cells took 2 min, a fine-tuning step about 2.6 s
  (spatial batches of 8, Tahoe batches of 32), and embedding 2,000 spatial cells about
  4 min.
- **NVIDIA GPUs**: the 4,608-token spatial batches dominate memory; gradient checkpointing
  keeps activation memory roughly per block, and Tahoe batches are short after trimming,
  so they can be larger (`streams.tahoe.batch_size`).

## What has been tested

The pipeline has run end to end on MPS with a stand-in: a CosMx 1000-plex lung dataset
(9,320 cells) and a 200-cell Tahoe-100M slice of lung lines. With the defaults, 180 steps
raised kNN domain mixing between Tahoe controls and spatial tumour cells from 0 to 0.22
(1 = fully mixed), measured on the trained encoder (the exported target encoder lags behind
on a run this short). Cell-type kNN accuracy stayed at 0.79, with no collapse. The effective
rank of the embeddings fell from 13.7 to 7 (JEPA alone: 11), so check `collapse` in the
evaluation report. These are pipeline checks only. They have not been repeated on
whole-transcriptome spatial data or with full-length runs.

## Limitations

- Models with special tokens are not supported by the multi-stream loop.
- `perturb_dataset` and the legacy `infer` entry point still assume the uniform layout.
- The `laplacian` and RDA position encodings need a uniform layout.

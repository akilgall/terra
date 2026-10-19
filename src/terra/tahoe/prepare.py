r"""Prepare Tahoe-100M and spatial datasets for alignment fine-tuning.

Two subcommands share one set of label codes (``label_codes.json``):

``tahoe``: find a plate's DMSO-control row groups on the Hub (controls are a
plate's last wells; the same row groups also hold neighbouring treated wells),
read them, convert to raw counts, keep a lineage-matched subset balanced
between control and treated cells, tokenize as cell-only records and add the
``*_value`` label columns::

    python -m terra.tahoe.prepare tahoe --out DIR --model-dir BUNDLE \
        --lineage-map configs/finetune/lineage_map.yaml --n-cells 200 \
        --seq-len-cell 2048 --seq-len-neigh 256 --model-input-size 4608

``spatial``: add the same label columns to a tokenized spatial dataset::

    python -m terra.tahoe.prepare spatial --dataset DS --adata H5AD \
        --cell-type-column cell_type --lineage-map MAP --out DIR
"""

import argparse
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import datasets
import numpy as np
import pandas as pd

from .convert import CONTROL_DRUG, cell_line_table, gene_lookup, records_to_anndata, restrict_to_vocab, sample_table
from .labels import DOMAIN_SPATIAL, DOMAIN_TAHOE, LabelCodes, add_value_columns, load_lineage_map
from .stream import REPO_ID, find_row_groups, list_shards, read_footer, read_row_groups, sample_number

logger = logging.getLogger(__name__)


def control_samples(sample_metadata: pd.DataFrame, plate: str) -> set[int]:
    """Sample numbers of a plate's DMSO-control wells."""
    rows = sample_metadata[(sample_metadata["plate"] == plate)
                           & (sample_metadata["drug"] == CONTROL_DRUG)]
    return {sample_number(s) for s in rows["sample"]}


def balanced_subset(adata: ad.AnnData,
                    n_cells: int,
                    organs: list[str],
                    seed: int,
                    ) -> ad.AnnData:
    """
    Subset to `organs`, balanced between control and treated cells.

    Takes half controls and half treated where possible, topping up from the
    other group when one is short; keeps the original cell order.
    """
    pool = adata.obs[adata.obs["organ"].isin(organs).to_numpy()]
    controls = pool.index[pool["is_control"].to_numpy(dtype=bool)]
    treated = pool.index[~pool["is_control"].to_numpy(dtype=bool)]
    n_controls = min(len(controls), n_cells // 2)
    n_treated = min(len(treated), n_cells - n_controls)
    n_controls = min(len(controls), n_cells - n_treated)
    rng = np.random.default_rng(seed)
    picked = set(rng.choice(controls, n_controls, replace=False)) | \
        set(rng.choice(treated, n_treated, replace=False))
    if len(picked) < n_cells:
        logger.warning(f"Only {len(picked)} cells available for organs {organs}.")
    keep = adata.obs_names.isin(list(picked))
    return adata[keep].copy()


def _load_metadata(cache_dir: str | None) -> tuple[pd.DataFrame, ...]:
    def table(name: str) -> pd.DataFrame:
        return datasets.load_dataset(REPO_ID, name, split="train",
                                     cache_dir=cache_dir).to_pandas()
    return table("gene_metadata"), table("sample_metadata"), table("cell_line_metadata")


def _read_control_records(args, sample_metadata: pd.DataFrame) -> tuple[list[dict], list]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shards = list_shards(fs)
    plate = read_footer(shards[0], fs)[0].plate
    wanted = control_samples(sample_metadata, plate)
    logger.info(f"Locating {plate} DMSO wells {sorted(wanted)} in shard footers...")
    infos = find_row_groups(shards, wanted, lambda s: read_footer(s, fs), args.max_shards)
    if not infos:
        raise SystemExit(f"No row groups with {plate} controls in the first "
                         f"{args.max_shards} shards; raise --max-shards.")
    chosen = infos[:args.max_row_groups]
    logger.info(f"Reading {len(chosen)} row group(s).")
    return read_row_groups(chosen, fs), chosen


def _layout_overrides(args) -> dict:
    if args.seq_len_cell is None:
        return {}
    overrides = {"seq_len_cell": args.seq_len_cell, "seq_len_neigh": args.seq_len_neigh}
    if args.model_input_size is not None:
        overrides["model_input_size"] = args.model_input_size
    return overrides


def prepare_tahoe(args) -> None:
    """Run the `tahoe` subcommand."""
    from terra.inference.tokenize import tokenize_adata

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gene_meta, sample_meta, cell_line_meta = _load_metadata(args.cache_dir)
    records, chosen = _read_control_records(args, sample_meta)
    lookup, var = gene_lookup(gene_meta)
    adata = records_to_anndata(records, lookup, var, sample_table(sample_meta),
                               cell_line_table(cell_line_meta))
    adata = balanced_subset(adata, args.n_cells, args.organs.split(","), args.seed)
    adata.write_h5ad(out / "tahoe.h5ad")
    with open(out / "selection.json", "w") as f:
        json.dump([asdict(info) for info in chosen], f, indent=1)
    logger.info(f"Wrote {adata.n_obs} cells "
                f"({int(adata.obs['is_control'].sum())} controls) to {out / 'tahoe.h5ad'}.")
    if args.stop_after == "anndata":
        return
    with open(Path(args.model_dir) / "token_dictionary.pkl", "rb") as f:
        adata = restrict_to_vocab(adata, pickle.load(f))
    dataset = tokenize_adata(adata, args.model_dir, str(out / "cache"), nproc=1,
                             processing_mode="sequential", use_generator=False,
                             cell_only=True, data_overrides=_layout_overrides(args))
    lineage_map = load_lineage_map(args.lineage_map)
    codes = LabelCodes.from_values(lineage_map, cell_lines=adata.obs["cell_line_id"],
                                   drugs=adata.obs["drug"])
    dataset = add_value_columns(dataset, adata.obs, codes, lineage_map, domain=DOMAIN_TAHOE,
                                lineage_column="organ", lineage_source="tahoe_organs")
    dataset.save_to_disk(str(out / "tahoe_ds"))
    codes.save(out / "label_codes.json")
    logger.info(f"Saved tokenized dataset to {out / 'tahoe_ds'}.")


def label_spatial(args) -> None:
    """Run the `spatial` subcommand."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lineage_map = load_lineage_map(args.lineage_map)
    codes = (LabelCodes.load(args.label_codes) if args.label_codes
             else LabelCodes.from_values(lineage_map, cell_lines=[], drugs=[]))
    obs = ad.read_h5ad(args.adata, backed="r").obs
    if "cell_id" not in obs:
        obs = obs.assign(cell_id=obs.index.astype(str))
    dataset = datasets.load_from_disk(args.dataset)
    dataset = add_value_columns(dataset, obs, codes, lineage_map, domain=DOMAIN_SPATIAL,
                                lineage_column=args.cell_type_column,
                                lineage_source="spatial_cell_types")
    dataset.save_to_disk(str(out / "spatial_ds"))
    codes.save(out / "label_codes.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    tahoe = sub.add_parser("tahoe", help="Tahoe-100M control/treated subset")
    tahoe.add_argument("--out", required=True)
    tahoe.add_argument("--model-dir", required=True)
    tahoe.add_argument("--lineage-map", required=True)
    tahoe.add_argument("--n-cells", type=int, default=200)
    tahoe.add_argument("--organs", default="Lung", help="Comma-separated Tahoe organs.")
    tahoe.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache.")
    tahoe.add_argument("--max-shards", type=int, default=12)
    tahoe.add_argument("--max-row-groups", type=int, default=2)
    tahoe.add_argument("--seq-len-cell", type=int, default=None)
    tahoe.add_argument("--seq-len-neigh", type=int, default=None)
    tahoe.add_argument("--model-input-size", type=int, default=None)
    tahoe.add_argument("--seed", type=int, default=0)
    tahoe.add_argument("--stop-after", choices=["anndata", "tokenize"], default="tokenize")

    spatial = sub.add_parser("spatial", help="label a tokenized spatial dataset")
    spatial.add_argument("--dataset", required=True)
    spatial.add_argument("--adata", required=True)
    spatial.add_argument("--cell-type-column", required=True)
    spatial.add_argument("--lineage-map", required=True)
    spatial.add_argument("--label-codes", default=None)
    spatial.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if args.command == "tahoe":
        prepare_tahoe(args)
    else:
        label_spatial(args)


if __name__ == "__main__":
    main()

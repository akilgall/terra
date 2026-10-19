"""Tiny TERRA bundles and datasets for fast end-to-end tests.

Importable from any test (``tests/conftest.py`` puts ``tests/`` on
``sys.path``).
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

N_GENES = 10
NORMS = {"rank_cell_norm_method": None, "rank_gene_norm_method": None,
         "rank_count_norm_method": None, "count_cell_norm_method": None,
         "count_gene_norm_method": None, "count_count_norm_method": "shifted_log"}
LINEAGE_MAP = {"lineages": {"lung": {"tahoe_organs": ["Lung"],
                                     "spatial_cell_types": ["tumor"]}}}
META = {"gt_type": "combined", "count_encoding": "mlp", "n_value_bins": 100,
        "cell_pos_enc": "coord", "enc_emb_dim": 32, "enc_depth": 2, "pred_emb_dim": 16,
        "pred_depth": 2, "num_heads": 4, "mlp_ratio": 2.0, "use_flash_attention": True,
        "api_version": "v3", "predict_gene": True, "pos_learnable": False,
        "special_tokens": [], "use_bfloat16": False, "restrict_special_attention": False}
SEEN_GENES = 5  # genes listed in gene_count_dictionary.pkl ("seen in pretraining")


def skip_without_squidpy() -> None:
    """Skip the calling test when squidpy, which builds the spatial neighbour
    graph, cannot be imported (e.g. squidpy 1.8 with a scanpy 1.13 pre-release)."""
    pytest.importorskip("squidpy")


def genes() -> list[str]:
    """Ensembl ids of the tiny vocabulary."""
    return [f"ENSG{i:011d}" for i in range(1, N_GENES + 1)]


def make_bundle(folder: Path) -> Path:
    """Pretrained-style bundle with a uniform 3 x 4 layout (seq_len 12)."""
    folder.mkdir(parents=True)
    token_dict = {"<pad>": 0, "<mask>": 1}
    token_dict.update({g: i + 2 for i, g in enumerate(genes())})
    with open(folder / "token_dictionary.pkl", "wb") as f:
        pickle.dump(token_dict, f)
    with open(folder / "gene_count_dictionary.pkl", "wb") as f:
        pickle.dump(dict.fromkeys(genes()[:SEEN_GENES], 100), f)
    data = {"tokenizer_type": "cell_graph", "seq_len_cell": 4, "seq_len_neighborhood": 8,
            "n_segments": 3, "n_neighs": 2, "model_input_size": 12,
            "sep_gene_tokens_neb": False, **NORMS}
    config = {"data": data, "meta": META,
              "mask": {"n_targets": 1, "n_contexts": 1, "per_block_mask_ratio": 0.5}}
    with open(folder / "model_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    from terra.utils.helper import init_model

    torch.manual_seed(0)
    encoder, predictor = init_model(
        gt_type="combined", count_encoding="mlp", n_value_bins=100, cell_pos_enc="coord",
        device="cpu", vocab_size=len(token_dict), seq_len=12, n_special_tokens=0,
        n_segments=3, n_special_values=0, enc_emb_dim=32, enc_depth=2, pred_emb_dim=16,
        pred_depth=2, num_heads=4, mlp_ratio=2.0, use_flash_attention=True,
        api_version="v3", predict_gene=True, mlp_bias=True)
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                "target_encoder": encoder.state_dict(), "epoch": 1},
               folder / "model_checkpoint.pt")
    return folder


def make_adata(n_cells: int, spatial: bool, seed: int):
    """Poisson counts over the tiny vocabulary; spatial cells get coordinates
    and a cell type, others Tahoe-style drug metadata."""
    import anndata

    rng = np.random.default_rng(seed)
    adata = anndata.AnnData(X=rng.poisson(2.0, size=(n_cells, N_GENES)).astype(np.float32))
    adata.var_names = genes()
    adata.var["ensembl_id"] = genes()
    adata.obs_names = [f"{'s' if spatial else 't'}{i}" for i in range(n_cells)]
    adata.obs["cell_id"] = adata.obs_names.to_list()
    if spatial:
        adata.obsm["spatial"] = rng.uniform(0, 50, size=(n_cells, 2)).astype(np.float32)
        adata.obs["cell_type"] = ["tumor", "fibroblast"] * (n_cells // 2)
    else:
        adata.obs["organ"] = "Lung"
        adata.obs["is_control"] = [True, False] * (n_cells // 2)
        adata.obs["drug"] = ["DMSO_TF", "Drug"] * (n_cells // 2)
        adata.obs["dose_uM"] = [0.0, 0.5] * (n_cells // 2)
        adata.obs["cell_line_id"] = "CVCL_0023"
    return adata


def tokenize(bundle: Path, adata, cell_only: bool, out: Path, **layout):
    """Tokenize with the asymmetric tokenizer (default 6 + 2 x 3), add the
    label columns and save; returns the dataset path."""
    from terra.tahoe.labels import DOMAIN_SPATIAL, DOMAIN_TAHOE, LabelCodes, add_value_columns
    from terra.tokenizers.cell_tokenizers import CellGraphTokenizer

    if not cell_only:
        skip_without_squidpy()
    layout = {"model_input_size": 12, "seq_len_cell": 6, "seq_len_neigh": 3, **layout}
    tk = CellGraphTokenizer(nproc=1, n_neighs=2,
                            token_dictionary_file_path=str(bundle / "token_dictionary.pkl"),
                            include_special_tokens=False, cell_only=cell_only,
                            **layout, **NORMS)
    ds = tk._create_dataset(tk._tokenize_adata(adata=adata.copy()), use_generator=False)
    codes = LabelCodes.from_values(LINEAGE_MAP, cell_lines=["CVCL_0023"],
                                   drugs=["DMSO_TF", "Drug"])
    if cell_only:
        ds = add_value_columns(ds, adata.obs, codes, LINEAGE_MAP, DOMAIN_TAHOE, "organ",
                               "tahoe_organs")
    else:
        ds = add_value_columns(ds, adata.obs, codes, LINEAGE_MAP, DOMAIN_SPATIAL, "cell_type",
                               "spatial_cell_types")
    ds.save_to_disk(str(out))
    return str(out)

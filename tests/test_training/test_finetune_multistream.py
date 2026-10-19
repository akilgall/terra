"""End-to-end smoke test of multi-stream alignment fine-tuning on a tiny bundle,
plus LoRA merge scaling for the exported model."""

import numpy as np
import pytest
import torch
import yaml

datasets = pytest.importorskip("datasets")
anndata = pytest.importorskip("anndata")

from tiny_terra import NORMS, make_adata, make_bundle, skip_without_squidpy, tokenize  # noqa: E402

from terra.tokenizers.cell_tokenizers import CellGraphTokenizer  # noqa: E402
from terra.training.finetune_self_supervised import (  # noqa: E402
    finetune_self_supervised,
    merge_lora_weights,
    prepare_finetuned_model,
)
from terra.training.lora import apply_lora  # noqa: E402
from terra.training.multistream_finetune import (  # noqa: E402
    ema_update,
    resolve_layout,
    resolve_n_positions,
    run_model_config,
)
from terra.utils.helper import init_model  # noqa: E402


def _args(tmp_path, bundle, spatial_path, tahoe_path, use_peft=True, device="cpu"):
    return {
        "model": {"pretrained_checkpoint_path": str(bundle),
                  "finetune_checkpoint_path": str(tmp_path / "runs")},
        "data": {
            "seq_len_cell": 6, "seq_len_neigh": 3, "batch_size": 4, "num_workers": 0,
            "pin_memory": False, "epoch_batches": 6,
            "streams": {
                "spatial": {"kind": "spatial_full", "paths": [spatial_path]},
                "spatial_cell": {"kind": "spatial_cell_only", "paths": [spatial_path]},
                "tahoe": {"kind": "cell_only", "paths": [tahoe_path],
                          "tokenized_seq_len_cell": 6},
            }},
        "finetune": {
            "device": str(device), "num_epochs": 2, "lr": 1e-3, "start_lr": 1e-4, "final_lr": 1e-5,
            "warmup_epochs": 1, "weight_decay": 0.04, "final_weight_decay": 0.4,
            "ema_momentum": 0.99, "final_ema_momentum": 1.0, "clip_grad": 2.0,
            "use_bfloat16": False, "use_peft": use_peft, "peft_method": "lora",
            "peft_rank": 4, "peft_alpha": 16, "peft_dropout": 0.0, "peft_bias": "none",
            "peft_target_modules": ["qkv", "proj", "fc1", "fc2"], "save_every": 1,
            "grad_checkpointing": True, "extra_trainable": ["token_embed"],
            "token_embed_rows": "unseen", "token_embed_lr_mult": 5.0,
            "multistream": {
                "view_consistency": {"weight": 1.0},
                "alignment": {"weight": 1.0, "min_cells": 2, "bank_size": 64,
                              "lambda_mean": 1.0, "control_only": True},
                "variance": {"weight": 0.1}}},
    }


@pytest.fixture
def setup(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    spatial = tokenize(bundle, make_adata(16, True, 0), False, tmp_path / "spatial_ds")
    tahoe = tokenize(bundle, make_adata(16, False, 1), True, tmp_path / "tahoe_ds")
    return tmp_path, bundle, spatial, tahoe


@pytest.mark.parametrize("use_peft", [True, False])
def test_multistream_finetune_runs_and_exports(setup, use_peft, device):
    tmp_path, bundle, spatial, tahoe = setup

    run_dir = finetune_self_supervised(
        _args(tmp_path, bundle, spatial, tahoe, use_peft, device),
        run_name=f"smoke_{use_peft}_{device.type}")

    checkpoint = torch.load(f"{run_dir}/checkpoint_epoch_2.pt", map_location="cpu")
    assert np.isfinite(checkpoint["loss"])
    assert checkpoint["multistream"]["global_step"] == 12
    assert set(checkpoint["multistream"]["bank"]["banks"]) == {0}
    with open(f"{run_dir}/model_config.yaml") as f:
        run_config = yaml.safe_load(f)
    assert run_config["data"]["seq_len_cell"] == 6
    assert run_config["data"]["seq_len_neigh"] == 3
    assert run_config["data"]["model_input_size"] == 12
    assert run_config["meta"]["n_positions"] == 12

    exported = prepare_finetuned_model(run_dir, str(bundle), str(tmp_path / f"out_{use_peft}"),
                                       use_peft=use_peft)
    exported_weights = torch.load(f"{exported}/model_checkpoint.pt", map_location="cpu")
    assert not any("lora_" in key for key in exported_weights["encoder"])
    with open(f"{exported}/model_config.yaml") as f:
        assert yaml.safe_load(f)["data"]["seq_len_neigh"] == 3


def test_token_embed_trains_only_unseen_rows(setup):
    tmp_path, bundle, spatial, tahoe = setup
    before = torch.load(bundle / "model_checkpoint.pt", map_location="cpu")["encoder"]

    run_dir = finetune_self_supervised(_args(tmp_path, bundle, spatial, tahoe), run_name="emb")

    after = torch.load(f"{run_dir}/checkpoint_epoch_2.pt", map_location="cpu")["encoder"]
    key_before = next(k for k in before if k.endswith("token_embed.weight"))
    key_after = next(k for k in after if k.endswith("token_embed.weight"))
    changed = (after[key_after] - before[key_before]).abs().sum(dim=1) > 0
    seen = list(range(2, 7))  # the 5 genes in gene_count_dictionary
    assert not changed[seen].any()
    assert changed[7:].any()


def test_merge_lora_weights_applies_scaling():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=8, target_modules=["0"],
                                            lora_dropout=0.0))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)

    merged = merge_lora_weights(model.state_dict(), scaling=8 / 2)
    expected = model.merge_and_unload().state_dict()

    torch.testing.assert_close(merged["0.weight"], expected["0.weight"])


def test_resume_continues_from_saved_step(setup):
    tmp_path, bundle, spatial, tahoe = setup
    args = _args(tmp_path, bundle, spatial, tahoe)
    first = finetune_self_supervised({**args, "finetune": {**args["finetune"], "num_epochs": 1}},
                                     run_name="first")
    resumed_args = {**args, "finetune": {**args["finetune"],
                                         "resume_from": f"{first}/checkpoint_epoch_1.pt"}}

    run_dir = finetune_self_supervised(resumed_args, run_name="resumed")

    checkpoint = torch.load(f"{run_dir}/checkpoint_epoch_2.pt", map_location="cpu")
    assert checkpoint["multistream"]["global_step"] == 12
    assert not (tmp_path / "runs" / "resumed" / "checkpoint_epoch_1.pt").exists()


def test_legacy_args_without_streams_take_legacy_path(setup, monkeypatch):
    tmp_path, bundle, spatial, tahoe = setup
    args = _args(tmp_path, bundle, spatial, tahoe)
    legacy = {**args, "data": {k: v for k, v in args["data"].items() if k != "streams"}}

    with pytest.raises(KeyError, match="finetune_dataset"):
        finetune_self_supervised(legacy, run_name="legacy")


@pytest.mark.parametrize("overrides, lengths", [
    ({}, (4, 4, 4)),
    ({"seq_len_cell": 6, "seq_len_neigh": 3}, (6, 3, 3)),
    ({"seq_len_cell": 6, "n_neighs": 1}, (6, 6)),
])
def test_resolve_layout(overrides, lengths):
    data = {"seq_len_cell": 4, "seq_len_neighborhood": 8, "n_neighs": 2, "n_segments": 3}

    assert resolve_layout(data, overrides).lengths == lengths


def test_n_positions_keeps_pretrained_table_unless_a_segment_is_longer():
    config = {"data": {"seq_len_cell": 4, "seq_len_neighborhood": 8},
              "meta": {"special_tokens": []}}

    assert resolve_n_positions(config, resolve_layout({**config["data"], "n_neighs": 2},
                                                      {"seq_len_cell": 10})) == 12
    assert resolve_n_positions(config, resolve_layout({**config["data"], "n_neighs": 2},
                                                      {"seq_len_cell": 20})) == 20
    config["meta"]["n_positions"] = 30
    assert resolve_n_positions(config, resolve_layout({**config["data"], "n_neighs": 2},
                                                      {})) == 30


def test_run_model_config_describes_layout_without_mutating_input():
    config = {"data": {"seq_len_cell": 4, "seq_len_neighborhood": 8, "n_neighs": 2,
                       "n_segments": 3, "model_input_size": 12}, "meta": {"x": 1}}
    layout = resolve_layout(config["data"], {"seq_len_cell": 8, "seq_len_neigh": 2})

    out = run_model_config(config, layout, n_positions=12)

    assert out["data"]["seq_len_neighborhood"] == 4
    assert out["data"]["model_input_size"] == 12
    assert out["meta"] == {"x": 1, "n_positions": 12}
    assert config["data"]["seq_len_cell"] == 4 and "n_positions" not in config["meta"]


def test_run_model_config_keeps_uniform_layouts_legacy():
    config = {"data": {"seq_len_cell": 4, "seq_len_neighborhood": 8, "n_neighs": 2,
                       "n_segments": 3, "model_input_size": 12}, "meta": {}}

    out = run_model_config(config, resolve_layout(config["data"], {}), n_positions=12)

    assert "seq_len_neigh" not in out["data"]
    assert out["data"] == config["data"]


def _tiny_encoder():
    encoder, _ = init_model(
        gt_type="combined", count_encoding="mlp", n_value_bins=100, cell_pos_enc="coord",
        device="cpu", vocab_size=12, seq_len=12, n_special_tokens=0, n_segments=3,
        n_special_values=0, enc_emb_dim=16, enc_depth=1, pred_emb_dim=8, pred_depth=1,
        num_heads=2, mlp_ratio=2.0, use_flash_attention=True, api_version="v3",
        predict_gene=True, mlp_bias=True)
    return encoder


def test_ema_update_with_lora_tracks_merged_weights():
    import copy

    torch.manual_seed(0)
    target = _tiny_encoder()
    online = apply_lora(copy.deepcopy(target), peft_rank=2, peft_alpha=8, peft_dropout=0.0,
                        peft_target_modules=["qkv", "proj", "fc1", "fc2"])
    for name, param in online.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)

    ema_update(online, target, momentum=0.0, lora_scale=8 / 2)

    expected = copy.deepcopy(online).merge_and_unload().state_dict()
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_ema_update_without_lora_interpolates():
    torch.manual_seed(0)
    online, target = _tiny_encoder(), _tiny_encoder()
    before = {k: v.clone() for k, v in target.state_dict().items()}

    ema_update(online, target, momentum=0.25)

    key = next(k for k in before if k.endswith("qkv.weight"))
    torch.testing.assert_close(target.state_dict()[key],
                               0.25 * before[key] + 0.75 * online.state_dict()[key])


def test_example_config_is_valid():
    from pathlib import Path

    from terra.training.multistream import parse_streams
    from terra.training.stream_objective import TERMS, MultiStreamObjective

    path = Path(__file__).parents[2] / "configs" / "finetune" / "tahoe_alignment.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)

    specs = parse_streams(config)
    objective = MultiStreamObjective(config["finetune"]["multistream"])

    assert [s.kind for s in specs] == ["spatial_full", "spatial_cell_only", "cell_only"]
    assert set(config["finetune"]["multistream"]) <= set(TERMS)
    assert objective.min_cells == 16 and objective.control_only
    assert resolve_layout({"seq_len_cell": 256, "seq_len_neighborhood": 2560, "n_neighs": 10},
                          config["data"]).total == 4608


def test_legacy_loop_still_trains_and_exports(tmp_path):
    skip_without_squidpy()
    bundle = make_bundle(tmp_path / "bundle")
    tk = CellGraphTokenizer(nproc=1, model_input_size=12, n_neighs=2,
                            token_dictionary_file_path=str(bundle / "token_dictionary.pkl"),
                            include_special_tokens=False, **NORMS)
    tk._create_dataset(tk._tokenize_adata(adata=make_adata(16, True, 0)),
                       use_generator=False).save_to_disk(str(tmp_path / "legacy_ds"))
    args = _args(tmp_path, bundle, "", "")
    args["data"] = {"finetune_dataset": [str(tmp_path / "legacy_ds")], "batch_size": 4,
                    "num_workers": 0, "pin_memory": False, "sample_gene_masks": True}

    run_dir = finetune_self_supervised(args, run_name="legacy")

    checkpoint = torch.load(f"{run_dir}/checkpoint_epoch_2.pt", map_location="cpu")
    assert "multistream" not in checkpoint and np.isfinite(checkpoint["loss"])
    exported = prepare_finetuned_model(run_dir, str(bundle), str(tmp_path / "legacy_out"),
                                       use_peft=True)
    with open(f"{exported}/model_config.yaml") as f:
        assert yaml.safe_load(f)["data"]["seq_len_cell"] == 4

"""Optimizer parameter-group overrides, scheduler lr scaling and row masks.

Fine-tuning on Tahoe-100M trains the gene-token embedding: 12,422 of TERRA's
21,952 gene rows never occurred in pretraining and are random init. They get
their own AdamW group (no weight decay, larger learning rate via
``lr_scale``), and a gradient hook keeps the pretrained rows frozen.
"""

import torch
import torch.nn as nn

from terra.utils.helper import init_opt, register_row_grad_mask, unseen_gene_rows
from terra.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(6, 4)
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _opt(**kwargs):
    torch.manual_seed(0)
    encoder, predictor = _Tiny(), _Tiny()
    optimizer, _, scheduler, wd_scheduler = init_opt(
        encoder, predictor, iterations_per_epoch=10, start_lr=1e-4, ref_lr=1e-3,
        warmup=1, num_epochs=2, wd=0.04, final_wd=0.4, **kwargs)
    return encoder, predictor, optimizer, scheduler, wd_scheduler


def _ids(params) -> set[int]:
    return {id(p) for p in params}


def test_default_groups_unchanged():
    encoder, predictor, optimizer, _, _ = _opt()

    def decayed(m):
        return _ids(p for n, p in m.named_parameters() if "bias" not in n and p.dim() > 1)

    def not_decayed(m):
        return _ids(p for n, p in m.named_parameters() if "bias" in n or p.dim() == 1)

    groups = optimizer.param_groups
    assert len(groups) == 4
    assert [_ids(g["params"]) for g in groups] == [
        decayed(encoder), decayed(predictor), not_decayed(encoder), not_decayed(predictor)]
    assert [g.get("WD_exclude", False) for g in groups] == [False, False, True, True]
    assert all("lr_scale" not in g for g in groups)


def test_override_moves_matching_params_into_own_group():
    encoder, predictor, optimizer, _, _ = _opt(
        param_group_overrides=[{"match": "token_embed", "lr_mult": 5.0, "weight_decay": 0.0}])

    groups = optimizer.param_groups
    override = groups[-1]
    expected = _ids([encoder.token_embed.weight, predictor.token_embed.weight])
    assert len(groups) == 5
    assert _ids(override["params"]) == expected
    assert override["lr_scale"] == 5.0
    assert override["weight_decay"] == 0.0
    assert override["WD_exclude"] is True
    for group in groups[:-1]:
        assert not (_ids(group["params"]) & expected)


def test_lr_schedule_applies_lr_scale():
    _, _, optimizer, scheduler, wd_scheduler = _opt(
        param_group_overrides=[{"match": "token_embed", "lr_mult": 5.0, "weight_decay": 0.0}])

    new_lr = scheduler.step()
    wd_scheduler.step()

    assert [g["lr"] for g in optimizer.param_groups] == [new_lr] * 4 + [5.0 * new_lr]
    assert optimizer.param_groups[-1]["weight_decay"] == 0.0


def test_override_with_weight_decay_follows_wd_schedule():
    _, _, optimizer, _, wd_scheduler = _opt(
        param_group_overrides=[{"match": "proj", "lr_mult": 1.0, "weight_decay": 0.04}])

    new_wd = wd_scheduler.step()

    assert optimizer.param_groups[-1]["WD_exclude"] is False
    assert optimizer.param_groups[-1]["weight_decay"] == new_wd


def test_schedulers_ignore_missing_lr_scale():
    param = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([{"params": [param]}], lr=0.1)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=2, start_lr=0.0,
                                     ref_lr=1.0, T_max=10)
    CosineWDSchedule(optimizer, ref_wd=0.1, T_max=10).step()

    assert scheduler.step() == optimizer.param_groups[0]["lr"]


def test_row_grad_mask_protects_pretrained_rows():
    embedding = nn.Embedding(5, 3)
    handle = register_row_grad_mask(embedding, trainable_rows=torch.tensor([1, 3]))

    embedding(torch.tensor([0, 1, 2, 3, 4])).sum().backward()

    grad = embedding.weight.grad
    assert torch.all(grad[[0, 2, 4]] == 0)
    assert torch.all(grad[[1, 3]] == 1)

    handle.remove()
    embedding.weight.grad = None
    embedding(torch.tensor([0])).sum().backward()
    assert torch.all(embedding.weight.grad[0] == 1)


def test_unseen_gene_rows():
    token_dict = {"<pad>": 0, "<mask>": 1, "ENSG01": 2, "ENSG02": 3, "ENSG03": 4,
                  "spv_batch": 5}
    gene_counts = {"ENSG02": 10}

    rows = unseen_gene_rows(token_dict, gene_counts)

    assert rows.tolist() == [2, 4]
    assert rows.dtype == torch.long

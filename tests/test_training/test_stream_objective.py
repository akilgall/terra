"""Per-stream objective (``terra.training.stream_objective``) on a tiny model."""

import math

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.masks.block_masking import BlockMaskCollator  # noqa: E402
from terra.training.multistream import MultiStreamCollator, StreamTaggedDataset  # noqa: E402
from terra.training.stream_objective import (  # noqa: E402
    MultiStreamObjective,
    StreamModels,
    online_cells,
    target_cells,
    target_tokens,
)
from terra.utils.helper import init_model  # noqa: E402

N_ROWS = 8
KINDS = ["spatial_full", "spatial_cell_only", "cell_only"]


def _rows(seed):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(2, 20, (N_ROWS, 12), generator=g)
    tokens[:, 10:] = 0  # short last neighbour
    return {"gene_tokens": tokens.tolist(),
            "gene_expr": torch.rand(N_ROWS, 12, generator=g).tolist(),
            "rel_x_coord": torch.rand(N_ROWS, 3, generator=g).tolist(),
            "rel_y_coord": torch.rand(N_ROWS, 3, generator=g).tolist(),
            "n_nonzero_tokens": [10] * N_ROWS,
            "lineage_value": [0, 0, 0, 0, 0, 0, -1, 1],
            "is_control_value": [1, 1, 1, 1, 0, 0, 1, 1]}


def _batch(kind, seed=0):
    ds = datasets.Dataset.from_dict(_rows(seed)).with_format("torch")
    cells = CellGraphDataset(gt_type="combined", cell_pos_enc="coord", dataset=ds,
                             vocab_size=20, seq_len_cell=6, seq_len_neighborhood=6,
                             seq_len_neigh=3, special_tokens=[], sampling_strategy=None)
    masker = BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=3, seq_len_cell=6,
                               seq_len_neighborhood=6, seq_len_neigh=3, n_special_tokens=0,
                               per_block_mask_ratio=0.5)
    tagged = StreamTaggedDataset(cells, KINDS.index(kind))
    return MultiStreamCollator(masker, KINDS)([tagged[i] for i in range(N_ROWS)])


@pytest.fixture
def models():
    torch.manual_seed(0)
    kwargs = {"gt_type": "combined", "count_encoding": "mlp", "n_value_bins": 100,
              "cell_pos_enc": "coord", "device": "cpu", "vocab_size": 20, "seq_len": 12,
              "n_special_tokens": 0, "n_segments": 3, "n_special_values": 0,
              "enc_emb_dim": 16, "enc_depth": 1, "pred_emb_dim": 8, "pred_depth": 1,
              "num_heads": 2, "mlp_ratio": 2.0, "use_flash_attention": True,
              "api_version": "v3", "predict_gene": True, "mlp_bias": True,
              "seq_len_cell": 6, "n_positions": 12}
    encoder, predictor = init_model(**kwargs)
    target, _ = init_model(**kwargs)
    target.load_state_dict(encoder.state_dict())
    target.requires_grad_(False)
    return StreamModels(encoder, predictor, target.eval())


def _objective(**alignment):
    return MultiStreamObjective({"alignment": {"min_cells": 2, "bank_size": 16, **alignment}})


@pytest.mark.parametrize("kind", KINDS)
def test_losses_are_finite_and_train_online_models_only(models, kind):
    objective = _objective()
    objective.bank.push(torch.randn(4, 16), torch.zeros(4, dtype=torch.long))
    target_before = {k: v.clone() for k, v in models.target_encoder.state_dict().items()}

    loss, logs = objective(_batch(kind), models, epoch=0.0)
    loss.backward()

    assert torch.isfinite(loss) and all(math.isfinite(v) for v in logs.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in models.encoder.parameters())
    assert all(p.grad is None for p in models.target_encoder.parameters())
    for key, value in models.target_encoder.state_dict().items():
        torch.testing.assert_close(value, target_before[key])


def test_spatial_cell_only_fills_bank_with_mapped_lineages(models):
    objective = _objective()

    _, logs = objective(_batch("spatial_cell_only"), models, epoch=0.0)

    assert sorted(objective.bank.labels()) == [0, 1]
    assert len(objective.bank.get(0)) == 6
    assert set(logs) == {"jepa", "view_consistency", "variance", "emb_std"}


def test_cell_only_alignment_needs_bank_and_uses_controls(models):
    objective = _objective()
    _, empty = objective(_batch("cell_only"), models, epoch=0.0)
    objective(_batch("spatial_cell_only", seed=1), models, epoch=0.0)

    _, filled = objective(_batch("cell_only"), models, epoch=0.0)

    assert set(empty) == {"jepa", "alignment", "variance", "align_lineages", "emb_std"}
    assert empty["alignment"] == 0.0 and empty["align_lineages"] == 0
    assert filled["alignment"] > 0.0
    assert filled["align_lineages"] == 1  # lineage 0: 4 controls; lineage 1: 1 cell


def test_alignment_on_all_cells_when_not_control_only(models):
    objective = _objective(control_only=False, min_cells=6)
    objective(_batch("spatial_cell_only", seed=1), models, epoch=0.0)

    _, logs = objective(_batch("cell_only"), models, epoch=0.0)

    assert logs["align_lineages"] == 1  # 6 lineage-0 cells only without the control filter


def test_online_cells_match_target_cells_for_equal_weights(models):
    view = _batch("cell_only").jepa

    online, online_valid = online_cells(models, view)
    target, target_valid = target_cells(target_tokens(models.target_encoder, view), view)

    torch.testing.assert_close(online, target)
    assert torch.equal(online_valid, target_valid)


def test_bank_holds_online_whole_cell_embeddings(models):
    with torch.no_grad():  # online != target, as during training
        for p in models.encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
    objective = _objective()
    batch = _batch("spatial_cell_only")

    objective(batch, models, epoch=0.0)

    cells, valid = online_cells(models, batch.jepa)
    lineage_0 = valid & (batch.jepa.udata["lineage_value"] == 0)
    torch.testing.assert_close(objective.bank.get(0), cells[lineage_0].detach())


def _two_mask_draws(kind):
    torch.manual_seed(1)
    first = _batch(kind)
    torch.manual_seed(2)
    second = _batch(kind)
    assert not torch.equal(first.jepa.masks_enc[0], second.jepa.masks_enc[0])
    return first, second


@pytest.mark.parametrize(("kind", "term"), [("spatial_cell_only", "view_consistency"),
                                            ("cell_only", "alignment")])
def test_cell_terms_use_the_whole_cell_not_the_jepa_context(models, kind, term):
    logs = []
    for batch in _two_mask_draws(kind):
        objective = _objective()
        objective.bank.push(torch.randn(4, 16, generator=torch.Generator().manual_seed(0)),
                            torch.zeros(4, dtype=torch.long))
        logs.append(objective(batch, models, epoch=0.0)[1])

    assert logs[0][term] > 0.0
    assert logs[0][term] == pytest.approx(logs[1][term], rel=1e-5)
    assert logs[0]["jepa"] != pytest.approx(logs[1]["jepa"], rel=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_every_stream_logs_online_embedding_spread(models, kind):
    _, logs = _objective()(_batch(kind), models, epoch=0.0)

    assert logs["emb_std"] > 0.0


def test_emb_std_is_not_part_of_the_loss(models):
    objective = MultiStreamObjective({"variance": {"weight": 0.0}})

    loss, logs = objective(_batch("spatial_full"), models, epoch=0.0)

    assert loss.item() == pytest.approx(logs["jepa"])


def test_term_schedules_gate_weights(models):
    objective = MultiStreamObjective({"alignment": {"weight": 2.0, "start_epoch": 1,
                                                    "ramp_epochs": 2}})

    assert objective.weights(0.5)["alignment"] == 0.0
    assert objective.weights(2.0)["alignment"] == pytest.approx(1.0)
    assert objective.weights(0.0)["variance"] == pytest.approx(0.1)


def test_variance_floor_defaults_to_pretrained_spread():
    # gamma 1 would keep inflating whole-cell embeddings, whose per-dimension
    # std is ~0.2 in TERRA-96M
    assert MultiStreamObjective().gamma == pytest.approx(0.2)
    assert MultiStreamObjective({"variance": {"gamma": 0.5}}).gamma == pytest.approx(0.5)


def test_required_columns():
    objective = MultiStreamObjective({"alignment": {"control_only": True}})

    assert objective.required_columns("spatial_full") == []
    assert objective.required_columns("spatial_cell_only") == ["lineage_value"]
    assert objective.required_columns("cell_only") == ["lineage_value", "is_control_value"]


def test_state_round_trip(models):
    objective = _objective()
    objective(_batch("spatial_cell_only"), models, epoch=0.0)

    restored = _objective()
    restored.load_state_dict(objective.state_dict())

    torch.testing.assert_close(restored.bank.get(0), objective.bank.get(0))


def test_total_loss_is_weighted_sum_of_terms(models):
    objective = MultiStreamObjective({"cell_only_jepa": {"weight": 0.5},
                                      "view_consistency": {"weight": 2.0},
                                      "variance": {"weight": 0.0}})
    torch.manual_seed(0)
    batch = _batch("spatial_cell_only")

    loss, logs = objective(batch, models, epoch=0.0)

    assert loss.item() == pytest.approx(0.5 * logs["jepa"] + 2.0 * logs["view_consistency"],
                                        rel=1e-5)


def test_unknown_term_raises():
    with pytest.raises(ValueError, match="alignmnet"):
        MultiStreamObjective({"alignmnet": {"weight": 1.0}})

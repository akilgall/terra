"""Per-stream objective (``terra.training.stream_objective``) on a tiny model."""

import math

import pytest
import torch

datasets = pytest.importorskip("datasets")

from terra.datasets.cell_datasets import CellGraphDataset  # noqa: E402
from terra.masks.block_masking import BlockMaskCollator  # noqa: E402
from terra.training.multistream import MultiStreamCollator, StreamTaggedDataset  # noqa: E402
from terra.training.stream_objective import MultiStreamObjective, StreamModels  # noqa: E402
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
    assert set(logs) == {"jepa", "view_consistency", "variance"}


def test_cell_only_alignment_needs_bank_and_uses_controls(models):
    objective = _objective()
    _, empty = objective(_batch("cell_only"), models, epoch=0.0)
    objective(_batch("spatial_cell_only", seed=1), models, epoch=0.0)

    _, filled = objective(_batch("cell_only"), models, epoch=0.0)

    assert set(empty) == {"jepa", "alignment", "variance", "align_lineages"}
    assert empty["alignment"] == 0.0 and empty["align_lineages"] == 0
    assert filled["alignment"] > 0.0
    assert filled["align_lineages"] == 1  # lineage 0: 4 controls; lineage 1: 1 cell


def test_alignment_on_all_cells_when_not_control_only(models):
    objective = _objective(control_only=False, min_cells=6)
    objective(_batch("spatial_cell_only", seed=1), models, epoch=0.0)

    _, logs = objective(_batch("cell_only"), models, epoch=0.0)

    assert logs["align_lineages"] == 1  # 6 lineage-0 cells only without the control filter


def test_term_schedules_gate_weights(models):
    objective = MultiStreamObjective({"alignment": {"weight": 2.0, "start_epoch": 1,
                                                    "ramp_epochs": 2}})

    assert objective.weights(0.5)["alignment"] == 0.0
    assert objective.weights(2.0)["alignment"] == pytest.approx(1.0)
    assert objective.weights(0.0)["variance"] == pytest.approx(0.1)


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

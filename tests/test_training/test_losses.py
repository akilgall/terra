"""Alignment-fine-tuning losses (``terra.training.losses``)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from terra.training.losses import (
    EmbeddingBank,
    TermSchedule,
    embedding_spread,
    jepa_loss,
    pool_centre,
    pool_tokens,
    routed_alignment_loss,
    variance_loss,
    view_consistency_loss,
)


def test_jepa_loss_matches_legacy_smooth_l1():
    torch.manual_seed(0)
    z = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
    h = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]

    expected = (F.smooth_l1_loss(z[0], h[0]) + F.smooth_l1_loss(z[1], h[1])) / 2

    torch.testing.assert_close(jepa_loss(z, h, "smooth_l1"), expected)
    torch.testing.assert_close(jepa_loss(z, h, "mse"),
                               (F.mse_loss(z[0], h[0]) + F.mse_loss(z[1], h[1])) / 2)


def test_jepa_loss_rejects_unknown_type():
    with pytest.raises(ValueError, match="loss"):
        jepa_loss([torch.zeros(1)], [torch.zeros(1)], "huber")


def test_pool_centre_uses_centre_real_tokens_only():
    h = torch.arange(1.0, 7.0).view(1, 6, 1).repeat(1, 1, 2)  # token value = position+1
    tokens = torch.tensor([[5, 6, 0, 7, 8, 0]])
    segments = torch.tensor([[1, 1, 0, 2, 2, 0]])

    pooled, valid = pool_centre(h, tokens, segments)

    assert pooled.tolist() == [[1.5, 1.5]]
    assert valid.tolist() == [True]


def test_pool_centre_flags_rows_without_centre_tokens():
    h = torch.ones(2, 3, 2)
    tokens = torch.tensor([[0, 0, 4], [3, 0, 0]])
    segments = torch.tensor([[0, 0, 2], [1, 0, 0]])

    pooled, valid = pool_centre(h, tokens, segments)

    assert valid.tolist() == [False, True]
    assert torch.isfinite(pooled).all()


def test_pool_tokens_is_mean_over_tokens():
    z = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])

    assert pool_tokens(z).tolist() == [[2.0, 4.0]]


def test_view_consistency_loss_bounds_and_stop_gradient():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    target = torch.tensor([[2.0, 0.0], [0.0, -1.0]], requires_grad=True)

    loss = view_consistency_loss(a, target)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(1.0))  # (0 + 2) / 2
    assert target.grad is None
    assert a.grad is not None


def test_variance_loss_zero_for_spread_and_positive_for_collapse():
    spread = torch.randn(256, 8) * 3.0
    collapsed = torch.ones(256, 8) + 1e-6 * torch.randn(256, 8)

    assert variance_loss(spread, gamma=1.0).item() == pytest.approx(0.0, abs=1e-3)
    assert variance_loss(collapsed, gamma=1.0).item() > 0.9


def test_variance_loss_is_zero_for_single_row():
    assert variance_loss(torch.ones(1, 4)).item() == 0.0


def test_embedding_spread_is_mean_per_dimension_std():
    z = torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]], requires_grad=True)

    assert embedding_spread(z) == pytest.approx(1.0)  # dim 0: std 2, dim 1: std 0
    assert embedding_spread(torch.ones(5, 3)) == 0.0
    assert embedding_spread(torch.ones(1, 3)) == 0.0


def test_embedding_bank_fifo_per_lineage_ignores_unmapped():
    bank = EmbeddingBank(size=3)
    bank.push(torch.arange(4.0).view(4, 1), torch.tensor([0, 0, -1, 1]))
    bank.push(torch.tensor([[10.0], [11.0]]), torch.tensor([0, 0]))

    assert bank.get(0).view(-1).tolist() == [1.0, 10.0, 11.0]
    assert bank.get(1).view(-1).tolist() == [3.0]
    assert bank.get(-1) is None
    assert bank.get(5) is None


def test_embedding_bank_detaches_and_round_trips():
    bank = EmbeddingBank(size=4)
    z = torch.randn(2, 3, requires_grad=True)
    bank.push(z, torch.tensor([2, 2]))

    restored = EmbeddingBank(size=4)
    restored.load_state_dict(bank.state_dict())

    assert not bank.get(2).requires_grad
    torch.testing.assert_close(restored.get(2), z.detach())


def _numpy_coral(a: np.ndarray, b: np.ndarray) -> float:
    d = a.shape[1]
    return float(((np.cov(a, rowvar=False) - np.cov(b, rowvar=False)) ** 2).sum() / (4 * d * d))


def test_routed_alignment_matches_coral_plus_mean_oracle():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 3))
    b = rng.normal(loc=2.0, scale=0.5, size=(50, 3))
    bank = EmbeddingBank(size=64)
    bank.push(torch.tensor(b, dtype=torch.float32), torch.zeros(50, dtype=torch.long))

    loss, n_matched = routed_alignment_loss(
        torch.tensor(a, dtype=torch.float32), torch.zeros(40, dtype=torch.long), bank,
        min_cells=8, lambda_mean=1.0)

    mean_term = float(((a.mean(0) - b.mean(0)) ** 2).sum() / 3)
    assert n_matched == 1
    assert loss.item() == pytest.approx(_numpy_coral(a, b) + mean_term, rel=1e-4)


def test_routed_alignment_skips_small_or_unmatched_lineages():
    bank = EmbeddingBank(size=64)
    bank.push(torch.randn(20, 3), torch.zeros(20, dtype=torch.long))
    z = torch.randn(10, 3, requires_grad=True)
    labels = torch.tensor([0] * 4 + [1] * 6)  # lineage 0 too small, lineage 1 not in bank

    loss, n_matched = routed_alignment_loss(z, labels, bank, min_cells=5, lambda_mean=1.0)
    loss.backward()

    assert n_matched == 0
    assert loss.item() == 0.0


def test_routed_alignment_gradient_flows_to_z_only():
    bank = EmbeddingBank(size=64)
    bank.push(torch.randn(20, 3) + 1.0, torch.zeros(20, dtype=torch.long))
    z = torch.randn(10, 3, requires_grad=True)

    loss, _ = routed_alignment_loss(z, torch.zeros(10, dtype=torch.long), bank,
                                    min_cells=5, lambda_mean=1.0)
    loss.backward()

    assert z.grad is not None and z.grad.abs().sum() > 0


@pytest.mark.parametrize("epoch, expected", [(0.0, 0.0), (0.99, 0.0), (1.0, 0.0),
                                             (1.5, 1.0), (2.0, 2.0), (5.0, 2.0)])
def test_term_schedule_ramp(epoch, expected):
    schedule = TermSchedule(weight=2.0, start_epoch=1, ramp_epochs=1)

    assert schedule.value(epoch) == pytest.approx(expected)


def test_term_schedule_without_ramp_is_a_step():
    schedule = TermSchedule(weight=0.5, start_epoch=2, ramp_epochs=0)

    assert schedule.value(1.9) == 0.0
    assert schedule.value(2.0) == 0.5

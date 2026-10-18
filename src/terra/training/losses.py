"""Losses for multi-stream alignment fine-tuning.

Streams (see ``terra.training.multistream``): spatial cells with their
neighbours (JEPA), the same cells with neighbours hidden (JEPA + consistency
with the full view), and Tahoe-100M cells (JEPA + alignment to the spatial
cell-only embeddings of the same lineage + an anti-collapse variance term).
"""

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from terra.models.distribution_alignment import coral_loss

__all__ = [
    "jepa_loss",
    "pool_centre",
    "pool_tokens",
    "view_consistency_loss",
    "variance_loss",
    "EmbeddingBank",
    "routed_alignment_loss",
    "TermSchedule",
]


def jepa_loss(z: list[torch.Tensor],
              h: list[torch.Tensor],
              loss_fn_type: str = "smooth_l1",
              ) -> torch.Tensor:
    """Prediction loss averaged over target masks (as in the legacy loop)."""
    if loss_fn_type not in ("smooth_l1", "l1", "mse"):
        raise ValueError(f"Unknown JEPA loss_fn_type {loss_fn_type!r}.")
    loss = 0.0
    for zi, hi in zip(z, h, strict=True):
        if loss_fn_type == "smooth_l1":
            loss = loss + F.smooth_l1_loss(zi, hi)
        elif loss_fn_type == "l1":
            loss = loss + torch.mean(torch.abs(zi - hi))
        else:
            loss = loss + F.mse_loss(zi, hi)
    return loss / len(h)


def pool_centre(h: torch.Tensor,
                tokens: torch.Tensor,
                segments: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean embedding over the centre cell's real gene tokens.

    Layout-agnostic: uses segment ids (centre = 1), not slot offsets.

    Returns
    -------
    pooled:
        (B, D) mean embeddings (zeros for rows without centre tokens).
    valid:
        (B,) whether the row had at least one centre token.
    """
    mask = ((segments == 1) & (tokens != 0)).to(h.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    pooled = (h * mask).sum(dim=1) / counts.clamp(min=1.0)
    return pooled, counts.squeeze(-1) > 0


def pool_tokens(z: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis of (B, L, D) embeddings (e.g. context tokens)."""
    return z.mean(dim=1)


def view_consistency_loss(online: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`1 - cosine` between online and (stop-gradient) target embeddings."""
    return (1.0 - F.cosine_similarity(online, target.detach(), dim=-1)).mean()


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg variance hinge: penalise dimensions whose std falls below gamma.

    Returns 0 for fewer than two rows, where the variance is undefined.
    """
    if z.size(0) < 2:
        return z.sum() * 0.0
    z = z.float()
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


class EmbeddingBank:
    """
    Per-lineage FIFO of detached embeddings.

    Holds recent spatial cell-only embeddings so Tahoe batches can be aligned
    to the same lineage even though the two domains never share a batch.
    Label -1 (unmapped) is ignored.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError(f"Bank size must be positive, got {size}.")
        self.size = size
        self._banks: dict[int, deque] = {}

    def push(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        """Append rows of `z` to the queue of their label."""
        z = z.detach()
        for label in labels.unique().tolist():
            if label < 0:
                continue
            rows = z[labels == label]
            queue = self._banks.setdefault(label, deque(maxlen=self.size))
            queue.extend(rows.unbind(0))

    def get(self, label: int) -> torch.Tensor | None:
        """Stacked embeddings for a label, or `None` if empty."""
        queue = self._banks.get(label)
        if not queue:
            return None
        return torch.stack(list(queue))

    def labels(self) -> list[int]:
        """Labels that currently hold embeddings."""
        return [label for label, queue in self._banks.items() if queue]

    def state_dict(self) -> dict:
        """Serializable contents (CPU tensors)."""
        return {"size": self.size,
                "banks": {label: torch.stack(list(queue)).cpu()
                          for label, queue in self._banks.items() if queue}}

    def load_state_dict(self, state: dict) -> None:
        """Restore contents written by `state_dict`."""
        self.size = state["size"]
        self._banks = {int(label): deque(rows.unbind(0), maxlen=self.size)
                       for label, rows in state["banks"].items()}


def routed_alignment_loss(z: torch.Tensor,
                          labels: torch.Tensor,
                          bank: EmbeddingBank,
                          min_cells: int,
                          lambda_mean: float,
                          ) -> tuple[torch.Tensor, int]:
    """
    CORAL + mean matching between `z` and the bank, per shared lineage.

    CORAL only matches covariances, so without the first-moment term the two
    domains could stay translated copies of each other.

    Returns
    -------
    loss:
        Mean over matched lineages (0 if none matched).
    n_matched:
        Lineages with at least `min_cells` rows in both `z` and the bank.
    """
    losses = []
    for label in labels.unique().tolist():
        if label < 0:
            continue
        z_l = z[labels == label].float()
        bank_l = bank.get(label)
        if bank_l is None or len(z_l) < min_cells or len(bank_l) < min_cells:
            continue
        bank_l = bank_l.to(device=z_l.device, dtype=z_l.dtype)
        mean_term = (z_l.mean(0) - bank_l.mean(0)).pow(2).sum() / z_l.size(1)
        losses.append(coral_loss(z_l, bank_l) + lambda_mean * mean_term)
    if not losses:
        return z.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


@dataclass(frozen=True)
class TermSchedule:
    """
    Loss weight on a per-epoch schedule.

    0 before `start_epoch`, then a linear ramp to `weight` over `ramp_epochs`
    (a step when `ramp_epochs` is 0).
    """

    weight: float
    start_epoch: float = 0.0
    ramp_epochs: float = 0.0

    def value(self, epoch: float) -> float:
        """Weight at a (fractional) epoch."""
        if epoch < self.start_epoch:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.weight
        progress = min(1.0, (epoch - self.start_epoch) / self.ramp_epochs)
        return self.weight * progress

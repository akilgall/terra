"""Per-stream training objective for multi-stream alignment fine-tuning.

One optimizer step sees one stream (see ``terra.training.multistream``):

- ``spatial_full``: JEPA on cells with their neighbours.
- ``spatial_cell_only``: JEPA on the same cells with neighbours hidden, plus
  view consistency between the online cell-only embedding and the target
  encoder's full-view embedding. The online cell-only embeddings (detached)
  fill a per-lineage bank, so Tahoe cells are aligned to where spatial cells
  are now rather than to the lagging EMA target.
- ``cell_only`` (Tahoe-100M): JEPA, plus alignment of (control) cells to the
  bank of the same lineage, plus a variance term against collapse.

Cell embeddings are means of the layer-normed embeddings of the centre cell's
real tokens on the whole (unmasked) cell-only view, for the online and the
target encoder alike: the representation `embed_dataset` returns, so the
consistency and alignment terms train what inference uses. JEPA itself runs
on the masked view.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from terra.masks.utils import apply_masks
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
from terra.training.multistream import StreamBatch, View

__all__ = ["TERMS", "StreamModels", "MultiStreamObjective"]

TERMS = ("cell_only_jepa", "view_consistency", "alignment", "variance")
DEFAULT_WEIGHTS = {"cell_only_jepa": 1.0, "view_consistency": 1.0, "alignment": 1.0,
                   "variance": 0.1}
# Variance floor: about the per-dimension std of TERRA-96M whole-cell
# embeddings. A higher floor (VICReg's 1.0) keeps inflating the embeddings.
DEFAULT_GAMMA = 0.2


@dataclass(frozen=True)
class StreamModels:
    """Models for one step (`encoder`/`predictor` with PEFT wrappers unwrapped)."""

    encoder: Callable
    predictor: Callable
    target_encoder: nn.Module


def _layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),))


def target_tokens(target_encoder: nn.Module, view: View) -> torch.Tensor:
    """Layer-normed target-encoder embeddings of every token (no grad)."""
    with torch.no_grad():
        h, _ = target_encoder(batch=view.udata, masks_attention=view.attn)
        return _layer_norm(h)


def target_cells(h: torch.Tensor, view: View) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre-cell embeddings from target token embeddings, and row validity."""
    pooled, valid = pool_centre(h, view.udata["tokens"], view.udata["segments"])
    return pooled.float(), valid


def online_cells(models: StreamModels, view: View) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Online-encoder centre-cell embeddings of the whole (unmasked) view.

    The same pooling as `target_cells` and as the cell embeddings of
    `embed_dataset`, so terms on these train what inference returns.
    """
    h, _ = models.encoder(batch=view.udata, masks_attention=view.attn)
    return target_cells(_layer_norm(h), view)


def online_jepa(models: StreamModels,
                view: View,
                h: torch.Tensor,
                loss_fn_type: str,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JEPA loss of the online encoder + predictor, and online cell embeddings.

    Cell embeddings are pooled context tokens, averaged over context masks.
    """
    z, token_emb = models.encoder(batch=view.udata, masks=view.masks_enc, masks_attention=None)
    pred = models.predictor(z=z, token_emb=token_emb, batch=view.udata,
                            masks_enc=view.masks_enc, masks_pred=view.masks_pred,
                            masks_attention=None)
    loss = jepa_loss(pred, apply_masks(h, view.masks_pred, concat=False), loss_fn_type)
    cells = torch.stack([pool_tokens(_layer_norm(zi)) for zi in z]).mean(dim=0)
    return loss, cells.float()


def _schedule(name: str, cfg: dict | None) -> TermSchedule:
    cfg = cfg or {}
    return TermSchedule(weight=float(cfg.get("weight", DEFAULT_WEIGHTS[name])),
                        start_epoch=float(cfg.get("start_epoch", 0.0)),
                        ramp_epochs=float(cfg.get("ramp_epochs", 0.0)))


class MultiStreamObjective:
    """
    Loss for a single-stream batch.

    Parameters
    ----------
    cfg:
        `finetune.multistream` section: per-term `{weight, start_epoch,
        ramp_epochs}` for `cell_only_jepa`, `view_consistency`, `alignment`
        and `variance`; `alignment` also takes `min_cells`, `lambda_mean`,
        `bank_size`, `control_only`, `key` and `control_key`; `variance`
        takes `gamma`.
    loss_fn_type:
        JEPA loss (`smooth_l1`, `l1` or `mse`).
    """

    def __init__(self, cfg: dict | None = None, loss_fn_type: str = "smooth_l1"):
        cfg = cfg or {}
        unknown = sorted(set(cfg) - set(TERMS))
        if unknown:
            raise ValueError(f"Unknown finetune.multistream terms {unknown}; "
                             f"expected some of {TERMS}.")
        align = cfg.get("alignment") or {}
        self.schedules = {name: _schedule(name, cfg.get(name)) for name in TERMS}
        self.min_cells = int(align.get("min_cells", 32))
        self.lambda_mean = float(align.get("lambda_mean", 1.0))
        self.control_only = bool(align.get("control_only", True))
        self.label_key = align.get("key", "lineage_value")
        self.control_key = align.get("control_key", "is_control_value")
        self.gamma = float((cfg.get("variance") or {}).get("gamma", DEFAULT_GAMMA))
        self.loss_fn_type = loss_fn_type
        self.bank = EmbeddingBank(int(align.get("bank_size", 1024)))

    def required_columns(self, kind: str) -> list[str]:
        """Dataset `*_value` columns a stream of this kind must provide."""
        if kind == "spatial_cell_only":
            return [self.label_key]
        if kind == "cell_only":
            return [self.label_key] + ([self.control_key] if self.control_only else [])
        return []

    def weights(self, epoch: float) -> dict[str, float]:
        """Scheduled term weights at a (fractional) epoch."""
        return {name: schedule.value(epoch) for name, schedule in self.schedules.items()}

    def __call__(self,
                 batch: StreamBatch,
                 models: StreamModels,
                 epoch: float,
                 ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Total loss, plus its detached terms and monitors for one batch.

        Monitors (not part of the loss): `emb_std`, the spread of the online
        cell embeddings (see `embedding_spread`), and for Tahoe batches
        `align_lineages`, the number of lineages aligned.
        """
        if batch.kind == "spatial_full":
            h = target_tokens(models.target_encoder, batch.jepa)
            loss, online = online_jepa(models, batch.jepa, h, self.loss_fn_type)
            return loss, {"jepa": loss.item(), "emb_std": embedding_spread(online)}
        if batch.kind == "spatial_cell_only":
            terms, extra = self._spatial_cell_only(batch, models)
        else:
            terms, extra = self._cell_only(batch, models)
        scheduled = self.weights(epoch)
        weights = {**scheduled, "jepa": scheduled["cell_only_jepa"]}
        loss = sum(weights[name] * value for name, value in terms.items())
        return loss, {**{name: value.item() for name, value in terms.items()}, **extra}

    def _spatial_cell_only(self, batch: StreamBatch, models: StreamModels) -> tuple[dict, dict]:
        full, full_valid = target_cells(target_tokens(models.target_encoder, batch.full),
                                        batch.full)
        h = target_tokens(models.target_encoder, batch.jepa)
        jepa, _ = online_jepa(models, batch.jepa, h, self.loss_fn_type)
        cells, valid = online_cells(models, batch.jepa)
        labels = batch.jepa.udata[self.label_key]
        self.bank.push(cells[valid], labels[valid])
        paired = valid & full_valid
        consistency = (view_consistency_loss(cells[paired], full[paired])
                       if paired.any() else cells.sum() * 0.0)
        return ({"jepa": jepa, "view_consistency": consistency,
                 "variance": variance_loss(cells[valid], self.gamma)},
                {"emb_std": embedding_spread(cells[valid])})

    def _cell_only(self, batch: StreamBatch, models: StreamModels) -> tuple[dict, dict]:
        h = target_tokens(models.target_encoder, batch.jepa)
        jepa, _ = online_jepa(models, batch.jepa, h, self.loss_fn_type)
        cells, valid = online_cells(models, batch.jepa)
        labels = batch.jepa.udata[self.label_key]
        keep = valid & (batch.jepa.udata[self.control_key].bool() if self.control_only
                        else torch.ones_like(valid))
        align, n_matched = routed_alignment_loss(
            cells[keep], labels[keep], self.bank, self.min_cells, self.lambda_mean)
        return ({"jepa": jepa, "alignment": align,
                 "variance": variance_loss(cells[valid], self.gamma)},
                {"align_lineages": float(n_matched), "emb_std": embedding_spread(cells[valid])})

    def state_dict(self) -> dict:
        """Serializable state (the embedding bank)."""
        return {"bank": self.bank.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore state written by `state_dict`."""
        self.bank.load_state_dict(state["bank"])

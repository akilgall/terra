"""Multi-stream alignment fine-tuning loop.

Reached through :func:`terra.training.finetune_self_supervised.finetune_self_supervised`
when ``data.streams`` is set (see ``configs/finetune/tahoe_alignment.yaml``).

The sequence layout may differ from pretraining (e.g. a 2048-gene centre cell
and 256 genes per neighbour): the model is rebuilt for the new layout, the
pretrained position tables are reused (grown with their sincos continuation
only if a segment is longer than the table), and the run directory gets its
own ``model_config.yaml`` describing the new layout, which
:func:`~terra.training.finetune_self_supervised.prepare_finetuned_model`
carries into the exported model.
"""

import logging
import pickle
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm

from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.segment_layout import SegmentLayout
from terra.masks.block_masking import BlockMaskCollator
from terra.training.lora import apply_lora, lora_scale_from_config, merge_lora_weights
from terra.training.multistream import StreamSpec, build_multistream_loader, load_stream_dataset
from terra.training.stream_objective import MultiStreamObjective, StreamModels
from terra.utils.device import autocast, should_pin_memory
from terra.utils.helper import (
    extend_pos_embed_state_dict,
    init_model,
    init_opt,
    register_row_grad_mask,
    unseen_gene_rows,
)

logger = logging.getLogger(__name__)

__all__ = ["finetune_multistream", "resolve_layout", "resolve_n_positions",
           "run_model_config", "ema_update"]

LOG_EVERY = 10
DEFAULT_PEFT_TARGETS = ["qkv", "proj", "fc1", "fc2"]


def resolve_layout(data_cfg: dict, overrides: dict, n_special_tokens: int = 0) -> SegmentLayout:
    """
    Fine-tune sequence layout.

    `seq_len_cell`, `seq_len_neigh` and `n_neighs` come from the fine-tune
    `data` section when set, else from the pretrained `data` config; without
    `seq_len_neigh` the layout stays uniform.
    """
    seq_len_cell = int(overrides.get("seq_len_cell", data_cfg["seq_len_cell"]))
    neigh = overrides.get("seq_len_neigh", data_cfg.get("seq_len_neigh"))
    n_neighs = overrides.get("n_neighs")
    if n_neighs is None:
        n_neighs = data_cfg["n_neighs"] if "n_neighs" in data_cfg else data_cfg["n_segments"] - 1
    return SegmentLayout(seq_len_cell=seq_len_cell,
                         seq_len_neigh=seq_len_cell if neigh is None else int(neigh),
                         n_neighs=int(n_neighs), n_special_tokens=n_special_tokens)


def resolve_n_positions(model_config: dict, layout: SegmentLayout) -> int:
    """
    Position-table size for the fine-tune layout.

    The pretrained size, grown only if a segment is longer (positions restart
    at 1 in every segment).
    """
    data, meta = model_config["data"], model_config["meta"]
    pretrained = meta.get("n_positions", data["seq_len_cell"] + data["seq_len_neighborhood"]
                          + len(meta["special_tokens"]))
    return max(int(pretrained), layout.seq_len_cell, layout.seq_len_neigh)


def run_model_config(model_config: dict, layout: SegmentLayout, n_positions: int) -> dict:
    """
    Pretrained model config updated to the fine-tune layout (new dict).

    `seq_len_neigh` is only written for asymmetric layouts (or if the
    pretrained config had it), since it switches tokenization of the
    exported model to fixed per-neighbour budgets.
    """
    data = {**model_config["data"],
            "seq_len_cell": layout.seq_len_cell,
            "seq_len_neigh": layout.seq_len_neigh,
            "seq_len_neighborhood": layout.n_neighs * layout.seq_len_neigh,
            "n_neighs": layout.n_neighs,
            "n_segments": layout.n_segments,
            "model_input_size": layout.total}
    if layout.is_uniform and "seq_len_neigh" not in model_config["data"]:
        data = {key: value for key, value in data.items() if key != "seq_len_neigh"}
    if SegmentLayout.from_data_config(data, n_special_tokens=layout.n_special_tokens) != layout:
        raise ValueError(f"Run config {data} does not reproduce layout {layout}.")
    return {**model_config, "data": data,
            "meta": {**model_config["meta"], "n_positions": n_positions}}


def _read_bundle(bundle: Path) -> tuple[dict, dict]:
    with open(bundle / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    with open(bundle / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)
    if model_config["data"]["tokenizer_type"] != "cell_graph":
        raise ValueError("Multi-stream fine-tuning needs a cell_graph model, got "
                         f"{model_config['data']['tokenizer_type']!r}.")
    if model_config["meta"]["special_tokens"]:
        raise ValueError("Multi-stream fine-tuning does not support special tokens "
                         f"(model has {model_config['meta']['special_tokens']}).")
    return model_config, token_dict


def _masker(model_config: dict, layout: SegmentLayout, data_args: dict) -> BlockMaskCollator:
    """Block-mask sampler for the layout (every view needs JEPA masks)."""
    if data_args.get("sample_segments"):
        raise ValueError("data.sample_segments is not supported with data.streams; "
                         "use a spatial_cell_only stream for neighbour-free views.")
    mask = model_config["mask"]
    return BlockMaskCollator(
        n_targets=mask["n_targets"], n_contexts=mask["n_contexts"],
        n_segments=layout.n_segments, seq_len_cell=layout.seq_len_cell,
        seq_len_neighborhood=layout.n_neighs * layout.seq_len_neigh,
        seq_len_neigh=layout.seq_len_neigh, n_special_tokens=layout.n_special_tokens,
        per_block_mask_ratio=mask["per_block_mask_ratio"], sample_segments=False,
        sample_gene_masks=True)


def _stream_datasets(specs: list[StreamSpec],
                     objective: MultiStreamObjective,
                     model_config: dict,
                     layout: SegmentLayout,
                     vocab_size: int,
                     data_args: dict,
                     ) -> list:
    meta = model_config["meta"]
    return [init_cell_dataset(
        tokenizer_type="cell_graph",
        dataset=load_stream_dataset(spec.paths, objective.required_columns(spec.kind)),
        vocab_size=vocab_size,
        seq_len_cell=layout.seq_len_cell,
        seq_len_neighborhood=layout.n_neighs * layout.seq_len_neigh,
        seq_len_neigh=layout.seq_len_neigh,
        tokenized_seq_len_cell=(spec.tokenized_seq_len_cell
                                or data_args.get("tokenized_seq_len_cell")),
        tokenized_seq_len_neigh=(spec.tokenized_seq_len_neigh
                                 or data_args.get("tokenized_seq_len_neigh")),
        centre_position_cap=data_args.get("centre_position_cap"),
        gt_type=meta["gt_type"], cell_pos_enc=meta["cell_pos_enc"],
        special_tokens=meta["special_tokens"], sampling_strategy=None,
        n_nonzero_tokens_list=None, include_cell_id=False,
        sep_gene_tokens_neb=model_config["data"]["sep_gene_tokens_neb"],
    ) for spec in specs]


def _build_models(model_config: dict,
                  device: torch.device,
                  token_dict: dict,
                  layout: SegmentLayout,
                  n_positions: int,
                  grad_checkpointing: bool,
                  ) -> tuple[nn.Module, nn.Module, nn.Module]:
    meta = model_config["meta"]

    def build():
        return init_model(
            gt_type=meta["gt_type"], count_encoding=meta["count_encoding"],
            n_value_bins=meta["n_value_bins"], cell_pos_enc=meta["cell_pos_enc"],
            device=device, vocab_size=len(token_dict), seq_len=layout.seq_len,
            n_special_tokens=layout.n_special_tokens, n_segments=layout.n_segments,
            n_special_values=sum(1 for key in token_dict if "spv" in key),
            enc_emb_dim=meta["enc_emb_dim"], enc_depth=meta["enc_depth"],
            pred_emb_dim=meta["pred_emb_dim"], pred_depth=meta["pred_depth"],
            num_heads=meta["num_heads"], mlp_ratio=meta["mlp_ratio"],
            use_flash_attention=meta["use_flash_attention"], api_version=meta["api_version"],
            sep_gene_tokens_neb=model_config["data"]["sep_gene_tokens_neb"],
            predict_gene=meta["predict_gene"], pos_learnable=meta["pos_learnable"],
            mlp_bias=meta.get("mlp_bias", True), seq_len_cell=layout.seq_len_cell,
            n_positions=n_positions, grad_checkpointing=grad_checkpointing)

    encoder, predictor = build()
    target_encoder, _ = build()
    return encoder, predictor, target_encoder


def _load_pretrained(path: Path, models: dict[str, nn.Module], n_positions: int) -> None:
    """Strictly load pretrained weights, growing position tables if needed."""
    checkpoint = torch.load(path, map_location="cpu")
    for key, model in models.items():
        state = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}
        model.load_state_dict(extend_pos_embed_state_dict(state, n_positions))
        logger.info("Loaded pretrained %s from %s", key, path)


def _unwrap(model: nn.Module, use_peft: bool) -> nn.Module:
    return model.base_model.model if use_peft else model


def _protect_pretrained_rows(encoder: nn.Module, token_dict: dict, bundle: Path) -> None:
    """Train only the token embeddings of genes unseen in pretraining."""
    counts_file = bundle / "gene_count_dictionary.pkl"
    found = [module for name, module in encoder.named_modules()
             if name.rsplit(".", 1)[-1] == "token_embed"]
    if len(found) != 1 or not isinstance(found[0], nn.Embedding):
        raise ValueError("token_embed_rows: unseen needs one nn.Embedding token_embed.")
    embedding = found[0]
    if not counts_file.exists():
        raise FileNotFoundError(f"{counts_file} is needed for token_embed_rows: unseen; "
                                "use token_embed_rows: all to train every row.")
    with open(counts_file, "rb") as f:
        rows = unseen_gene_rows(token_dict, pickle.load(f))
    register_row_grad_mask(embedding, rows)
    logger.info("token_embed: %d unseen gene rows trainable, %d rows fixed",
                len(rows), embedding.num_embeddings - len(rows))


def _configure_trainable(encoder: nn.Module,
                         predictor: nn.Module,
                         ft: dict,
                         token_dict: dict,
                         bundle: Path,
                         device: torch.device,
                         ) -> tuple[nn.Module, nn.Module]:
    """Apply LoRA, unfreeze `extra_trainable` and protect pretrained token rows."""
    use_peft = ft.get("use_peft", False)
    if use_peft:
        dropout = ft["peft_dropout"]
        if ft.get("grad_checkpointing") and device.type == "mps" and dropout > 0:
            logger.warning("peft_dropout %s set to 0: gradient checkpointing on MPS "
                           "cannot replay dropout.", dropout)
            dropout = 0.0
        lora_kwargs = {"peft_rank": ft["peft_rank"], "peft_alpha": ft["peft_alpha"],
                       "peft_dropout": dropout, "peft_bias": ft["peft_bias"],
                       "peft_target_modules": ft.get("peft_target_modules",
                                                     DEFAULT_PEFT_TARGETS)}
        encoder = apply_lora(encoder, **lora_kwargs)
        predictor = apply_lora(predictor, **lora_kwargs)
    extra = ft.get("extra_trainable") or []
    for name, param in encoder.named_parameters():
        if any(pattern in name for pattern in extra):
            param.requires_grad = True
    if "token_embed" in extra and ft.get("token_embed_rows", "unseen") == "unseen":
        _protect_pretrained_rows(encoder, token_dict, bundle)
    for label, model in (("encoder", encoder), ("predictor", predictor)):
        logger.info("Trainable %s params: %d", label,
                    sum(p.numel() for p in model.parameters() if p.requires_grad))
    return encoder, predictor


def _optimizer(encoder: nn.Module, predictor: nn.Module, ft: dict, ipe: int):
    """
    AdamW and its (lr, wd) schedules.

    An unfrozen token_embed gets its own group without weight decay, so its
    fixed rows stay exactly unchanged.
    """
    overrides = ft.get("param_group_overrides")
    if overrides is None and "token_embed" in (ft.get("extra_trainable") or []):
        overrides = [{"match": "token_embed.weight", "weight_decay": 0.0,
                      "lr_mult": ft.get("token_embed_lr_mult", 5.0)}]
    optimizer, _, lr_schedule, wd_schedule = init_opt(
        encoder=encoder, predictor=predictor, iterations_per_epoch=ipe,
        start_lr=ft.get("start_lr", ft["lr"]), ref_lr=ft["lr"],
        warmup=ft.get("warmup_epochs", 0), num_epochs=ft["num_epochs"],
        wd=ft.get("weight_decay", 0.04), final_wd=ft.get("final_weight_decay", 0.4),
        final_lr=ft.get("final_lr", 1.0e-06), use_bfloat16=ft.get("use_bfloat16", False),
        param_group_overrides=overrides)
    return optimizer, (lr_schedule, wd_schedule)


def ema_update(encoder: nn.Module,
               target_encoder: nn.Module,
               momentum: float,
               lora_scale: float | None = None,
               ) -> None:
    """
    EMA update `target = momentum * target + (1 - momentum) * online`.

    The online encoder's LoRA adapters are merged first when `lora_scale` is
    given.
    """
    with torch.no_grad():
        if lora_scale is None:
            for online, target in zip(encoder.parameters(), target_encoder.parameters(),
                                      strict=True):
                target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
            return
        merged = merge_lora_weights(encoder.state_dict(), lora_scale)
        for name, target in target_encoder.state_dict().items():
            if name in merged and target.is_floating_point():
                target.mul_(momentum).add_(merged[name].to(target.device),
                                           alpha=1.0 - momentum)


class MultiStreamTrainer:
    """Optimizer steps, EMA, logging and checkpoints for one run."""

    def __init__(self, encoder, predictor, target_encoder, objective, optimizer,
                 schedules, ft: dict, device: torch.device, ipe: int, stream_names):
        self.encoder, self.predictor = encoder, predictor
        self.target_encoder = target_encoder
        self.objective, self.optimizer = objective, optimizer
        self.lr_schedule, self.wd_schedule = schedules
        self.ft, self.device, self.ipe = ft, device, ipe
        self.stream_names = list(stream_names)
        use_peft = ft.get("use_peft", False)
        self.models = StreamModels(_unwrap(encoder, use_peft), _unwrap(predictor, use_peft),
                                   target_encoder)
        self.lora_scale = lora_scale_from_config(ft) if use_peft else None
        self.num_epochs = ft["num_epochs"]
        self.total_steps = ipe * self.num_epochs
        self.global_step = 0
        self.start_epoch = 0

    def momentum(self) -> float:
        """EMA momentum, linear from `ema_momentum` to `final_ema_momentum`."""
        start = self.ft.get("ema_momentum", 0.996)
        end = self.ft.get("final_ema_momentum", 1.0)
        return start + self.global_step * (end - start) / self.total_steps

    def step(self, batch, epoch: float) -> tuple[float, dict]:
        """One optimizer step on a single-stream batch."""
        self.lr_schedule.step()
        self.wd_schedule.step()
        with autocast(self.device, self.ft.get("use_bfloat16", False)):
            loss, logs = self.objective(batch.to(self.device), self.models, epoch)
        loss.backward()
        clip_grad = self.ft.get("clip_grad")
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        ema_update(self.encoder, self.target_encoder, self.momentum(), self.lora_scale)
        self.global_step += 1
        return loss.item(), logs

    def run_epoch(self, loader, epoch: int, log_fn=None) -> dict[str, float]:
        """Train one epoch; returns mean loss terms keyed `<stream>/<term>`."""
        self.encoder.train()
        self.predictor.train()
        sums, counts = {}, {}
        for itr, batch in enumerate(tqdm(loader)):
            loss, logs = self.step(batch, epoch + itr / self.ipe)
            name = self.stream_names[batch.stream]
            named = {"loss": loss, **{f"{name}/{key}": value for key, value in logs.items()}}
            for key, value in named.items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
            if log_fn is not None and self.global_step % LOG_EVERY == 0:
                log_fn({**named, "step": self.global_step})
        return {key: sums[key] / counts[key] for key in sums}

    def fit(self, loader, sampler, run_dir: Path, log_fn=None) -> None:
        """Train from `start_epoch` to `num_epochs`, checkpointing on schedule."""
        save_every = self.ft.get("save_every", 10)
        for epoch in range(self.start_epoch, self.num_epochs):
            sampler.set_epoch(epoch)
            means = self.run_epoch(loader, epoch, log_fn)
            logger.info("Epoch %d/%d: %s", epoch + 1, self.num_epochs,
                        ", ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())))
            if log_fn is not None:
                log_fn({"epoch": epoch + 1, **means})
            if (epoch + 1) % save_every == 0 or epoch + 1 == self.num_epochs:
                self.save(run_dir, epoch + 1, means["loss"])

    def save(self, run_dir: Path, epoch: int, loss: float) -> None:
        """Write `checkpoint_epoch_<epoch>.pt`."""
        path = run_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({"encoder": self.encoder.state_dict(),
                    "predictor": self.predictor.state_dict(),
                    "target_encoder": self.target_encoder.state_dict(),
                    "opt": self.optimizer.state_dict(), "scaler": None,
                    "epoch": epoch, "loss": loss, "world_size": 1, "lr": self.ft["lr"],
                    "multistream": {"global_step": self.global_step,
                                    **self.objective.state_dict()}}, path)
        logger.info("Checkpoint saved to %s", path)

    def resume(self, path: str) -> None:
        """Restore a checkpoint written by `save` and fast-forward schedules."""
        checkpoint = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.objective.load_state_dict(checkpoint["multistream"])
        self.global_step = checkpoint["multistream"]["global_step"]
        self.start_epoch = checkpoint["epoch"]
        for _ in range(self.global_step):
            self.lr_schedule.step()
            self.wd_schedule.step()
        logger.info("Resumed from %s at epoch %d, step %d", path, self.start_epoch,
                    self.global_step)


def finetune_multistream(args: dict,
                         streams: list[StreamSpec],
                         device: torch.device,
                         run_dir: Path,
                         log_fn=None,
                         ) -> str:
    """
    Multi-stream alignment fine-tuning of a pretrained TERRA model.

    Parameters
    ----------
    args:
        Fine-tune config (`model`, `data` with `streams`, `finetune` with an
        optional `multistream` section).
    streams:
        Parsed `data.streams`.
    device:
        Training device.
    run_dir:
        Existing run directory for checkpoints and `model_config.yaml`.
    log_fn:
        Optional metrics logger (e.g. `wandb.log`).

    Returns
    -------
    str
        The run directory.
    """
    ft, data_args = args["finetune"], args["data"]
    bundle = Path(args["model"]["pretrained_checkpoint_path"])
    model_config, token_dict = _read_bundle(bundle)
    layout = resolve_layout(model_config["data"], data_args)
    n_positions = resolve_n_positions(model_config, layout)
    with open(run_dir / "model_config.yaml", "w") as f:
        yaml.safe_dump(run_model_config(model_config, layout, n_positions), f)
    logger.info("Layout %s (%d tokens), %d positions", layout.lengths, layout.seq_len,
                n_positions)

    objective = MultiStreamObjective(ft.get("multistream"), ft.get("loss_fn_type", "smooth_l1"))
    loader, sampler = build_multistream_loader(
        _stream_datasets(streams, objective, model_config, layout, len(token_dict), data_args),
        streams, _masker(model_config, layout, data_args), data_args["batch_size"],
        epoch_batches=data_args.get("epoch_batches"),
        length_grouped=data_args.get("length_grouped", True),
        trim=data_args.get("trim_trailing_pad", True),
        num_workers=data_args.get("num_workers", 0),
        pin_memory=should_pin_memory(device, data_args.get("pin_memory", False)))

    encoder, predictor, target_encoder = _build_models(
        model_config, device, token_dict, layout, n_positions, ft.get("grad_checkpointing", False))
    _load_pretrained(bundle / "model_checkpoint.pt", {
        "encoder": encoder, "predictor": predictor, "target_encoder": target_encoder},
        n_positions)
    target_encoder.requires_grad_(False)
    target_encoder.eval()
    encoder, predictor = _configure_trainable(encoder, predictor, ft, token_dict, bundle, device)
    for model in (encoder, predictor, target_encoder):
        model.to(device)

    trainer = MultiStreamTrainer(
        encoder, predictor, target_encoder, objective,
        *_optimizer(encoder, predictor, ft, len(loader)),
        ft=ft, device=device, ipe=len(loader), stream_names=[s.name for s in streams])
    if ft.get("resume_from"):
        trainer.resume(ft["resume_from"])
    trainer.fit(loader, sampler, run_dir, log_fn)
    logger.info("Multi-stream fine-tuning complete: %s", run_dir)
    return str(run_dir)

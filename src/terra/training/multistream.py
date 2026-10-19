"""Streams, batching and batch views for multi-stream alignment fine-tuning.

Three stream kinds share one model and one sequence layout:

- ``spatial_full``: spatial cells with their neighbours (standard JEPA view).
- ``spatial_cell_only``: the same spatial cells with neighbours hidden; the
  batch also carries the untouched full view as the consistency target.
- ``cell_only``: cells without neighbours (Tahoe-100M).

Every batch comes from a single stream (the mask collator trims masks to the
shortest row, so mixing short cell-only rows with long spatial rows would
shrink the spatial context). Cell-only views are cut to the batch's last real
token: records are padded to the full layout, and the trimmed tail is pure
padding (exact, see the tests).
"""

from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from terra.masks.block_masking import BlockMaskCollator
from terra.masks.utils import trim_trailing_pad

__all__ = [
    "STREAM_KINDS",
    "StreamSpec",
    "parse_streams",
    "StreamTaggedDataset",
    "StreamBatchSampler",
    "View",
    "StreamBatch",
    "MultiStreamCollator",
    "trim_trailing_pad",
    "load_stream_dataset",
    "build_multistream_loader",
]

STREAM_KINDS = ("spatial_full", "spatial_cell_only", "cell_only")


@dataclass(frozen=True)
class StreamSpec:
    """One training stream (`data.streams.<name>` in the fine-tune config)."""

    name: str
    kind: str
    paths: tuple[str, ...]
    weight: float = 1.0
    batch_size: int | None = None
    tokenized_seq_len_cell: int | None = None
    tokenized_seq_len_neigh: int | None = None


def parse_streams(args: dict) -> list[StreamSpec] | None:
    """
    Stream specs from fine-tune args.

    Returns `None` when `data.streams` is absent (legacy single-stream
    fine-tuning).
    """
    streams = (args.get("data") or {}).get("streams")
    if not streams:
        return None
    specs = []
    for name, cfg in streams.items():
        kind = cfg.get("kind")
        if kind not in STREAM_KINDS:
            raise ValueError(f"Stream {name!r}: kind must be one of {STREAM_KINDS}, got {kind!r}.")
        paths = tuple(cfg.get("paths") or ())
        if not paths:
            raise ValueError(f"Stream {name!r}: 'paths' must list at least one dataset.")
        weight = float(cfg.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Stream {name!r}: weight must be positive, got {weight}.")
        specs.append(StreamSpec(
            name=name, kind=kind, paths=paths, weight=weight,
            batch_size=cfg.get("batch_size"),
            tokenized_seq_len_cell=cfg.get("tokenized_seq_len_cell"),
            tokenized_seq_len_neigh=cfg.get("tokenized_seq_len_neigh")))
    return specs


class StreamTaggedDataset(Dataset):
    """Wraps a cell dataset and tags every item with its stream index."""

    def __init__(self, dataset: Dataset, stream_id: int):
        self.dataset = dataset
        self.stream_id = stream_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return {**self.dataset[index], "stream_id": self.stream_id}


class StreamBatchSampler(Sampler):
    """
    Batches of global indices into a `ConcatDataset` of streams.

    Every batch comes from one stream.

    Optimizer steps per epoch are split between streams in proportion to
    `weights` (a stream with fewer batches cycles through them). Within a
    stream, indices are shuffled and, if `sort_keys` is given, sorted by key
    inside mega-batches of `mega_batch` batches so rows of similar length
    share a batch. Seeded per epoch (`set_epoch`).
    """

    def __init__(self,
                 lengths: list[int],
                 batch_sizes: list[int],
                 weights: list[float],
                 seed: int = 0,
                 sort_keys: list[list[int]] | None = None,
                 epoch_batches: int | None = None,
                 mega_batch: int = 50):
        self.lengths = list(lengths)
        self.batch_sizes = list(batch_sizes)
        self.weights = list(weights)
        self.seed = seed
        self.sort_keys = sort_keys
        self.epoch_batches = epoch_batches
        self.mega_batch = mega_batch
        self.offsets = [sum(self.lengths[:s]) for s in range(len(self.lengths))]
        self.epoch = 0
        for name, size, length in zip(range(len(lengths)), batch_sizes, lengths, strict=True):
            if length < size:
                raise ValueError(f"Stream {name} has {length} rows, fewer than its "
                                 f"batch size {size}.")

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for a new epoch."""
        self.epoch = epoch

    def _available(self) -> list[int]:
        return [n // b for n, b in zip(self.lengths, self.batch_sizes, strict=True)]

    def _steps(self) -> list[int]:
        """
        Steps per stream, proportional to weights.

        Sums exactly to the epoch length (largest-remainder rounding).
        """
        total = self.epoch_batches or sum(self._available())
        shares = [total * w / sum(self.weights) for w in self.weights]
        steps = [int(share) for share in shares]
        by_remainder = sorted(range(len(shares)), key=lambda s: shares[s] - steps[s],
                              reverse=True)
        for s in by_remainder[:total - sum(steps)]:
            steps[s] += 1
        return steps

    def _stream_batches(self, s: int, generator: torch.Generator) -> list[list[int]]:
        order = torch.randperm(self.lengths[s], generator=generator).tolist()
        size = self.batch_sizes[s]
        if self.sort_keys is not None:
            keys = self.sort_keys[s]
            chunk = self.mega_batch * size
            order = [i for start in range(0, len(order), chunk)
                     for i in sorted(order[start:start + chunk], key=lambda i: -keys[i])]
        offset = self.offsets[s]
        return [[offset + i for i in order[start:start + size]]
                for start in range(0, len(order) - size + 1, size)]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        schedule = []
        for s, n_steps in enumerate(self._steps()):
            batches = self._stream_batches(s, generator)
            schedule += [batches[i % len(batches)] for i in range(n_steps)]
        for i in torch.randperm(len(schedule), generator=generator).tolist():
            yield schedule[i]

    def __len__(self) -> int:
        return sum(self._steps())


@dataclass(frozen=True)
class View:
    """One model input view: features, block masks and attention mask."""

    udata: dict
    masks_enc: list[torch.Tensor] | None
    masks_pred: list[torch.Tensor] | None
    attn: torch.Tensor

    def to(self, device: torch.device) -> "View":
        """Copy on `device`."""
        def move(value):
            return value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        masks = [None if m is None else [move(t) for t in m]
                 for m in (self.masks_enc, self.masks_pred)]
        return View({k: move(v) for k, v in self.udata.items()}, masks[0], masks[1],
                    move(self.attn))


@dataclass(frozen=True)
class StreamBatch:
    """A collated single-stream batch."""

    stream: int
    kind: str
    jepa: View
    full: View | None = None

    def to(self, device: torch.device) -> "StreamBatch":
        """Copy on `device`."""
        return StreamBatch(self.stream, self.kind, self.jepa.to(device),
                           None if self.full is None else self.full.to(device))


def _attention(udata: dict) -> torch.Tensor:
    return (udata["tokens"] != 0).unsqueeze(1).unsqueeze(1)


class MultiStreamCollator:
    """Collate a single-stream batch into the views its stream kind needs."""

    def __init__(self, masker: BlockMaskCollator, kinds: list[str], trim_trailing_pad: bool = True):
        self.masker = masker
        self.kinds = list(kinds)
        self.trim = trim_trailing_pad

    def __call__(self, items: list[dict]) -> StreamBatch:
        """Collate items of one stream into a `StreamBatch`."""
        stream_ids = {item["stream_id"] for item in items}
        if len(stream_ids) != 1:
            raise ValueError(f"Batch mixes streams {sorted(stream_ids)}; batches must "
                             "come from one stream.")
        stream = stream_ids.pop()
        kind = self.kinds[stream]
        items = [{k: v for k, v in item.items() if k != "stream_id"} for item in items]
        if kind == "spatial_full":
            udata, masks_enc, masks_pred, attn, _ = self.masker(items)
            return StreamBatch(stream, kind, View(udata, list(masks_enc), list(masks_pred), attn))
        collated = torch.utils.data.default_collate(items)
        full = None
        if kind == "spatial_cell_only":
            full = View(collated, None, None, _attention(collated))
            collated = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in collated.items()}
            self.masker.zero_beyond(collated, self.masker.segment_cutoff(1))
        if self.trim:
            collated = trim_trailing_pad(collated)
        masks_enc, masks_pred = self.masker.sample_masks_for_tokens(collated["tokens"])
        jepa = View(collated, list(masks_enc), list(masks_pred), _attention(collated))
        return StreamBatch(stream, kind, jepa, full)


def load_stream_dataset(paths: tuple[str, ...], required_columns: list[str] = ()):
    """
    Concatenate a stream's tokenized datasets (torch format).

    Raises if label columns the objective needs are missing.
    """
    dataset = concatenate_datasets([load_from_disk(path) for path in paths])
    missing = [c for c in required_columns if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Datasets {list(paths)} lack columns {missing}; add them "
                         "with `python -m terra.tahoe.prepare spatial|tahoe`.")
    return dataset.with_format("torch")


def build_multistream_loader(cell_datasets: list[Dataset],
                             specs: list[StreamSpec],
                             masker: BlockMaskCollator,
                             batch_size: int,
                             epoch_batches: int | None = None,
                             length_grouped: bool = True,
                             trim: bool = True,
                             num_workers: int = 0,
                             pin_memory: bool = False,
                             seed: int = 0,
                             ) -> tuple[DataLoader, StreamBatchSampler]:
    """
    Loader yielding one `StreamBatch` per step.

    `cell_datasets[s]` serves stream `specs[s]`; a spec's `batch_size`
    overrides the default. With `length_grouped`, rows of similar non-zero
    token counts share a batch, which shortens trimmed cell-only views.
    """
    sort_keys = ([[int(n) for n in ds.n_nz_tokens] for ds in cell_datasets]
                 if length_grouped else None)
    sampler = StreamBatchSampler(
        lengths=[len(ds) for ds in cell_datasets],
        batch_sizes=[spec.batch_size or batch_size for spec in specs],
        weights=[spec.weight for spec in specs],
        seed=seed, sort_keys=sort_keys, epoch_batches=epoch_batches)
    collator = MultiStreamCollator(masker, [spec.kind for spec in specs],
                                   trim_trailing_pad=trim)
    tagged = ConcatDataset([StreamTaggedDataset(ds, s) for s, ds in enumerate(cell_datasets)])
    loader = DataLoader(tagged, batch_sampler=sampler, collate_fn=collator,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, sampler

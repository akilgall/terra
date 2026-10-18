"""Segment layout of a TERRA cell-graph token sequence.

A sequence holds the centre (index) cell first, then its ``n_neighs`` nearest
neighbours, each cell occupying one contiguous segment of gene-token slots.
The pretrained TERRA layout is uniform (every segment has ``seq_len_cell``
slots). The asymmetric layout gives the centre cell ``seq_len_cell`` slots and
every neighbour ``seq_len_neigh`` slots, so a whole-transcriptome centre cell
can keep many more genes than its neighbours.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "SegmentLayout",
    "segment_gather_index",
]


@dataclass(frozen=True)
class SegmentLayout:
    """
    Where each cell's gene tokens sit in a sequence.

    Parameters
    ----------
    seq_len_cell:
        Gene-token slots of the centre cell (segment 1).
    seq_len_neigh:
        Gene-token slots of each neighbour segment (segments 2..).
    n_neighs:
        Number of neighbour segments (0 for cell-only sequences).
    n_special_tokens:
        Special tokens prepended before the gene tokens.
    """

    seq_len_cell: int
    seq_len_neigh: int
    n_neighs: int
    n_special_tokens: int = 0

    def __post_init__(self):
        if self.seq_len_cell <= 0 or self.seq_len_neigh <= 0:
            raise ValueError(
                "seq_len_cell and seq_len_neigh must be positive, got "
                f"{self.seq_len_cell} and {self.seq_len_neigh}.")
        if self.n_neighs < 0 or self.n_special_tokens < 0:
            raise ValueError(
                "n_neighs and n_special_tokens must be non-negative, got "
                f"{self.n_neighs} and {self.n_special_tokens}.")

    @property
    def lengths(self) -> tuple[int, ...]:
        """Slots per segment, centre first."""
        return (self.seq_len_cell,) + (self.seq_len_neigh,) * self.n_neighs

    @property
    def offsets(self) -> tuple[int, ...]:
        """
        Segment boundaries, excluding special tokens.

        Segment ``s`` (1-based) spans ``offsets[s-1]:offsets[s]``.
        """
        starts = [0]
        for length in self.lengths:
            starts.append(starts[-1] + length)
        return tuple(starts)

    @property
    def total(self) -> int:
        """Gene-token slots over all segments."""
        return self.seq_len_cell + self.n_neighs * self.seq_len_neigh

    @property
    def n_segments(self) -> int:
        """Centre cell plus neighbours."""
        return self.n_neighs + 1

    @property
    def seq_len(self) -> int:
        """Full sequence length including special tokens."""
        return self.total + self.n_special_tokens

    @property
    def is_uniform(self) -> bool:
        """Whether every segment has the same number of slots."""
        return self.seq_len_cell == self.seq_len_neigh

    def _check_segment(self, segment: int) -> None:
        if not 1 <= segment <= self.n_segments:
            raise ValueError(
                f"segment must be in [1, {self.n_segments}], got {segment}.")

    def cutoff(self, k: int) -> int:
        """Sequence index after the first ``k`` segments (incl. specials)."""
        if not 1 <= k <= self.n_segments:
            raise ValueError(
                f"k must be in [1, {self.n_segments}] segments, got {k}.")
        return self.n_special_tokens + self.offsets[k]

    def segment_slice(self, segment: int) -> slice:
        """Gene-token slice of a 1-based segment (excluding specials)."""
        self._check_segment(segment)
        return slice(self.offsets[segment - 1], self.offsets[segment])

    def segment_ids(self, gene_tokens: torch.Tensor) -> torch.Tensor:
        """1-based segment id per slot, 0 where the slot holds padding."""
        if gene_tokens.numel() != self.total:
            raise ValueError(
                f"gene_tokens length {gene_tokens.numel()} does not match the "
                f"layout length {self.total}.")
        ids = torch.repeat_interleave(
            torch.arange(1, self.n_segments + 1),
            torch.tensor(self.lengths))
        return ids * (gene_tokens != 0).long()

    def expand_per_segment(self, values: torch.Tensor) -> torch.Tensor:
        """Repeat one value per segment across that segment's slots."""
        if values.numel() != self.n_segments:
            raise ValueError(
                f"expected {self.n_segments} per-segment values, got "
                f"{values.numel()}.")
        return torch.repeat_interleave(values, torch.tensor(self.lengths))

    @classmethod
    def from_lengths(cls,
                     seq_len_cell: int,
                     seq_len_neighborhood: int,
                     seq_len_neigh: int | None = None,
                     n_special_tokens: int = 0,
                     ) -> "SegmentLayout":
        """
        Build a layout from the dataset/model length arguments.

        Parameters
        ----------
        seq_len_cell:
            Centre-cell slots.
        seq_len_neighborhood:
            Slots over all neighbours together.
        seq_len_neigh:
            Slots per neighbour. ``None`` means the legacy uniform layout
            (``seq_len_neigh = seq_len_cell``).
        n_special_tokens:
            Special tokens prepended before the gene tokens.
        """
        neigh = seq_len_cell if seq_len_neigh is None else seq_len_neigh
        if neigh <= 0 or seq_len_neighborhood < 0:
            raise ValueError(
                f"Invalid lengths: seq_len_neigh={neigh}, "
                f"seq_len_neighborhood={seq_len_neighborhood}.")
        if seq_len_neighborhood % neigh != 0:
            raise ValueError(
                f"seq_len_neighborhood {seq_len_neighborhood} is not a "
                f"multiple of seq_len_neigh {neigh}.")
        return cls(seq_len_cell=seq_len_cell, seq_len_neigh=neigh,
                   n_neighs=seq_len_neighborhood // neigh,
                   n_special_tokens=n_special_tokens)

    @classmethod
    def from_data_config(cls,
                         data_cfg: Mapping,
                         overrides: Mapping | None = None,
                         n_special_tokens: int = 0,
                         ) -> "SegmentLayout":
        """
        Build a layout from a ``data`` config section and cross-check it.

        Parameters
        ----------
        data_cfg:
            ``data`` section of a model config (``seq_len_cell``,
            ``seq_len_neighborhood`` and optionally ``seq_len_neigh``,
            ``n_neighs``, ``n_segments``, ``model_input_size``).
        overrides:
            Keys that take precedence over ``data_cfg`` (e.g. fine-tune
            arguments switching to an asymmetric layout).
        n_special_tokens:
            Special tokens prepended before the gene tokens.
        """
        cfg = {**data_cfg, **(overrides or {})}
        neigh = cfg.get("seq_len_neigh")
        if neigh is not None and "seq_len_neighborhood" in (overrides or {}):
            neighborhood = cfg["seq_len_neighborhood"]
        elif neigh is not None and "n_neighs" in cfg:
            neighborhood = cfg["n_neighs"] * neigh
        else:
            neighborhood = cfg["seq_len_neighborhood"]
        layout = cls.from_lengths(cfg["seq_len_cell"], neighborhood,
                                  seq_len_neigh=neigh,
                                  n_special_tokens=n_special_tokens)
        checks = {"n_neighs": layout.n_neighs,
                  "n_segments": layout.n_segments,
                  "model_input_size": layout.total}
        for key, expected in checks.items():
            if key in cfg and cfg[key] is not None and int(cfg[key]) != expected:
                raise ValueError(
                    f"Inconsistent data config: {key}={cfg[key]} but the "
                    f"layout {layout.lengths} implies {expected}.")
        if neigh is not None and "seq_len_neighborhood" in cfg and \
                int(cfg["seq_len_neighborhood"]) != layout.n_neighs * layout.seq_len_neigh:
            raise ValueError(
                f"Inconsistent data config: seq_len_neighborhood="
                f"{cfg['seq_len_neighborhood']} but n_neighs * seq_len_neigh = "
                f"{layout.n_neighs * layout.seq_len_neigh}.")
        return layout


def segment_gather_index(stored_lengths: Sequence[int],
                         keep_lengths: Sequence[int],
                         ) -> torch.Tensor:
    """
    Gather index that shrinks a stored record segment by segment.

    Keeps the first ``keep_lengths[s]`` slots of each stored segment and drops
    trailing segments that are not kept.

    Parameters
    ----------
    stored_lengths:
        Slots per segment as stored in the tokenized record.
    keep_lengths:
        Slots per segment to keep (one entry per kept segment).
    """
    if len(keep_lengths) > len(stored_lengths):
        raise ValueError(
            f"cannot keep {len(keep_lengths)} segments from a record with "
            f"{len(stored_lengths)}.")
    pieces = []
    start = 0
    # keep_lengths may be shorter than stored_lengths (trailing segments dropped).
    for s, (stored, keep) in enumerate(zip(stored_lengths, keep_lengths, strict=False)):
        if keep > stored:
            raise ValueError(
                f"segment {s + 1}: cannot keep {keep} slots from {stored} "
                "stored; seq_len_cell / seq_len_neigh exceed the tokenized "
                "lengths.")
        pieces.append(torch.arange(start, start + keep))
        start += stored
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)

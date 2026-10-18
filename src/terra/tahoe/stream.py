"""Targeted reads of Tahoe-100M parquet shards on the Hugging Face Hub.

The ``expression_data`` config is 3,388 parquet shards (~95.6M cells, ~338
GB). Shards are grouped by plate; every 1,000-row row group spans a few
consecutive samples (wells) and mixes ~50 cell lines. Parquet footers carry
per-row-group min/max statistics of ``plate`` and ``sample``, so the row groups
holding particular wells (e.g. a plate's DMSO controls, its last wells) can be
found from footers alone and read individually, without streaming whole
shards.

Files are opened through an fsspec filesystem (``huggingface_hub.HfFileSystem``
for the Hub) in its default seekable mode: parquet needs range reads, and the
non-seekable streaming mode raises "Cannot seek streaming HF file".
"""

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

__all__ = [
    "REPO_ID",
    "RowGroupInfo",
    "sample_number",
    "list_shards",
    "read_footer",
    "find_row_groups",
    "read_row_groups",
]

REPO_ID = "tahoebio/Tahoe-100M"
_SHARD_GLOB = f"datasets/{REPO_ID}/data/train-*.parquet"
_SAMPLE_PATTERN = re.compile(r"^smp_(\d+)$")


@dataclass(frozen=True)
class RowGroupInfo:
    """Location and sample range of one parquet row group."""

    shard: str
    row_group: int
    plate: str
    sample_min: int
    sample_max: int
    n_rows: int

    def covers(self, sample: int) -> bool:
        """Whether the row group's sample range includes `sample`."""
        return self.sample_min <= sample <= self.sample_max


def sample_number(sample: str) -> int:
    """`'smp_1783'` -> 1783."""
    match = _SAMPLE_PATTERN.match(sample)
    if match is None:
        raise ValueError(f"Not a Tahoe sample id: {sample!r}.")
    return int(match.group(1))


def list_shards(fs) -> list[str]:
    """Sorted paths of the `expression_data` shards on the Hub."""
    return sorted(fs.glob(_SHARD_GLOB))


def _column_stats(row_group, name: str):
    for i in range(row_group.num_columns):
        column = row_group.column(i)
        if column.path_in_schema == name:
            return column.statistics
    raise ValueError(f"Column {name!r} not found in parquet row group.")


def read_footer(shard: str, fs) -> list[RowGroupInfo]:
    """Row-group sample ranges of one shard, from its parquet footer."""
    with fs.open(shard, "rb") as handle:
        metadata = pq.ParquetFile(handle).metadata
    infos = []
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        plate = _column_stats(row_group, "plate")
        sample = _column_stats(row_group, "sample")
        if plate is None or sample is None or not sample.has_min_max:
            raise ValueError(f"{shard}: row group {index} has no plate/sample statistics.")
        infos.append(RowGroupInfo(
            shard=shard, row_group=index, plate=str(plate.min),
            sample_min=sample_number(str(sample.min)),
            sample_max=sample_number(str(sample.max)),
            n_rows=row_group.num_rows))
    return infos


def find_row_groups(shards: Iterable[str],
                    wanted_samples: set[int],
                    footer_fn: Callable[[str], list[RowGroupInfo]],
                    max_shards: int,
                    ) -> list[RowGroupInfo]:
    """
    Collect the row groups that cover any wanted sample.

    Scans shard footers in order and stops once every wanted sample is
    covered or after `max_shards` footers.
    """
    found, covered = [], set()
    for count, shard in enumerate(shards):
        if count >= max_shards:
            break
        for info in footer_fn(shard):
            hits = {s for s in wanted_samples if info.covers(s)}
            if hits:
                found.append(info)
                covered |= hits
        if covered >= wanted_samples:
            break
    if covered != wanted_samples:
        logger.warning(
            f"Samples not found in the scanned shards: {sorted(wanted_samples - covered)}.")
    return found


def read_row_groups(infos: Iterable[RowGroupInfo], fs) -> list[dict]:
    """Read row groups and return their rows as plain dicts."""
    records = []
    for info in infos:
        with fs.open(info.shard, "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(info.row_group)
        records.extend(table.to_pylist())
    return records

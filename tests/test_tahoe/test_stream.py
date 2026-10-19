"""``terra.tahoe.stream``: row-group footers and targeted reads.

Offline tests write tiny parquet shards locally (same schema columns as
Tahoe-100M) and read them through an fsspec filesystem. The network test is
opt-in (``TERRA_TAHOE_NET=1``).
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem

from terra.tahoe.stream import RowGroupInfo, find_row_groups, read_footer, read_row_groups, sample_number


def _write_shard(path, samples_per_group):
    """One row group per entry; each row group holds the given samples."""
    writer = None
    for samples in samples_per_group:
        rows = {"genes": [[1, 3]] * len(samples), "expressions": [[-2.0, 1.0]] * len(samples),
                "drug": ["DMSO_TF" if s == "smp_9" else "Drug" for s in samples],
                "sample": samples, "BARCODE_SUB_LIB_ID": [f"{path.name}-{s}" for s in samples],
                "cell_line_id": ["CVCL_0023"] * len(samples), "moa-fine": ["x"] * len(samples),
                "canonical_smiles": ["C"] * len(samples), "pubchem_cid": ["1"] * len(samples),
                "plate": ["plate1"] * len(samples)}
        table = pa.table(rows)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema)
        writer.write_table(table)
    writer.close()
    return str(path)


@pytest.fixture
def shards(tmp_path):
    return [
        _write_shard(tmp_path / "s0.parquet", [["smp_1", "smp_2"], ["smp_2", "smp_3"]]),
        _write_shard(tmp_path / "s1.parquet", [["smp_7", "smp_8"], ["smp_8", "smp_9"]]),
    ]


def test_sample_number():
    assert sample_number("smp_1783") == 1783
    with pytest.raises(ValueError):
        sample_number("plate4")


def test_read_footer_uses_statistics(shards):
    infos = read_footer(shards[1], LocalFileSystem())

    assert infos == [
        RowGroupInfo(shards[1], 0, "plate1", 7, 8, 2),
        RowGroupInfo(shards[1], 1, "plate1", 8, 9, 2),
    ]


def test_find_row_groups_scans_until_samples_covered(shards):
    calls = []

    def footer(shard):
        calls.append(shard)
        return read_footer(shard, LocalFileSystem())

    found = find_row_groups(shards, wanted_samples={9}, footer_fn=footer, max_shards=5)

    assert [(i.shard, i.row_group) for i in found] == [(shards[1], 1)]
    assert calls == shards


def test_find_row_groups_respects_scan_limit(shards):
    found = find_row_groups(shards, wanted_samples={9},
                            footer_fn=lambda s: read_footer(s, LocalFileSystem()), max_shards=1)

    assert found == []


def test_read_row_groups_returns_records(shards):
    info = RowGroupInfo(shards[1], 1, "plate1", 8, 9, 2)

    records = read_row_groups([info], LocalFileSystem())

    assert [r["sample"] for r in records] == ["smp_8", "smp_9"]
    assert records[1]["drug"] == "DMSO_TF"
    assert records[0]["genes"] == [1, 3]


@pytest.mark.skipif(not os.environ.get("TERRA_TAHOE_NET"),
                    reason="set TERRA_TAHOE_NET=1 to read one Tahoe-100M row group")
def test_reads_one_row_group_from_hub():
    from huggingface_hub import HfFileSystem

    from terra.tahoe.stream import list_shards

    fs = HfFileSystem()
    shard = list_shards(fs)[0]
    infos = read_footer(shard, fs)
    records = read_row_groups(infos[:1], fs)

    assert len(infos) > 0 and len(records) == infos[0].n_rows
    assert records[0]["genes"][0] == 1 and records[0]["expressions"][0] < 0

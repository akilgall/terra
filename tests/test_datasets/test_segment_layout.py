"""Tests for ``terra.datasets.segment_layout``.

``SegmentLayout`` describes where each cell's gene tokens sit in a TERRA
sequence: the centre cell first (``seq_len_cell`` slots), then ``n_neighs``
neighbour segments of ``seq_len_neigh`` slots each. The pretrained TERRA-96M
layout is uniform (11 x 256); the asymmetric layout gives the centre cell a
larger budget (e.g. 2048) while neighbours keep 256.
"""

import pytest
import torch

from terra.datasets.segment_layout import SegmentLayout, segment_gather_index

TERRA_96M_DATA = {
    "seq_len_cell": 256,
    "seq_len_neighborhood": 2560,
    "model_input_size": 2816,
    "n_neighs": 10,
    "n_segments": 11,
}


def test_uniform_layout_matches_terra_96m():
    layout = SegmentLayout.from_lengths(seq_len_cell=256, seq_len_neighborhood=2560)

    assert layout.seq_len_neigh == 256
    assert layout.n_neighs == 10
    assert layout.n_segments == 11
    assert layout.total == 2816
    assert layout.is_uniform
    assert layout.lengths == (256,) * 11
    assert layout.offsets == tuple(256 * s for s in range(12))


@pytest.mark.parametrize("k", range(1, 12))
def test_uniform_cutoff_matches_legacy_formula(k):
    layout = SegmentLayout.from_lengths(256, 2560, n_special_tokens=2)

    assert layout.cutoff(k) == 2 + 256 * k


def test_asymmetric_layout_offsets():
    layout = SegmentLayout.from_lengths(seq_len_cell=2048, seq_len_neighborhood=2560,
                                        seq_len_neigh=256)

    assert layout.lengths == (2048,) + (256,) * 10
    assert layout.offsets == (0, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
                              4096, 4352, 4608)
    assert layout.total == 4608
    assert layout.seq_len == 4608
    assert not layout.is_uniform
    assert layout.cutoff(1) == 2048
    assert layout.cutoff(11) == 4608


def test_cell_only_layout_has_no_neighbours():
    layout = SegmentLayout.from_lengths(seq_len_cell=2048, seq_len_neighborhood=0,
                                        seq_len_neigh=256)

    assert layout.n_neighs == 0
    assert layout.n_segments == 1
    assert layout.total == 2048


def test_segment_slice_is_one_based():
    layout = SegmentLayout.from_lengths(8, 6, seq_len_neigh=3)

    assert layout.segment_slice(1) == slice(0, 8)
    assert layout.segment_slice(2) == slice(8, 11)
    assert layout.segment_slice(3) == slice(11, 14)
    with pytest.raises(ValueError, match="segment"):
        layout.segment_slice(0)
    with pytest.raises(ValueError, match="segment"):
        layout.segment_slice(4)


def test_cutoff_rejects_out_of_range():
    layout = SegmentLayout.from_lengths(8, 6, seq_len_neigh=3)

    with pytest.raises(ValueError, match="segments"):
        layout.cutoff(0)
    with pytest.raises(ValueError, match="segments"):
        layout.cutoff(4)


def test_segment_ids_mask_padding():
    layout = SegmentLayout.from_lengths(4, 4, seq_len_neigh=2)
    tokens = torch.tensor([7, 8, 0, 0, 9, 0, 0, 0])

    ids = layout.segment_ids(tokens)

    assert ids.tolist() == [1, 1, 0, 0, 2, 0, 0, 0]
    assert ids.dtype == torch.long


def test_segment_ids_rejects_wrong_length():
    layout = SegmentLayout.from_lengths(4, 4, seq_len_neigh=2)

    with pytest.raises(ValueError, match="length"):
        layout.segment_ids(torch.ones(7, dtype=torch.long))


def test_expand_per_segment():
    layout = SegmentLayout.from_lengths(3, 4, seq_len_neigh=2)

    out = layout.expand_per_segment(torch.tensor([0.0, 1.5, -2.0]))

    assert out.tolist() == [0.0, 0.0, 0.0, 1.5, 1.5, -2.0, -2.0]


def test_from_data_config_on_terra_96m_section():
    layout = SegmentLayout.from_data_config(dict(TERRA_96M_DATA))

    assert layout.is_uniform
    assert layout.total == 2816


def test_from_data_config_with_overrides():
    layout = SegmentLayout.from_data_config(
        dict(TERRA_96M_DATA),
        overrides={"seq_len_cell": 2048, "seq_len_neigh": 256, "model_input_size": 4608})

    assert layout.lengths[0] == 2048
    assert layout.total == 4608


@pytest.mark.parametrize("bad", [
    {"n_segments": 12},
    {"model_input_size": 4608},
    {"n_neighs": 9},
    {"seq_len_neigh": 300},
])
def test_from_data_config_rejects_inconsistent_keys(bad):
    cfg = dict(TERRA_96M_DATA)
    cfg.update(bad)

    with pytest.raises(ValueError):
        SegmentLayout.from_data_config(cfg)


@pytest.mark.parametrize("kwargs", [
    {"seq_len_cell": 0, "seq_len_neighborhood": 10},
    {"seq_len_cell": 4, "seq_len_neighborhood": 10, "seq_len_neigh": 3},
    {"seq_len_cell": 4, "seq_len_neighborhood": -4},
])
def test_from_lengths_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        SegmentLayout.from_lengths(**kwargs)


def test_segment_gather_index_uniform_matches_legacy_gather():
    stride, keep, n_seg = 4, 2, 3
    expected = torch.cat([torch.arange(s * stride, s * stride + keep) for s in range(n_seg)])

    idx = segment_gather_index([stride] * n_seg, [keep] * n_seg)

    assert torch.equal(idx, expected)


def test_segment_gather_index_asymmetric():
    idx = segment_gather_index([10, 4, 4], [8, 3, 3])

    assert idx.tolist() == list(range(8)) + [10, 11, 12] + [14, 15, 16]


def test_segment_gather_index_keeps_fewer_segments():
    idx = segment_gather_index([4, 2, 2], [4, 2])

    assert idx.tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("stored,keep", [([4, 2], [5, 2]), ([4, 2], [4, 2, 2])])
def test_segment_gather_index_rejects_growth(stored, keep):
    with pytest.raises(ValueError):
        segment_gather_index(stored, keep)

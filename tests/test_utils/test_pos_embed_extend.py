"""``extend_pos_embed_state_dict``: grow frozen sincos position tables.

Only needed when the centre budget exceeds the pretrained table (2,816
positions for TERRA-96M). The pretrained rows must be untouched sincos values;
the new tail is the deterministic sincos continuation.
"""

import pytest
import torch

from terra.models.utils import get_1d_sincos_pos_embed
from terra.utils.helper import extend_pos_embed_state_dict

DIM = 16


def _table(n_positions: int) -> torch.Tensor:
    rows = torch.from_numpy(get_1d_sincos_pos_embed(
        embed_dim=DIM, n_zero_pos=0, n_sincos_pos=n_positions)).float()
    return torch.cat([torch.zeros(1, DIM), rows])


def _state(n_positions: int) -> dict:
    return {"backbone.pos_embed.weight": _table(n_positions),
            "backbone.token_embed.weight": torch.randn(5, DIM)}


def test_extends_table_with_sincos_tail():
    state = _state(12)

    extended = extend_pos_embed_state_dict(state, n_positions=24)

    table = extended["backbone.pos_embed.weight"]
    assert table.shape == (25, DIM)
    assert torch.equal(table[:13], state["backbone.pos_embed.weight"])
    torch.testing.assert_close(table, _table(24))


def test_does_not_mutate_input():
    state = _state(12)
    before = {k: v.clone() for k, v in state.items()}

    extended = extend_pos_embed_state_dict(state, n_positions=24)

    assert extended is not state
    for key, value in before.items():
        assert torch.equal(state[key], value)
    assert extended["backbone.token_embed.weight"] is state["backbone.token_embed.weight"]


def test_tables_already_large_enough_are_kept():
    state = _state(24)

    extended = extend_pos_embed_state_dict(state, n_positions=12)

    assert extended["backbone.pos_embed.weight"] is state["backbone.pos_embed.weight"]


def test_small_drift_is_accepted():
    state = _state(12)
    state["backbone.pos_embed.weight"] = state["backbone.pos_embed.weight"] + 1e-4

    extended = extend_pos_embed_state_dict(state, n_positions=24)

    assert extended["backbone.pos_embed.weight"].shape == (25, DIM)


def test_tampered_table_is_refused():
    state = _state(12)
    state["backbone.pos_embed.weight"][5, 3] += 0.5

    with pytest.raises(ValueError, match="sincos"):
        extend_pos_embed_state_dict(state, n_positions=24)

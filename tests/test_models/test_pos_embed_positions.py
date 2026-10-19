"""Explicit centre length and position-table size on the encoder.

The pretrained TERRA-96M encoder derives ``seq_len_cell`` by dividing the
sequence evenly over its segments and sizes its frozen sincos position table
by ``seq_len``. For the asymmetric layout the centre length is passed
explicitly, and ``n_positions`` decouples the table size from ``seq_len`` so
the pretrained 2,817-row tables still load strictly. Positions are per-segment
ranks, so the largest index used is the centre budget.
"""

import os

import pytest
import torch

from terra.models.gene_transformers import GeneTransformerRankEncoder
from terra.models.utils import get_1d_sincos_pos_embed
from terra.utils.helper import init_model, parse_arch_kwargs


def _encoder(**overrides) -> GeneTransformerRankEncoder:
    kwargs = {"vocab_size": 20, "seq_len": 14, "n_special_tokens": 0, "n_segments": 3,
              "cell_pos_enc": "segment", "embed_dim": 16, "depth": 1, "num_heads": 2,
              "use_flash_attention": True}
    kwargs.update(overrides)
    return GeneTransformerRankEncoder(**kwargs)


def _asymmetric_batch(batch_size: int = 2, centre: int = 8, neigh: int = 3) -> dict:
    """Two neighbours: [centre][neigh][neigh], every slot filled."""
    torch.manual_seed(1)
    seq_len = centre + 2 * neigh
    tokens = torch.randint(2, 20, (batch_size, seq_len))
    positions = torch.cat([torch.arange(1, centre + 1)] + [torch.arange(1, neigh + 1)] * 2)
    segments = torch.tensor([1] * centre + [2] * neigh + [3] * neigh)
    return {"tokens": tokens,
            "positions": positions.repeat(batch_size, 1),
            "segments": segments.repeat(batch_size, 1)}


def test_seq_len_cell_defaults_to_even_split():
    assert _encoder(seq_len=12).seq_len_cell == 4


def test_explicit_seq_len_cell_overrides_even_split():
    assert _encoder(seq_len_cell=8).seq_len_cell == 8


def test_position_table_defaults_to_seq_len():
    assert _encoder().pos_embed.weight.shape[0] == 15


def test_n_positions_sizes_table_and_loads_pretrained_state():
    pretrained = _encoder(seq_len=12)
    asymmetric = _encoder(seq_len=18, seq_len_cell=8, n_positions=12)

    result = asymmetric.load_state_dict(pretrained.state_dict(), strict=True)

    assert asymmetric.pos_embed.weight.shape[0] == 13
    assert result.missing_keys == [] and result.unexpected_keys == []


def test_sincos_rows_do_not_depend_on_table_size():
    small = get_1d_sincos_pos_embed(embed_dim=16, n_zero_pos=0, n_sincos_pos=12)
    large = get_1d_sincos_pos_embed(embed_dim=16, n_zero_pos=0, n_sincos_pos=40)

    assert torch.allclose(torch.from_numpy(small), torch.from_numpy(large[:12]))


def test_laplacian_rejects_non_uniform_layout():
    with pytest.raises(ValueError, match="uniform"):
        _encoder(cell_pos_enc="laplacian", seq_len_cell=8)


def test_rda_rejects_non_uniform_layout():
    with pytest.raises(ValueError, match="uniform"):
        _encoder(seq_len_cell=8, rda_kwargs={"enabled": True})


def test_laplacian_still_accepts_uniform_layout():
    assert _encoder(cell_pos_enc="laplacian", seq_len=12).seq_len_cell == 4


def test_cell_only_context_uses_explicit_centre_length(device):
    encoder = _encoder(seq_len_cell=8).to(device).eval()
    batch = {k: v.to(device) for k, v in _asymmetric_batch().items()}
    attn = (batch["tokens"] != 0)[:, None, None, :]
    reference = {k: v.clone() for k, v in batch.items()}
    for key in ("tokens", "positions", "segments"):
        reference[key][:, 8:] = 0
    reference_attn = (reference["tokens"] != 0)[:, None, None, :]

    with torch.no_grad():
        _, cell_only = encoder.return_layer_emb(
            layers=[1], batch=batch, masks_attention=attn, need_cell_only_context=True)
        expected, _ = encoder.return_layer_emb(
            layers=[1], batch=reference, masks_attention=reference_attn,
            need_cell_only_context=False)

    torch.testing.assert_close(cell_only[1][:, :8], expected[1][:, :8], rtol=1e-4, atol=1e-4)


def test_asymmetric_forward_matches_cpu(device):
    if device.type == "cpu":
        pytest.skip("reference device")
    torch.manual_seed(0)
    encoder = _encoder(seq_len_cell=8).eval()
    batch = _asymmetric_batch()

    with torch.no_grad():
        expected, _ = encoder(batch)
        actual, _ = encoder.to(device)({k: v.to(device) for k, v in batch.items()})

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not os.environ.get("TERRA_BUNDLE_DIR"),
                    reason="set TERRA_BUNDLE_DIR to a TERRA-96M bundle to run")
def test_terra_96m_weights_load_into_asymmetric_model():
    """Centre 2048 + 10 x 256 neighbours: the pretrained weights (incl. the
    2,817-row position tables) load strictly with n_positions=2816."""
    yaml = pytest.importorskip("yaml")
    bundle = os.environ["TERRA_BUNDLE_DIR"]
    with open(os.path.join(bundle, "model_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    meta, data = cfg["meta"], cfg["data"]
    encoder, _ = init_model(
        gt_type=meta["gt_type"], count_encoding=meta["count_encoding"],
        n_value_bins=meta["n_value_bins"], cell_pos_enc=meta["cell_pos_enc"], device="cpu",
        vocab_size=23407, seq_len=4608, n_special_tokens=0, n_segments=data["n_segments"],
        n_special_values=0, enc_emb_dim=meta["enc_emb_dim"], enc_depth=meta["enc_depth"],
        pred_emb_dim=meta["pred_emb_dim"], pred_depth=meta["pred_depth"],
        num_heads=meta["num_heads"], mlp_ratio=meta["mlp_ratio"],
        use_flash_attention=meta["use_flash_attention"], api_version=meta["api_version"],
        sep_gene_tokens_neb=data["sep_gene_tokens_neb"], predict_gene=meta["predict_gene"],
        pos_learnable=meta["pos_learnable"], mlp_bias=meta.get("mlp_bias", True),
        seq_len_cell=2048, n_positions=2816, **parse_arch_kwargs(cfg))
    checkpoint = torch.load(os.path.join(bundle, "model_checkpoint.pt"), map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in checkpoint["target_encoder"].items()}

    result = encoder.load_state_dict(state, strict=True)

    assert result.missing_keys == [] and result.unexpected_keys == []
    assert encoder.backbone.seq_len_cell == 2048
    assert encoder.backbone.pos_embed.weight.shape[0] == 2817

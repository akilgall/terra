"""Embedding with a model whose config uses the asymmetric layout (as
exported by multi-stream fine-tuning)."""

import shutil

import numpy as np
import pytest
import yaml

datasets = pytest.importorskip("datasets")
pytest.importorskip("anndata")

from tiny_terra import make_adata, make_bundle, skip_without_squidpy  # noqa: E402

from terra.datasets.segment_layout import config_layout_kwargs  # noqa: E402
from terra.inference.embed import embed_dataset  # noqa: E402
from terra.inference.tokenize import tokenize_adata  # noqa: E402
from terra.training.multistream_finetune import resolve_layout, run_model_config  # noqa: E402

TERRA_96M_DATA = {"seq_len_cell": 256, "seq_len_neighborhood": 2560, "n_neighs": 10,
                  "n_segments": 11, "model_input_size": 2816}


def test_layout_kwargs_are_empty_for_legacy_configs():
    kwargs = config_layout_kwargs({"data": TERRA_96M_DATA, "meta": {"special_tokens": []}})

    assert kwargs.model == kwargs.dataset == kwargs.masker == kwargs.selection == {}


def test_layout_kwargs_for_asymmetric_configs():
    config = run_model_config({"data": TERRA_96M_DATA, "meta": {"special_tokens": []}},
                              resolve_layout(TERRA_96M_DATA, {"seq_len_cell": 2048,
                                                              "seq_len_neigh": 256}),
                              n_positions=2816)

    kwargs = config_layout_kwargs(config)

    assert kwargs.model == {"seq_len_cell": 2048, "n_positions": 2816}
    assert kwargs.dataset == {"seq_len_neigh": 256}
    assert kwargs.masker == {"seq_len_neigh": 256}
    assert kwargs.selection == {"seq_len_neigh": 256}


@pytest.fixture
def asymmetric_bundle(tmp_path):
    """Tiny bundle re-described as centre 8 + 2 neighbours x 4 (16 tokens),
    keeping the pretrained 12-row position table."""
    pretrained = make_bundle(tmp_path / "pretrained")
    bundle = tmp_path / "asymmetric"
    shutil.copytree(pretrained, bundle)
    with open(pretrained / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    layout = resolve_layout(config["data"], {"seq_len_cell": 8, "seq_len_neigh": 4})
    with open(bundle / "model_config.yaml", "w") as f:
        yaml.safe_dump(run_model_config(config, layout, n_positions=12), f)
    return bundle


def _tokenize(bundle, tmp_path, spatial):
    if spatial:
        skip_without_squidpy()
    return tokenize_adata(make_adata(8, spatial, seed=3), str(bundle), str(tmp_path / "cache"),
                          nproc=1, processing_mode="sequential", use_generator=False)


def _embed(dataset, bundle, device, **kwargs):
    return embed_dataset(dataset, str(bundle), batch_size=4, num_workers=0,
                         device=str(device), **kwargs)


def test_embeds_spatial_records(asymmetric_bundle, tmp_path, device):
    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=True)

    out = _embed(dataset, asymmetric_bundle, device)

    assert len(dataset[0]["gene_tokens"]) == 16
    assert out["cell_emb"].shape == (8, 32)
    assert np.isfinite(out["cell_emb"]).all() and np.isfinite(out["neighborhood_emb"]).all()


def test_cell_only_records_have_identical_cell_and_spatial_embeddings(asymmetric_bundle,
                                                                      tmp_path):
    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=False)

    out = _embed(dataset, asymmetric_bundle, "cpu")

    # No neighbours: the cell-only context equals the full context only if the
    # encoder masks exactly the 8 centre slots.
    np.testing.assert_allclose(out["cell_emb"], out["spatial_cell_emb"], atol=1e-5)


def test_trim_padding_matches_untrimmed(asymmetric_bundle, tmp_path, device):
    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=False)

    full = _embed(dataset, asymmetric_bundle, device)
    trimmed = _embed(dataset, asymmetric_bundle, device, trim_padding=True)

    np.testing.assert_allclose(trimmed["cell_emb"], full["cell_emb"], atol=1e-5)


def test_trim_padding_rejects_token_embeddings(asymmetric_bundle, tmp_path):
    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=False)

    with pytest.raises(ValueError, match="trim_padding"):
        _embed(dataset, asymmetric_bundle, "cpu", trim_padding=True,
               return_token_embeddings=True)


def test_token_distance_loads_asymmetric_model(asymmetric_bundle, tmp_path):
    pytest.importorskip("geomloss")
    from terra.inference.token_distance import infer_token_distance

    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=True)

    out = infer_token_distance(dataset, dataset, str(asymmetric_bundle), batch_size=4,
                               num_workers=0, loss="energy", device="cpu")

    # Identical inputs: zero up to float32 cancellation in the energy distance.
    for key in ("cell_score", "spatial_cell_score", "neighborhood_score"):
        assert out[key].shape == (8,)
        np.testing.assert_allclose(out[key], 0.0, atol=1e-3)


def test_embedding_runs_in_inference_mode(asymmetric_bundle, tmp_path, monkeypatch):
    import torch

    import terra.inference.embed as embed_module

    dataset = _tokenize(asymmetric_bundle, tmp_path, spatial=False)
    seen = []
    original = embed_module.compute_mean_unmasked_emb

    def spy(*args, **kwargs):
        seen.append(torch.is_inference_mode_enabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(embed_module, "compute_mean_unmasked_emb", spy)

    _embed(dataset, asymmetric_bundle, "cpu", trim_padding=True)

    assert seen and all(seen)

"""Tests for ``terra.utils.helper.load_checkpoint``.

For inference and fine-tuning (``is_training=False``) a checkpoint that
fails to load must raise: silently continuing would run the model on its
random initial weights. Training resumes keep the historical fallback of
starting from epoch 0.
"""

import logging

import pytest
import torch
import torch.nn as nn

from terra.utils.helper import load_checkpoint


def _save_target_encoder_checkpoint(path, module: nn.Module, prefix: str = "module.") -> None:
    state = {f"{prefix}{key}": value for key, value in module.state_dict().items()}
    torch.save({"epoch": 3, "target_encoder": state}, path)


def _load_target_encoder(device, path, target_encoder, is_training=False):
    return load_checkpoint(
        device=device,
        r_path=path,
        encoder=None,
        predictor=None,
        target_encoder=target_encoder,
        opt=None,
        scaler=None,
        is_training=is_training,
    )


def test_loads_ddp_prefixed_target_encoder_onto_device(tmp_path, device):
    torch.manual_seed(0)
    source = nn.Linear(4, 3)
    path = tmp_path / "model_checkpoint.pt"
    _save_target_encoder_checkpoint(path, source)
    target = nn.Linear(4, 3).to(device)

    _, _, loaded, _, _, epoch, _ = _load_target_encoder(str(device), path, target)

    assert epoch == 3
    assert loaded.weight.device.type == device.type
    torch.testing.assert_close(loaded.weight.detach().cpu(), source.weight.detach())
    torch.testing.assert_close(loaded.bias.detach().cpu(), source.bias.detach())


def test_missing_checkpoint_raises_for_inference(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_target_encoder("cpu", tmp_path / "missing.pt", nn.Linear(4, 3))


def test_mismatched_checkpoint_raises_for_inference(tmp_path):
    path = tmp_path / "model_checkpoint.pt"
    _save_target_encoder_checkpoint(path, nn.Linear(4, 3))

    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_target_encoder("cpu", path, nn.Linear(5, 3))


def test_failed_resume_during_training_starts_from_scratch(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="terra.utils.helper"):
        result = _load_target_encoder(
            "cpu", tmp_path / "missing.pt", nn.Linear(4, 3), is_training=True)

    epoch, iter_number = result[5], result[6]
    assert epoch == 0
    assert iter_number is None
    assert any("loading checkpoint" in r.getMessage() for r in caplog.records)

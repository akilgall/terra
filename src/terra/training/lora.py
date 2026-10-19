"""LoRA adapters for self-supervised fine-tuning: wrapping and merging."""

import logging

import torch
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

__all__ = ["apply_lora", "merge_lora_weights", "lora_scale_from_config"]


def apply_lora(model,
               peft_rank=16,
               peft_alpha=256,
               peft_dropout=0.1,
               peft_bias='none',
               peft_target_modules=None):
    """Wrap ``model`` with LoRA adapters and freeze everything else."""
    if peft_target_modules is None:
        peft_target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    peft_config = LoraConfig(r=peft_rank,
                             lora_alpha=peft_alpha,
                             lora_dropout=peft_dropout,
                             bias=peft_bias,
                             task_type=None,
                             target_modules=peft_target_modules)
    peft_model = get_peft_model(model, peft_config)
    for name, p in peft_model.named_parameters():   # train only the adapters
        p.requires_grad = ('lora_' in name.lower())
    return peft_model


def lora_scale_from_config(finetune_cfg: dict) -> float:
    """LoRA update scale ``alpha / rank`` from a ``finetune`` config section."""
    return float(finetune_cfg['peft_alpha']) / float(finetune_cfg['peft_rank'])


def merge_lora_weights(state_dict: dict, scaling: float = 1.0) -> dict:
    """Fold LoRA adapters in a PEFT state dict into the base weights.

    For each adapted layer, ``W = W_base + scaling * (lora_B @ lora_A)`` with
    ``scaling = lora_alpha / r`` (see :func:`lora_scale_from_config`); layers without
    adapters are copied unchanged. The ``base_model.model.`` / ``base_layer``
    PEFT wrapper prefixes are stripped so the result matches the plain model's
    parameter names.
    """
    base_weights, lora_a_weights, lora_b_weights = {}, {}, {}
    for key, value in state_dict.items():
        if 'base_layer.weight' in key or 'base_layer.bias' in key:
            clean_key = key.replace('base_model.model.', '').replace('.base_layer', '')
            base_weights[clean_key] = value
        elif 'lora_A.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_A.default.weight', '.weight')
            lora_a_weights[clean_key] = value
        elif 'lora_B.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_B.default.weight', '.weight')
            lora_b_weights[clean_key] = value
        elif 'base_model.model.' in key:
            clean_key = key.replace('base_model.model.', '')
            base_weights[clean_key] = value

    merged_weights = {}
    n_merged = 0
    for key, base_weight in base_weights.items():
        if key in lora_a_weights and key in lora_b_weights:
            delta = torch.mm(lora_b_weights[key], lora_a_weights[key])
            merged_weights[key] = base_weight + scaling * delta
            n_merged += 1
        else:
            merged_weights[key] = base_weight
    logger.debug("Merged %d LoRA adapter layers (%d total layers)",
                 n_merged, len(merged_weights))
    return merged_weights

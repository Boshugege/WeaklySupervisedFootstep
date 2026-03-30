# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - import guard
    torch = None
    F = None

from .config import DASBandConfig


def _soft_center(prob):
    channels = torch.arange(prob.shape[-1], device=prob.device, dtype=prob.dtype)
    numer = torch.sum(prob * channels[None, None, :], dim=-1)
    denom = torch.sum(prob, dim=-1) + 1e-6
    return numer / denom


def compute_losses(logits, target, center_target, center_weight, config: DASBandConfig) -> Dict[str, "torch.Tensor"]:
    prob = torch.sigmoid(logits[:, 0])
    mask_loss = F.binary_cross_entropy_with_logits(logits, target)

    mu = _soft_center(prob)
    valid = center_weight > 0
    if torch.any(valid):
        center_err = F.huber_loss(
            mu[valid],
            center_target[valid],
            delta=float(config.center_huber_delta),
            reduction="none",
        )
        center_loss = torch.mean(center_err * center_weight[valid])
    else:
        center_loss = logits.new_tensor(0.0)

    if mu.shape[1] >= 3:
        smooth_loss = torch.mean(torch.abs(mu[:, 2:] - 2.0 * mu[:, 1:-1] + mu[:, :-2]))
    else:
        smooth_loss = logits.new_tensor(0.0)

    tv_t = torch.mean(torch.abs(prob[:, 1:, :] - prob[:, :-1, :])) if prob.shape[1] > 1 else logits.new_tensor(0.0)
    tv_c = torch.mean(torch.abs(prob[:, :, 1:] - prob[:, :, :-1])) if prob.shape[2] > 1 else logits.new_tensor(0.0)
    tv_loss = 0.5 * (tv_t + tv_c)
    area_loss = torch.mean(prob)

    total = (
        config.loss_mask_weight * mask_loss
        + config.loss_center_weight * center_loss
        + config.loss_smooth_weight * smooth_loss
        + config.loss_tv_weight * tv_loss
        + config.loss_area_weight * area_loss
    )
    return {
        "total": total,
        "mask": mask_loss,
        "center": center_loss,
        "smooth": smooth_loss,
        "tv": tv_loss,
        "area": area_loss,
    }

"""Classification and distillation losses. All take optional per-class weights."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def cross_entropy(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, y, weight=weights)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None,
               gamma: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
    loss = -((1 - log_p.exp()) ** gamma) * log_p
    if weights is not None:
        w = weights[y]
        return (loss * w).sum() / w.sum()
    return loss.mean()


@torch.no_grad()
def teacher_probs(teacher: nn.Module, x: torch.Tensor, temperature: float) -> torch.Tensor:
    if hasattr(teacher, "soft_targets"):
        return teacher.soft_targets(x, temperature)
    return torch.softmax(teacher(x) / temperature, dim=-1)


def kd_loss(student_logits: torch.Tensor, target_probs: torch.Tensor, temperature: float,
            sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """T^2 * KL(target || student) at temperature T, optionally weighted per sample."""
    log_q = F.log_softmax(student_logits / temperature, dim=-1)
    kl = (target_probs * (target_probs.clamp_min(1e-12).log() - log_q)).sum(dim=-1)
    if sample_weights is not None:
        kl = (kl * sample_weights).sum() / sample_weights.sum()
    else:
        kl = kl.mean()
    return kl * temperature ** 2


class DistillationLoss:
    """alpha * CE(y) + (1 - alpha) * KD(targets). With `minority_kd`, the KD term is weighted by class weight."""

    def __init__(self, alpha: float, temperature: float, class_weights: torch.Tensor | None,
                 minority_kd: bool = True):
        self.alpha = alpha
        self.temperature = temperature
        self.class_weights = class_weights
        self.minority_kd = minority_kd

    def __call__(self, logits: torch.Tensor, y: torch.Tensor, target_probs: torch.Tensor | None) -> torch.Tensor:
        ce = cross_entropy(logits, y, self.class_weights)
        if target_probs is None:
            return ce
        sample_w = self.class_weights[y] if (self.minority_kd and self.class_weights is not None) else None
        return self.alpha * ce + (1 - self.alpha) * kd_loss(logits, target_probs, self.temperature, sample_w)

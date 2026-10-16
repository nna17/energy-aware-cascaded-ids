"""Train the elastic supernet once with the sandwich rule (largest + smallest + random sub-models per step)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eids.data.datamodule import DataModule
from eids.distill.losses import DistillationLoss, teacher_probs
from eids.eval.evaluator import evaluate_subnet
from eids.models.elastic import ElasticMLP, SearchSpace
from eids.training.classifier import make_class_weights
from eids.training.trainer import fit
from eids.utils.config import Config


def train_supernet(supernet: ElasticMLP, space: SearchSpace, dm: DataModule, cfg: Config,
                   device: torch.device, seed: int, teacher: nn.Module | None = None,
                   name: str = "supernet") -> list[dict]:
    """Soft targets for every sub-model:
    - teacher only, or the largest sub-model only (in-place distillation), or the average of both;
    - none when `use_kd` is false and in-place distillation is off (baseline B4).
    """
    supernet.to(device)
    use_teacher = cfg.get("use_kd", True) and teacher is not None
    if use_teacher:
        teacher.to(device).eval()
    inplace = cfg.get("inplace_distillation", True)
    T = cfg.temperature
    loss_fn = DistillationLoss(cfg.alpha, T, make_class_weights(dm, cfg, device),
                               cfg.get("minority_kd_weighting", True))
    rng = np.random.default_rng(seed)
    largest, smallest = space.largest(), space.smallest()

    def step(xb, yb):
        t_probs = teacher_probs(teacher, xb, T) if use_teacher else None

        supernet.set_active(largest)
        logits = supernet(xb)
        total = loss_fn(logits, yb, t_probs)

        targets = t_probs
        if inplace:
            own = torch.softmax(logits.detach() / T, dim=-1)
            targets = own if t_probs is None else 0.5 * (t_probs + own)

        for sub in [smallest] + [space.sample(rng) for _ in range(cfg.get("n_random", 2))]:
            supernet.set_active(sub)
            total = total + loss_fn(supernet(xb), yb, targets)
        supernet.set_active(largest)
        return total

    def validate() -> float:
        scores = [evaluate_subnet(supernet, sub, dm, "val", device, recalib_batches=10)["macro_f1"]
                  for sub in (largest, smallest)]
        supernet.set_active(largest)
        return float(np.mean(scores))

    gen = torch.Generator().manual_seed(seed)
    return fit(
        supernet, step,
        train_batches=lambda: dm.batches("train", cfg.batch_size, device, shuffle=True, generator=gen),
        validate=validate, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.get("weight_decay", 0.0),
        patience=cfg.get("patience"), name=name,
    )

"""Train one fixed model: a teacher, the 1D-CNN baseline, or a fixed KD student (B2)."""
from __future__ import annotations

import torch
from torch import nn

from eids.data.datamodule import DataModule
from eids.distill.class_weights import class_weights
from eids.distill.losses import DistillationLoss, cross_entropy, focal_loss, teacher_probs
from eids.eval.evaluator import evaluate
from eids.training.trainer import fit
from eids.utils.config import Config


def make_class_weights(dm: DataModule, cfg: Config, device: torch.device) -> torch.Tensor:
    return class_weights(dm.class_counts(), cfg.get("class_weighting", "none"),
                         cfg.get("class_weight_beta", 0.999)).to(device)


def train_classifier(model: nn.Module, dm: DataModule, cfg: Config, device: torch.device, seed: int,
                     teacher: nn.Module | None = None, kd_cfg: Config | None = None,
                     name: str = "classifier") -> list[dict]:
    """Without a teacher: focal or CE loss. With a teacher: DistillationLoss using `kd_cfg` settings."""
    model.to(device)
    weights = make_class_weights(dm, cfg, device)

    if teacher is not None:
        teacher.to(device).eval()
        kd = kd_cfg or cfg
        loss_fn = DistillationLoss(kd.alpha, kd.temperature, weights, kd.get("minority_kd_weighting", True))

        def step(xb, yb):
            return loss_fn(model(xb), yb, teacher_probs(teacher, xb, kd.temperature))
    elif cfg.get("loss", "ce") == "focal":
        def step(xb, yb):
            return focal_loss(model(xb), yb, weights, cfg.get("focal_gamma", 2.0))
    else:
        def step(xb, yb):
            return cross_entropy(model(xb), yb, weights)

    gen = torch.Generator().manual_seed(seed)
    return fit(
        model, step,
        train_batches=lambda: dm.batches("train", cfg.batch_size, device, shuffle=True, generator=gen),
        validate=lambda: evaluate(model, dm, "val", device)["macro_f1"],
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.get("weight_decay", 0.0),
        patience=cfg.get("patience"), name=name,
    )

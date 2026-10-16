"""Generic training loop: the caller supplies the loss step and the validation score."""
from __future__ import annotations

import copy
import time
from typing import Callable, Iterator

import torch
from torch import nn

from eids.utils.logging import get_logger

log = get_logger()

StepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BatchesFn = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]


def fit(model: nn.Module, step: StepFn, train_batches: BatchesFn, validate: Callable[[], float],
        epochs: int, lr: float, weight_decay: float = 0.0, patience: int | None = None,
        name: str = "model") -> list[dict]:
    """Train with AdamW + cosine schedule; keep the weights with the best validation score (higher is better)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_score, best_state, bad_epochs, history = float("-inf"), None, 0, []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_batches():
            loss = step(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        sched.step()

        score = validate()
        history.append({"epoch": epoch, "loss": total / max(n, 1), "val_score": score,
                        "seconds": time.perf_counter() - start})
        log.info(f"[{name}] epoch {epoch:3d} loss {total / max(n, 1):.4f} val {score:.4f}")

        if score > best_score:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                log.info(f"[{name}] early stop at epoch {epoch}; best val {best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history

"""Teacher ensemble: averages member probabilities, which gives smoother soft labels for rare classes."""
from __future__ import annotations

import torch
from torch import nn


class TeacherEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def soft_targets(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        probs = [torch.softmax(m(x) / temperature, dim=-1) for m in self.members]
        return torch.stack(probs).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log of the averaged probabilities: valid logits for argmax and for cross-entropy.
        return torch.log(self.soft_targets(x).clamp_min(1e-12))

"""CBF-PDE auxiliary critic network for control barrier function residual method."""

from __future__ import annotations
import torch
import torch.nn as nn
from models.pde.state_builder import XI_DIM


class CBFAuxCritic(nn.Module):
    """MLP auxiliary critic for CBF-PDE safety-regularized method.

    Same architecture as HJBAuxCritic. The difference is in the residual
    computation (ReLU(-max_a [h_dot + alpha*h])), handled in the agent class.
    """

    def __init__(self, in_dim: int = XI_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute U(xi). Returns (batch,) scalar values."""
        return self.net(xi).squeeze(-1)

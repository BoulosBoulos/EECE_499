"""HJB auxiliary critic network for hard-HJB residual method."""

from __future__ import annotations
import torch
import torch.nn as nn
from models.pde.state_builder import XI_DIM


class HJBAuxCritic(nn.Module):
    """MLP auxiliary critic on reduced PDE state xi.
    
    Separate from the PPO recurrent critic. Trained with HJB residual loss
    and used to distill physics-informed values into the PPO value head.
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

"""Soft-HJB auxiliary critic network for entropy-regularized HJB residual method."""

from __future__ import annotations
import torch
import torch.nn as nn
from models.pde.state_builder import XI_DIM


class SoftHJBAuxCritic(nn.Module):
    """MLP auxiliary critic for soft-HJB method.
    
    Same architecture as HJBAuxCritic. The difference is in the residual
    computation (logsumexp instead of hard max) and the actor-alignment
    KL term, both handled in the agent class.
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

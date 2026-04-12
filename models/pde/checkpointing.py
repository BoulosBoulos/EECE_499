"""Checkpoint save/load for PDE-family models.

Each checkpoint stores policy, auxiliary critic, optimizers, and metadata
so the exact configuration can be reconstructed for evaluation.
"""

from __future__ import annotations
import torch
import os
from models.pde.state_builder import XI_DIM


def save_pde_checkpoint(
    path: str,
    policy_state: dict,
    policy_optim_state: dict,
    aux_state: dict,
    aux_optim_state: dict,
    obs_dim: int,
    method: str,
    config: dict,
    extra: dict | None = None,
):
    """Save a PDE-family checkpoint."""
    data = {
        "family": "pde",
        "method": method,
        "xi_dim": XI_DIM,
        "obs_dim": obs_dim,
        "policy": policy_state,
        "policy_optimizer": policy_optim_state,
        "aux_critic": aux_state,
        "aux_optimizer": aux_optim_state,
        "config": config,
    }
    if extra:
        data.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)


def load_pde_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load and validate a PDE-family checkpoint."""
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("family") != "pde":
        raise ValueError(f"Not a PDE-family checkpoint: family={data.get('family')}")
    if data.get("xi_dim") != XI_DIM:
        raise ValueError(f"XI_DIM mismatch: checkpoint has {data.get('xi_dim')}, expected {XI_DIM}")
    return data

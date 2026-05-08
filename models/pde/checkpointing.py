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
    arch: dict | None = None,
    extra: dict | None = None,
):
    """Save a PDE-family checkpoint.

    Phase 3 followup: ``arch`` dict captures every constructor argument that
    determines the model's tensor shapes (obs_dim, n_actions, hidden_dim,
    aux_hidden_dim, …). This lets ``Agent.from_checkpoint(path)`` reconstruct
    the agent without the caller having to know the architecture.
    """
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
    if arch is not None:
        data["arch"] = dict(arch)
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


def verify_arch(checkpoint_arch: dict | None, current: dict, strict: bool = True,
                ckpt_path: str | None = None) -> None:
    """Phase 3 followup: cross-check checkpoint's saved arch dict against the
    current agent's constructor params. Raises on mismatch when ``strict``;
    warns on missing arch (old checkpoint) regardless.
    """
    where = f" ({ckpt_path})" if ckpt_path else ""
    if not checkpoint_arch:
        # Old checkpoint, written before arch was persisted.
        print(f"[load] WARNING: checkpoint{where} has no 'arch' dict; "
              f"skipping architecture verification (legacy ckpt).")
        return
    mismatches = []
    for k, expected in checkpoint_arch.items():
        actual = current.get(k)
        if actual is None:
            continue  # Field not currently tracked; skip silently.
        if int(expected) != int(actual):
            mismatches.append(f"{k}: ckpt={expected}, current={actual}")
    if mismatches:
        msg = (f"Architecture mismatch loading checkpoint{where}: "
               + "; ".join(mismatches)
               + ". Use Agent.from_checkpoint(path, device=...) to reconstruct "
                 "with the saved architecture.")
        if strict:
            raise ValueError(msg)
        print(f"[load] WARNING: {msg}")


def peek_checkpoint_arch(path: str, device: str = "cpu") -> dict:
    """Read just the `arch` dict from a checkpoint without instantiating
    a model. Used by ``Agent.from_checkpoint`` to learn the architecture
    before constructing the agent.
    Returns {} if the checkpoint predates the `arch` field.
    """
    data = torch.load(path, map_location=device, weights_only=False)
    return dict(data.get("arch") or {})

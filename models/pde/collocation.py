"""Collocation sampler for PDE residual training.

Generates training points by mixing real rollout states with jittered copies.
Only primitive features are jittered; derived quantities (tau, t_cpa, d_cpa, TTC)
are recomputed after jitter to maintain physical consistency.
"""

from __future__ import annotations
import torch
import numpy as np
from models.pde.state_builder import (
    XI_DIM, N_AGENT_FEAT, N_AGENTS_PDE,
    IDX_V, IDX_A, IDX_PSI_DOT, IDX_D_STOP, IDX_D_CZ, IDX_D_EXIT,
    IDX_KAPPA, IDX_TTC_MIN, IDX_POTHOLE,
)

PRIMITIVE_JITTER_STD = {
    IDX_V: 0.3,
    IDX_A: 0.2,
    IDX_PSI_DOT: 0.1,
    IDX_D_STOP: 1.0,
    IDX_D_CZ: 1.0,
    IDX_D_EXIT: 1.0,
    IDX_KAPPA: 0.02,
    8: 0.05,   # alpha_cz
    9: 0.05,   # alpha_cross
    10: 2.0,   # d_occ
    11: 0.2,   # dt_seen
    IDX_POTHOLE: 1.0,
}

AGENT_PRIMITIVE_OFFSETS = [0, 1, 2, 3, 5, 7, 8]
AGENT_PRIMITIVE_STDS = [0.5, 0.5, 0.2, 0.2, 0.3, 1.0, 1.0]

CLAMP_RANGES = {
    IDX_V: (0.0, 13.89),
    IDX_D_STOP: (0.0, 100.0),
    IDX_D_CZ: (0.0, 100.0),
    IDX_D_EXIT: (0.0, 100.0),
    8: (0.0, 1.0),
    9: (0.0, 1.0),
    10: (0.0, 200.0),
    11: (0.0, 30.0),
    IDX_POTHOLE: (0.0, 200.0),
}


def _recompute_agent_derived(xi: torch.Tensor, eps: float = 1e-6, d_safe: float = 2.0, t_h: float = 3.0):
    """Recompute tau, delta_tau, t_cpa, d_cpa, TTC from primitive agent features."""
    v_ego = xi[:, IDX_V]
    d_cz_ego = xi[:, IDX_D_CZ]
    tau_ego = d_cz_ego / (v_ego + eps)

    ttc_min = torch.full((xi.shape[0],), 10.0, device=xi.device, dtype=xi.dtype)

    for ag_idx in range(N_AGENTS_PDE):
        start = 12 + ag_idx * N_AGENT_FEAT
        mask = xi[:, start + 21]
        dx = xi[:, start + 0]
        dy = xi[:, start + 1]
        dvx = xi[:, start + 2]
        dvy = xi[:, start + 3]
        v_i = xi[:, start + 5]
        d_cz_i = xi[:, start + 7]

        tau_i = d_cz_i / (v_i + eps)
        xi[:, start + 9] = tau_i
        xi[:, start + 10] = tau_i - tau_ego

        dv_sq = dvx ** 2 + dvy ** 2 + eps
        t_cpa = torch.clamp(-(dx * dvx + dy * dvy) / dv_sq, 0.0, t_h)
        px_cpa = dx + t_cpa * dvx
        py_cpa = dy + t_cpa * dvy
        d_cpa = torch.sqrt(px_cpa ** 2 + py_cpa ** 2 + eps)
        dv_norm = torch.sqrt(dv_sq)
        ttc_i = torch.clamp(d_cpa - d_safe, min=0.0) / dv_norm

        xi[:, start + 11] = t_cpa
        xi[:, start + 12] = d_cpa
        xi[:, start + 13] = ttc_i

        ttc_min = torch.where((mask > 0.5) & (ttc_i < ttc_min), ttc_i, ttc_min)

    xi[:, IDX_TTC_MIN] = ttc_min
    return xi


def sample_collocation(
    xi_rollout: torch.Tensor,
    ratio_real: float = 0.7,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample collocation points: mix of real rollout states and jittered copies.

    Args:
        xi_rollout: (N, XI_DIM) rollout PDE states
        ratio_real: fraction of real states in the output
        seed: optional random seed
    Returns:
        xi_colloc: (N, XI_DIM) collocation states
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = xi_rollout.shape[0]
    n_real = int(N * ratio_real)
    n_jitter = N - n_real

    perm = torch.randperm(N, device=xi_rollout.device)
    real_idx = perm[:n_real]
    jitter_base_idx = perm[n_real:]
    if n_jitter > 0 and len(jitter_base_idx) < n_jitter:
        extra = torch.randint(0, N, (n_jitter - len(jitter_base_idx),), device=xi_rollout.device)
        jitter_base_idx = torch.cat([jitter_base_idx, extra])

    xi_real = xi_rollout[real_idx]

    if n_jitter > 0:
        xi_jitter = xi_rollout[jitter_base_idx[:n_jitter]].clone()

        for dim_idx, std in PRIMITIVE_JITTER_STD.items():
            noise = torch.randn(n_jitter, device=xi_jitter.device, dtype=xi_jitter.dtype) * std
            xi_jitter[:, dim_idx] = xi_jitter[:, dim_idx] + noise
            if dim_idx in CLAMP_RANGES:
                lo, hi = CLAMP_RANGES[dim_idx]
                xi_jitter[:, dim_idx] = xi_jitter[:, dim_idx].clamp(lo, hi)

        for ag_idx in range(N_AGENTS_PDE):
            start = 12 + ag_idx * N_AGENT_FEAT
            for offset, std in zip(AGENT_PRIMITIVE_OFFSETS, AGENT_PRIMITIVE_STDS):
                noise = torch.randn(n_jitter, device=xi_jitter.device, dtype=xi_jitter.dtype) * std
                xi_jitter[:, start + offset] = xi_jitter[:, start + offset] + noise
            xi_jitter[:, start + 5] = xi_jitter[:, start + 5].clamp(min=0.0)
            xi_jitter[:, start + 7] = xi_jitter[:, start + 7].clamp(min=0.0)
            xi_jitter[:, start + 8] = xi_jitter[:, start + 8].clamp(min=0.0)

        xi_jitter = _recompute_agent_derived(xi_jitter)
        xi_colloc = torch.cat([xi_real, xi_jitter], dim=0)
    else:
        xi_colloc = xi_real

    shuffle = torch.randperm(xi_colloc.shape[0], device=xi_colloc.device)
    return xi_colloc[shuffle]

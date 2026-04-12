"""Map interaction benchmark observation vector to reduced PDE state xi.

Constants XI_DIM, N_AGENT_FEAT, N_AGENTS_PDE must stay aligned with
``models.pde.state_builder`` (no torch import on the training entry path).
"""

from __future__ import annotations

import numpy as np

XI_DIM = 79
N_AGENT_FEAT = 22
N_AGENTS_PDE = 3

_EGO = 8
_PHASE = 5
_DOM = 12
_PER_ACTOR = 21
_TOP_N = 3
_S_OBS = 3
_BASE = _EGO + _PHASE + _DOM + _TOP_N * _PER_ACTOR + _S_OBS  # 91


def xi_from_interaction_obs(obs: np.ndarray) -> np.ndarray:
    """Build xi ∈ R^{XI_DIM} from interaction observation (91 or 92 dims)."""
    o = np.asarray(obs, dtype=np.float64).ravel()
    xi = np.zeros(XI_DIM, dtype=np.float32)

    if o.size < _BASE:
        return xi

    xi[0] = float(o[0])
    xi[1] = float(o[1])
    xi[2] = float(o[3])
    xi[3] = float(o[4])
    xi[4] = float(o[5])
    xi[5] = float(o[6])
    xi[6] = 0.0
    ego_eta = float(o[7])
    dom_eta = float(o[17]) if o.size > 17 else ego_eta
    xi[7] = float(np.clip(min(ego_eta, dom_eta), 0.1, 10.0))

    xi[8] = float(o[23]) if o.size > 23 else 0.0
    xi[9] = float(o[9]) if o.size > 9 else 0.0
    xi[10] = float(np.clip(o[5] * 0.02, 0.0, 1.0))
    xi[11] = float(o[90]) if o.size > 90 else float(o[89]) if o.size > 89 else 0.0

    a0 = 25
    for j in range(N_AGENTS_PDE):
        src_start = a0 + j * _PER_ACTOR
        dst_start = 12 + j * N_AGENT_FEAT
        if src_start + _PER_ACTOR <= o.size:
            chunk = o[src_start : src_start + _PER_ACTOR].astype(np.float32)
            n_copy = min(N_AGENT_FEAT, _PER_ACTOR)
            xi[dst_start : dst_start + n_copy] = chunk[:n_copy]
            if N_AGENT_FEAT > _PER_ACTOR:
                xi[dst_start + _PER_ACTOR : dst_start + N_AGENT_FEAT] = 0.0

    if o.size > _BASE:
        xi[78] = float(o[_BASE])
    else:
        xi[78] = 100.0

    return xi

"""Conflict-centric state builder for the interaction benchmark.

Constructs s_t = [s_ego_kin, s_ego_phase, s_dominant, f_1, ..., f_N, s_obs]

Key differences from the legacy builder:
  * ETA-based features instead of generic d_cz
  * Explicit legal / effective priority
  * Actor commitment / yielding flags
  * Dominant-actor summary vector
  * Phase encoding for the ego episode state
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional

from scenario.conflict_map import (
    CONFLICT_ZONES, CZ_EXIT_CROSSWALK, PRIO_ACTOR, PRIO_EGO,
)


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _rot2d(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


# ── Phase encoding ──────────────────────────────────────────────────────────

PHASE_NAMES = ["approach", "decision", "committed", "clearing", "aborted"]
PHASE_DIM = len(PHASE_NAMES)


def phase_onehot(phase: str) -> np.ndarray:
    idx = PHASE_NAMES.index(phase) if phase in PHASE_NAMES else 0
    v = np.zeros(PHASE_DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


# ── Actor type encoding ────────────────────────────────────────────────────

ACTOR_TYPES = ["veh", "ped", "cyc"]
ACTOR_TYPE_DIM = 3


def actor_type_onehot(t: str) -> np.ndarray:
    idx = ACTOR_TYPES.index(t) if t in ACTOR_TYPES else 0
    v = np.zeros(ACTOR_TYPE_DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


class InteractionStateBuilder:
    """Build the conflict-centric state vector.

    State layout (91 dims with top_n=3):
      s_ego_kin       (8)   v, a, jerk, yaw_rate, d_preentry, d_conflict_entry,
                            d_conflict_exit, ego_eta_enter
      s_ego_phase     (5)   one-hot approach/decision/committed/clearing/aborted
      s_dominant     (12)   dom_type(3), ego_eta_enter, dom_eta_enter, delta_eta,
                            ego_eta_exit, dom_eta_exit, legal_prio, eff_prio,
                            conflict_occupied, dom_committed
      f_1..f_N  (N*21=63)  per-actor: rel_xy(2), rel_v(2), v_i, a_i, eta_enter,
                            eta_exit, delta_eta, legal_prio, committed, yielding,
                            crosswalk_progress, relevant, type(3), uncertainty,
                            distance, TTC, mask
      s_obs           (3)   n_visible, min_seen_dur, max_uncertainty
    """

    EGO_KIN_DIM = 8
    DOMINANT_DIM = 12
    PER_ACTOR_DIM = 21
    OBS_DIM = 3

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.state_dim = (
            self.EGO_KIN_DIM + PHASE_DIM + self.DOMINANT_DIM
            + self.top_n * self.PER_ACTOR_DIM + self.OBS_DIM
        )

    def build(self, obs: Dict[str, Any]) -> np.ndarray:
        """Build state vector from raw observation dict.

        Expected keys in *obs*:
          ego: {v, a, jerk, yaw_rate, d_preentry, d_conflict_entry,
                d_conflict_exit, ego_eta_enter, psi, p}
          phase: str
          actors: [{id, p, psi, v, a, actor_type, eta_enter, eta_exit,
                    legal_priority, committed, yielding, crosswalk_progress,
                    relevant, uncertainty, first_seen_step}]
          step: int
          dt: float
        """
        ego = obs.get("ego", {})
        phase = obs.get("phase", "approach")
        actors = obs.get("actors", [])
        step = obs.get("step", 0)
        dt = obs.get("dt", 0.5)

        # ── Ego kinematics ──────────────────────────────────────────────────
        s_ego = np.array([
            ego.get("v", 0.0),
            ego.get("a", 0.0),
            ego.get("jerk", 0.0),
            ego.get("yaw_rate", 0.0),
            ego.get("d_preentry", 30.0),
            ego.get("d_conflict_entry", 40.0),
            ego.get("d_conflict_exit", 50.0),
            ego.get("ego_eta_enter", 5.0),
        ], dtype=np.float32)

        # ── Ego phase ───────────────────────────────────────────────────────
        s_phase = phase_onehot(phase)

        # ── Per-actor features ──────────────────────────────────────────────
        psi_e = float(ego.get("psi", 0.0))
        p_e = np.array(ego.get("p", [0, 0]), dtype=float)
        v_e = float(ego.get("v", 0.0))
        R = _rot2d(-psi_e)

        scored = []
        for ag in actors:
            eta_e = float(ego.get("ego_eta_enter", 5.0))
            eta_a = float(ag.get("eta_enter", 10.0))
            delta = abs(eta_a - eta_e)
            dist = float(np.linalg.norm(np.array(ag.get("p", [0, 0])) - p_e)) + 1e-6
            scored.append((delta, dist, ag))
        scored.sort(key=lambda x: (x[0], x[1]))
        top_actors = [s[2] for s in scored[:self.top_n]]

        actor_feats = []
        for ag in top_actors:
            actor_feats.append(self._actor_features(ag, p_e, psi_e, R, v_e, ego, step, dt))
        while len(actor_feats) < self.top_n:
            actor_feats.append(np.zeros(self.PER_ACTOR_DIM, dtype=np.float32))
        f_actors = np.concatenate(actor_feats[:self.top_n])

        # ── Dominant conflict summary ───────────────────────────────────────
        s_dom = self._dominant_summary(ego, top_actors)

        # ── Observability ───────────────────────────────────────────────────
        n_vis = len(actors)
        min_seen = min(
            ((step - ag.get("first_seen_step", step)) * dt for ag in actors),
            default=0.0,
        )
        max_unc = max((ag.get("uncertainty", 0.0) for ag in actors), default=0.0)
        s_obs = np.array([n_vis, min_seen, max_unc], dtype=np.float32)

        return np.concatenate([s_ego, s_phase, s_dom, f_actors, s_obs]).astype(np.float32)

    def _actor_features(
        self, ag: dict, p_e: np.ndarray, psi_e: float,
        R: np.ndarray, v_e: float, ego: dict, step: int, dt: float,
    ) -> np.ndarray:
        p_i = np.array(ag.get("p", [0, 0]), dtype=float)
        dp = p_i - p_e
        rel_xy = R @ dp
        psi_i = float(ag.get("psi", 0.0))
        v_i = float(ag.get("v", 0.0))
        a_i = float(ag.get("a", 0.0))
        v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])
        v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])
        rel_v = R @ (v_i_vec - v_e_vec)

        eta_enter = float(ag.get("eta_enter", 10.0))
        eta_exit = float(ag.get("eta_exit", 12.0))
        ego_eta = float(ego.get("ego_eta_enter", 5.0))
        delta_eta = eta_enter - ego_eta

        dist = float(np.linalg.norm(dp)) + 1e-6
        dv = v_i_vec - v_e_vec
        dv_norm = float(np.linalg.norm(dv)) + 1e-6
        t_cpa = float(np.clip(-np.dot(dp, dv) / (dv_norm ** 2 + 1e-6), 0, 5))
        d_cpa = float(np.linalg.norm(dp + t_cpa * dv))
        ttc = max(d_cpa - 2.0, 0.0) / dv_norm

        return np.array([
            rel_xy[0], rel_xy[1],
            rel_v[0], rel_v[1],
            v_i, a_i,
            eta_enter, eta_exit, delta_eta,
            float(ag.get("legal_priority", 0.5)),
            float(ag.get("committed", 0)),
            float(ag.get("yielding", 0)),
            float(ag.get("crosswalk_progress", 0)),
            float(ag.get("relevant", 1)),
            *actor_type_onehot(ag.get("actor_type", "veh")),
            float(ag.get("uncertainty", 0.0)),
            min(dist / 50.0, 2.0),
            min(ttc, 10.0),
            1.0,  # mask = slot occupied
        ], dtype=np.float32)

    def _dominant_summary(self, ego: dict, top_actors: list) -> np.ndarray:
        """Build the 12-dim dominant-conflict summary vector."""
        ego_eta_enter = float(ego.get("ego_eta_enter", 5.0))
        ego_eta_exit = float(ego.get("ego_eta_exit", 7.0))

        if not top_actors:
            return np.zeros(self.DOMINANT_DIM, dtype=np.float32)

        dom = top_actors[0]
        dom_type = actor_type_onehot(dom.get("actor_type", "veh"))
        dom_eta_enter = float(dom.get("eta_enter", 10.0))
        dom_eta_exit = float(dom.get("eta_exit", 12.0))
        delta_eta = dom_eta_enter - ego_eta_enter
        legal_prio = float(dom.get("legal_priority", 0.5))

        dom_committed = float(dom.get("committed", 0))
        dom_yielding = float(dom.get("yielding", 0))
        eff_prio = legal_prio
        if dom_committed:
            eff_prio = max(eff_prio, 0.8)
        if dom_yielding:
            eff_prio = min(eff_prio, 0.2)
        conflict_occupied = float(dom.get("in_conflict_zone", 0))

        return np.array([
            *dom_type,
            ego_eta_enter, dom_eta_enter, delta_eta,
            ego_eta_exit, dom_eta_exit,
            legal_prio, eff_prio,
            conflict_occupied, dom_committed,
        ], dtype=np.float32)

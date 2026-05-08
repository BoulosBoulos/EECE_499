"""Rule-based TTC-threshold baseline policy.

Behavioral rule:
    - If any agent's TTC is below ttc_threshold, STOP (action=0).
    - If still far from conflict zone (d_cz > far_zone_dist), GO cautiously toward junction.
    - Otherwise, GO (action=3).

This is the standard heuristic baseline for unsignalized intersection papers.
No learning, no checkpoint — fully deterministic.
"""

import numpy as np


class RuleBasedTTCPolicy:
    """TTC-threshold rule-based policy for T-intersection."""

    def __init__(self, obs_dim=135, ttc_threshold=3.0, far_zone_dist=5.0, device="cpu"):
        self.obs_dim = obs_dim
        self.ttc_threshold = ttc_threshold
        self.far_zone_dist = far_zone_dist
        self.device = device

    def reset_hidden(self):
        """No hidden state; no-op for API compatibility."""
        pass

    def get_action(self, obs, deterministic=True):
        """Return (action, log_prob, value, hidden).

        Obs layout: ego(6) + geom(12) + vis(6) + 5*agent(22) + pothole(1) = 135
        d_cz at geom index 1 -> obs[7]
        Per-agent TTC at offset 13 within each 22-feature block (t_cpa, d_cpa, TTC are at 11,12,13)
        """
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()

        if len(obs_arr) < 24:
            return 0, 0.0, 0.0, None

        d_cz = float(obs_arr[7]) if len(obs_arr) > 7 else 100.0

        # Extract min TTC across top-5 tracked agents
        # Agent block: starts at index 24, each 22 features
        # TTC_i is at offset 13 within each block (see state/builder.py)
        # Mask is at offset 21 within each block; only consider active agents
        min_ttc = 10.0
        for i in range(5):
            base_idx = 24 + i * 22
            mask_idx = base_idx + 21
            ttc_idx = base_idx + 13
            if mask_idx < len(obs_arr) and float(obs_arr[mask_idx]) >= 0.5:
                if ttc_idx < len(obs_arr):
                    agent_ttc = float(obs_arr[ttc_idx])
                    if 0.01 < agent_ttc < 10.0:
                        min_ttc = min(min_ttc, agent_ttc)

        # Rule: STOP if unsafe, GO otherwise
        if min_ttc < self.ttc_threshold:
            action = 0  # STOP
        elif d_cz > self.far_zone_dist:
            action = 3  # GO (approach junction)
        else:
            action = 3  # GO (clear junction)

        return action, 0.0, 0.0, None

    def save(self, path):
        """No-op; rule-based has no parameters."""
        import json
        meta_path = path if path.endswith(".json") else path + ".rule_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"policy_type": "rule_based",
                        "ttc_threshold": self.ttc_threshold,
                        "far_zone_dist": self.far_zone_dist}, f)

    def load(self, path):
        """No-op for compatibility."""
        pass

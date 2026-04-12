"""Tests for the 88-dimensional conflict-centric state builder."""

import numpy as np
import unittest

from state.builder_interaction import (
    InteractionStateBuilder, phase_onehot, actor_type_onehot,
    PHASE_NAMES, ACTOR_TYPES,
)


class TestPhaseEncoding(unittest.TestCase):

    def test_all_phases_produce_onehot(self):
        for phase in PHASE_NAMES:
            v = phase_onehot(phase)
            self.assertEqual(v.shape, (5,))
            self.assertAlmostEqual(v.sum(), 1.0)
            self.assertEqual(v.argmax(), PHASE_NAMES.index(phase))

    def test_unknown_phase_defaults(self):
        v = phase_onehot("nonexistent")
        self.assertEqual(v.argmax(), 0)


class TestActorTypeEncoding(unittest.TestCase):

    def test_all_types_produce_onehot(self):
        for t in ACTOR_TYPES:
            v = actor_type_onehot(t)
            self.assertEqual(v.shape, (3,))
            self.assertAlmostEqual(v.sum(), 1.0)


class TestStateBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = InteractionStateBuilder(top_n=3)

    def test_state_dim_is_91(self):
        self.assertEqual(self.builder.state_dim, 91)

    def test_empty_obs_produces_correct_shape(self):
        obs = {
            "ego": {"v": 5.0, "a": 0.1, "jerk": 0.0, "yaw_rate": 0.0,
                    "d_preentry": 10.0, "d_conflict_entry": 15.0,
                    "d_conflict_exit": 25.0, "ego_eta_enter": 3.0,
                    "psi": 0.0, "p": [50.0, 10.0]},
            "phase": "decision",
            "actors": [],
            "step": 5,
            "dt": 0.5,
        }
        state = self.builder.build(obs)
        self.assertEqual(state.shape, (91,))
        self.assertEqual(state.dtype, np.float32)

    def test_with_actors_correct_shape(self):
        actors = [
            {"id": "a1", "p": [40, 0], "psi": 0.1, "v": 10, "a": -0.5,
             "actor_type": "veh", "eta_enter": 3.0, "eta_exit": 5.0,
             "legal_priority": 0.0, "committed": 1, "yielding": 0,
             "crosswalk_progress": 0.0, "relevant": 1, "uncertainty": 0.2,
             "first_seen_step": 0, "in_conflict_zone": 0},
            {"id": "p0", "p": [45, 5], "psi": 1.5, "v": 1.2, "a": 0,
             "actor_type": "ped", "eta_enter": 4.0, "eta_exit": 8.0,
             "legal_priority": 0.0, "committed": 0, "yielding": 1,
             "crosswalk_progress": 0.3, "relevant": 1, "uncertainty": 0.5,
             "first_seen_step": 2, "in_conflict_zone": 0},
        ]
        obs = {
            "ego": {"v": 8.0, "a": 0.5, "jerk": 0.1, "yaw_rate": 0.02,
                    "d_preentry": 8.0, "d_conflict_entry": 12.0,
                    "d_conflict_exit": 22.0, "ego_eta_enter": 1.5,
                    "ego_eta_exit": 2.75, "psi": 0.1, "p": [50.0, 5.0]},
            "phase": "committed",
            "actors": actors,
            "step": 10,
            "dt": 0.5,
        }
        state = self.builder.build(obs)
        self.assertEqual(state.shape, (91,))

    def test_dominant_summary_uses_first_actor(self):
        actors = [
            {"id": "close", "p": [48, 0], "psi": 0, "v": 10, "a": 0,
             "actor_type": "veh", "eta_enter": 1.0, "eta_exit": 3.0,
             "legal_priority": 0.0, "committed": 1, "yielding": 0,
             "crosswalk_progress": 0, "relevant": 1, "uncertainty": 0.1,
             "first_seen_step": 0, "in_conflict_zone": 1},
            {"id": "far", "p": [20, 0], "psi": 0, "v": 5, "a": 0,
             "actor_type": "cyc", "eta_enter": 6.0, "eta_exit": 8.0,
             "legal_priority": 0.5, "committed": 0, "yielding": 0,
             "crosswalk_progress": 0, "relevant": 1, "uncertainty": 0.5,
             "first_seen_step": 1, "in_conflict_zone": 0},
        ]
        obs = {
            "ego": {"v": 8, "a": 0, "jerk": 0, "yaw_rate": 0,
                    "d_preentry": 5, "d_conflict_entry": 10,
                    "d_conflict_exit": 20, "ego_eta_enter": 1.25,
                    "ego_eta_exit": 2.5, "psi": 0, "p": [50, 0]},
            "phase": "decision",
            "actors": actors,
            "step": 3,
            "dt": 0.5,
        }
        state = self.builder.build(obs)
        # Dominant summary starts at index 13 (8 ego_kin + 5 phase)
        dom_block = state[13:25]
        self.assertAlmostEqual(dom_block[-1], 1.0, places=1)

    def test_mask_zero_for_empty_slots(self):
        obs = {
            "ego": {"v": 5, "a": 0, "jerk": 0, "yaw_rate": 0,
                    "d_preentry": 10, "d_conflict_entry": 15,
                    "d_conflict_exit": 25, "ego_eta_enter": 3,
                    "psi": 0, "p": [50, 0]},
            "phase": "approach",
            "actors": [],
            "step": 0,
            "dt": 0.5,
        }
        state = self.builder.build(obs)
        # Per-actor features start at 25 (8+5+12), each 21 dims, last dim is mask
        for slot in range(3):
            mask_idx = 25 + slot * 21 + 20
            self.assertAlmostEqual(state[mask_idx], 0.0,
                                   msg=f"Empty slot {slot} should have mask=0")


class TestRewardConfig(unittest.TestCase):

    def test_reward_terms_in_config(self):
        try:
            import yaml
            with open("configs/interaction/benchmark.yaml") as f:
                cfg = yaml.safe_load(f)
            reward = cfg.get("reward", {})
            required = [
                "r_success", "r_collision", "r_row_violation",
                "r_conflict_intrusion", "r_forced_other_brake",
                "r_deadlock", "r_crosswalk_block", "r_progress",
                "r_time", "r_comfort", "r_unnecessary_wait",
            ]
            for term in required:
                self.assertIn(term, reward, f"Missing reward term: {term}")
        except ImportError:
            self.skipTest("yaml not available")


if __name__ == "__main__":
    unittest.main()

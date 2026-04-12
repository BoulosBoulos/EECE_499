"""Tests for episode lifecycle and event detection (offline, no SUMO)."""

import unittest
import numpy as np

from state.builder_interaction import InteractionStateBuilder
from scenario.template_sampler import TemplateSampler, TEMPLATE_FAMILIES
from scenario.scheduler import solve_spawn_for_eta, solve_ped_spawn, solve_ego_preroll
from scenario.conflict_map import SCENARIO_CONFLICT_ROUTES


class TestEpisodeLifecycle(unittest.TestCase):

    def test_template_sampler_covers_all_scenarios(self):
        sampler = TemplateSampler(rng=np.random.RandomState(42))
        for sc in ["1a", "1b", "1c", "1d", "2", "3", "4"]:
            template = sampler.sample(scenario_id=sc)
            self.assertEqual(template.scenario_id, sc)
            self.assertIn(template.template_family, TEMPLATE_FAMILIES)

    def test_spawning_produces_valid_times(self):
        dt, dp = solve_spawn_for_eta(4.0, 11.0, "left_in", bar_len=50.0)
        self.assertGreaterEqual(dt, 0.0)
        self.assertLessEqual(dp, 49.0)

    def test_ego_preroll_positive(self):
        t = solve_ego_preroll(4.0, 8.0, stem_len=50.0)
        self.assertGreaterEqual(t, 0.0)

    def test_ped_spawn_within_bounds(self):
        dt, dp = solve_ped_spawn(5.0, 1.2, "left_in", bar_len=50.0)
        self.assertGreaterEqual(dp, 0.0)
        self.assertLessEqual(dp, 42.0)

    def test_state_builder_handles_no_actors(self):
        builder = InteractionStateBuilder(top_n=3)
        obs = {
            "ego": {"v": 0, "a": 0, "jerk": 0, "yaw_rate": 0,
                    "d_preentry": 30, "d_conflict_entry": 40,
                    "d_conflict_exit": 50, "ego_eta_enter": 5,
                    "psi": 0, "p": [50, -40]},
            "phase": "approach",
            "actors": [],
            "step": 0,
            "dt": 0.5,
        }
        state = builder.build(obs)
        self.assertEqual(state.shape, (91,))
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))

    def test_state_builder_handles_many_actors(self):
        builder = InteractionStateBuilder(top_n=3)
        actors = []
        for i in range(7):
            actors.append({
                "id": f"a{i}", "p": [40 + i, i], "psi": 0.1 * i,
                "v": 5 + i, "a": -0.1, "actor_type": "veh",
                "eta_enter": 2.0 + i, "eta_exit": 4.0 + i,
                "legal_priority": 0.0, "committed": 0, "yielding": 0,
                "crosswalk_progress": 0, "relevant": 1, "uncertainty": 0.1,
                "first_seen_step": 0, "in_conflict_zone": 0,
            })
        obs = {
            "ego": {"v": 8, "a": 0, "jerk": 0, "yaw_rate": 0,
                    "d_preentry": 10, "d_conflict_entry": 15,
                    "d_conflict_exit": 25, "ego_eta_enter": 1.8,
                    "psi": 0, "p": [50, 0]},
            "phase": "decision",
            "actors": actors,
            "step": 5,
            "dt": 0.5,
        }
        state = builder.build(obs)
        self.assertEqual(state.shape, (91,))

    def test_scenario_routes_are_non_empty(self):
        for sc in ["1a", "1b", "1c", "2", "3", "4"]:
            routes = SCENARIO_CONFLICT_ROUTES.get(sc, [])
            self.assertGreater(len(routes), 0,
                               f"Scenario {sc} has no conflict routes")

    def test_pothole_scenario_has_empty_routes(self):
        routes = SCENARIO_CONFLICT_ROUTES.get("1d", [])
        self.assertEqual(len(routes), 0, "1d should have no social conflict routes")


class TestEvalManifest(unittest.TestCase):

    def test_generate_manifest(self):
        from experiments.interaction.eval_manifest import generate_manifest, manifest_summary
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.json")
            manifest = generate_manifest(
                scenarios=["1a", "1b"],
                episodes_per_scenario=5,
                seeds=[42, 123],
                out_path=path,
            )
            self.assertEqual(len(manifest), 2 * 2 * 5)  # 2 scenarios × 2 seeds × 5 eps
            self.assertTrue(os.path.isfile(path))

            summary = manifest_summary(manifest)
            self.assertEqual(summary["total_episodes"], 20)
            self.assertIn("1a", summary["scenarios"])
            self.assertIn("1b", summary["scenarios"])


if __name__ == "__main__":
    unittest.main()

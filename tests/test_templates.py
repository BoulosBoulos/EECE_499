"""Tests for the template sampler and conflict map."""

import numpy as np
import unittest

from scenario.conflict_map import (
    ROUTE_CONFLICTS, SCENARIO_CONFLICT_ROUTES, CONFLICT_ZONES,
    get_conflict_for_route, get_routes_for_scenario,
    legal_priority_for_ego_vs, PRIO_ACTOR, PRIO_EGO,
)
from scenario.template_sampler import (
    TemplateSampler, TEMPLATE_FAMILIES, PED_TEMPLATES,
    DEFAULT_TEMPLATE_PROBS, DEFAULT_ETA_BANDS,
)


class TestConflictMap(unittest.TestCase):

    def test_all_routes_have_edges(self):
        for name, rpc in ROUTE_CONFLICTS.items():
            self.assertTrue(len(rpc.actor_edges.split()) >= 2,
                            f"Route {name} has too few edges: {rpc.actor_edges}")

    def test_scenario_routes_exist(self):
        for sc, routes in SCENARIO_CONFLICT_ROUTES.items():
            for r in routes:
                self.assertIn(r, ROUTE_CONFLICTS,
                              f"Scenario {sc} references unknown route {r}")

    def test_ego_always_minor_priority(self):
        for name, rpc in ROUTE_CONFLICTS.items():
            if rpc.actor_type != "ped":
                self.assertLessEqual(rpc.legal_priority, 0.5,
                                     f"Route {name}: ego should not have priority over major road traffic")

    def test_conflict_zones_defined(self):
        self.assertTrue(len(CONFLICT_ZONES) >= 4)
        for cz_id, cz in CONFLICT_ZONES.items():
            self.assertGreater(cz.radius_m, 0)

    def test_get_routes_for_scenario(self):
        routes_1a = get_routes_for_scenario("1a")
        self.assertTrue(len(routes_1a) >= 2)
        for rpc in routes_1a:
            self.assertNotEqual(rpc.actor_type, "ped")

        routes_1b = get_routes_for_scenario("1b")
        self.assertTrue(len(routes_1b) >= 1)
        for rpc in routes_1b:
            self.assertEqual(rpc.actor_type, "ped")

    def test_legal_priority_lookup(self):
        p = legal_priority_for_ego_vs("car_left_right")
        self.assertEqual(p, PRIO_ACTOR)
        p_unknown = legal_priority_for_ego_vs("nonexistent_route")
        self.assertEqual(p_unknown, 0.5)


class TestTemplateSampler(unittest.TestCase):

    def test_all_families_have_eta_bands(self):
        for fam in TEMPLATE_FAMILIES:
            self.assertIn(fam, DEFAULT_ETA_BANDS)

    def test_all_families_have_probs(self):
        for fam in TEMPLATE_FAMILIES:
            self.assertIn(fam, DEFAULT_TEMPLATE_PROBS)

    def test_probs_sum_to_one(self):
        total = sum(DEFAULT_TEMPLATE_PROBS.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_sample_returns_template(self):
        sampler = TemplateSampler(rng=np.random.RandomState(42))
        for sc in ["1a", "1b", "1c", "1d", "2", "3", "4"]:
            template = sampler.sample(scenario_id=sc)
            self.assertIn(template.template_family, TEMPLATE_FAMILIES)
            self.assertEqual(template.scenario_id, sc)

    def test_ped_templates_only_for_ped_scenarios(self):
        sampler = TemplateSampler(rng=np.random.RandomState(42))
        for _ in range(50):
            template = sampler.sample(scenario_id="1a")
            self.assertNotIn(template.template_family, PED_TEMPLATES,
                             "1a should not produce pedestrian templates")

    def test_template_actors_have_required_fields(self):
        sampler = TemplateSampler(rng=np.random.RandomState(42))
        template = sampler.sample(scenario_id="3")
        for actor in template.actors:
            self.assertIn(actor.actor_type, ["veh", "ped", "cyc"])
            self.assertIn(actor.intent, ["yield", "proceed", "hesitate", "violate"])
            self.assertIn(actor.style, ["timid", "nominal", "assertive", "fast"])
            self.assertGreater(actor.approach_speed, 0)

    def test_eta_bands_respected(self):
        sampler = TemplateSampler(rng=np.random.RandomState(99))
        for _ in range(100):
            template = sampler.sample(scenario_id="1a")
            fam = template.template_family
            band = DEFAULT_ETA_BANDS[fam]
            for actor in template.actors:
                self.assertGreaterEqual(actor.delta_eta, band[0] - 0.01)
                self.assertLessEqual(actor.delta_eta, band[1] + 0.01)


class TestScheduler(unittest.TestCase):

    def test_spawn_for_eta_non_negative(self):
        from scenario.scheduler import solve_spawn_for_eta
        dt, dp = solve_spawn_for_eta(3.0, 11.0, "left_in")
        self.assertGreaterEqual(dt, 0.0)
        self.assertGreaterEqual(dp, 0.0)

    def test_ped_spawn_non_negative(self):
        from scenario.scheduler import solve_ped_spawn
        dt, dp = solve_ped_spawn(5.0, 1.2, "left_in")
        self.assertGreaterEqual(dt, 0.0)
        self.assertGreaterEqual(dp, 0.0)

    def test_ego_preroll(self):
        from scenario.scheduler import solve_ego_preroll
        t = solve_ego_preroll(4.0, 8.0, stem_len=50.0)
        self.assertGreaterEqual(t, 0.0)


if __name__ == "__main__":
    unittest.main()

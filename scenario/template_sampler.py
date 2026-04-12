"""Template-family sampler for the interaction benchmark.

Instead of sampling actors independently, each episode is generated from a
named *template* that specifies the intended conflict structure, timing
relation, and actor behavior style.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scenario.conflict_map import (
    ROUTE_CONFLICTS,
    SCENARIO_CONFLICT_ROUTES,
    RoutePairConflict,
    PRIO_ACTOR,
    PRIO_EGO,
    PRIO_AMBIGUOUS,
)

# ── Template family names ───────────────────────────────────────────────────

TEMPLATE_FAMILIES = [
    "ego_first_clear",
    "actor_first_clear",
    "tight_gap_ambiguous",
    "actor_yields",
    "actor_violation",
    "ped_committed",
    "ped_waiting",
    "ped_hesitant_then_commit",
    "late_fast_entry",
]

# Templates that require a pedestrian actor
PED_TEMPLATES = {"ped_committed", "ped_waiting", "ped_hesitant_then_commit"}

# Default ETA bands (delta_eta = actor_eta − ego_eta, in seconds)
DEFAULT_ETA_BANDS: Dict[str, Tuple[float, float]] = {
    "ego_first_clear":           ( 1.5,  3.5),
    "actor_first_clear":         (-3.5, -1.5),
    "tight_gap_ambiguous":       (-0.8,  0.8),
    "actor_yields":              ( 0.5,  2.0),
    "actor_violation":           (-0.5,  0.5),
    "ped_committed":             (-2.0, -0.5),
    "ped_waiting":               ( 1.0,  3.0),
    "ped_hesitant_then_commit":  ( 0.0,  1.5),
    "late_fast_entry":           (-0.3,  0.3),
}

# Default template sampling probabilities
DEFAULT_TEMPLATE_PROBS: Dict[str, float] = {
    "ego_first_clear":           0.20,
    "actor_first_clear":         0.25,
    "tight_gap_ambiguous":       0.20,
    "actor_yields":              0.10,
    "actor_violation":           0.05,
    "ped_committed":             0.05,
    "ped_waiting":               0.05,
    "ped_hesitant_then_commit":  0.05,
    "late_fast_entry":           0.05,
}


@dataclass
class ActorSpec:
    """Specification for one actor in the episode, derived from the template."""
    route_name: str
    route_edges: str
    actor_type: str               # "veh" | "ped" | "cyc"
    target_eta_enter: float       # target time from episode start to conflict entry
    delta_eta: float              # actor_eta − ego_eta (positive = ego arrives first)
    legal_priority: float
    conflict_type: str
    primary_cz: str
    # Latent behaviour parameters
    style: str = "nominal"        # "timid" | "nominal" | "assertive"
    intent: str = "proceed"       # "yield" | "proceed" | "hesitate" | "violate"
    approach_speed: float = 11.11
    gap_accept_s: float = 3.0
    assertiveness: float = 0.5    # 0 = very cautious, 1 = very aggressive
    ped_speed: float = 1.2
    ped_hesitant: bool = False
    ped_stop_midway: bool = False
    ped_stop_duration: float = 0.0


@dataclass
class EpisodeTemplate:
    """Complete template for one benchmark episode."""
    template_family: str
    scenario_id: str
    ego_target_eta_enter: float         # target ego ETA to conflict entry (s)
    actors: List[ActorSpec] = field(default_factory=list)
    has_pothole: bool = False
    pothole_x: float = 0.0
    pothole_y: float = -3.0


# ── Style / intent tables ───────────────────────────────────────────────────

VEH_STYLES = {
    "timid":      {"approach_speed": 9.0,  "gap_accept_s": 4.0, "assertiveness": 0.2},
    "nominal":    {"approach_speed": 11.11, "gap_accept_s": 3.0, "assertiveness": 0.5},
    "assertive":  {"approach_speed": 13.89, "gap_accept_s": 2.0, "assertiveness": 0.8},
}

MOTO_STYLES = {
    "timid":      {"approach_speed": 11.0,  "gap_accept_s": 4.0, "assertiveness": 0.2},
    "nominal":    {"approach_speed": 13.89, "gap_accept_s": 2.5, "assertiveness": 0.5},
    "assertive":  {"approach_speed": 16.67, "gap_accept_s": 1.5, "assertiveness": 0.9},
}

PED_STYLES = {
    "timid":      {"ped_speed": 0.8,  "gap_accept_s": 5.0, "assertiveness": 0.1},
    "nominal":    {"ped_speed": 1.2,  "gap_accept_s": 4.0, "assertiveness": 0.4},
    "assertive":  {"ped_speed": 2.0,  "gap_accept_s": 2.0, "assertiveness": 0.8},
    "fast":       {"ped_speed": 2.5,  "gap_accept_s": 1.5, "assertiveness": 0.9},
}


def _style_table(actor_type: str) -> dict:
    if actor_type == "ped":
        return PED_STYLES
    if actor_type == "cyc":
        return MOTO_STYLES
    return VEH_STYLES


def _intent_for_template(family: str) -> str:
    """Determine the latent intent label from the template family."""
    mapping = {
        "ego_first_clear":          "proceed",
        "actor_first_clear":        "proceed",
        "tight_gap_ambiguous":      "proceed",
        "actor_yields":             "yield",
        "actor_violation":          "violate",
        "ped_committed":            "proceed",
        "ped_waiting":              "yield",
        "ped_hesitant_then_commit": "hesitate",
        "late_fast_entry":          "proceed",
    }
    return mapping.get(family, "proceed")


def _style_for_template(family: str, rng: np.random.RandomState) -> str:
    """Pick an actor style that matches the template intent."""
    if family in ("actor_violation", "late_fast_entry"):
        return "assertive"
    if family == "actor_yields":
        return "timid"
    if family in ("ped_waiting",):
        return "timid"
    if family in ("ped_committed", "ped_hesitant_then_commit"):
        return rng.choice(["nominal", "assertive"])
    return rng.choice(["timid", "nominal", "assertive"])


class TemplateSampler:
    """Sample a complete EpisodeTemplate for a given scenario."""

    def __init__(
        self,
        template_probs: Dict[str, float] | None = None,
        eta_bands: Dict[str, Tuple[float, float]] | None = None,
        rng: np.random.RandomState | None = None,
    ):
        self.rng = rng or np.random.RandomState()
        self.template_probs = template_probs or dict(DEFAULT_TEMPLATE_PROBS)
        self.eta_bands = eta_bands or dict(DEFAULT_ETA_BANDS)

    def sample(
        self,
        scenario_id: str,
        has_pothole: bool = False,
        bar_len: float = 50.0,
        stem_len: float = 50.0,
    ) -> EpisodeTemplate:
        """Sample an episode template for the given scenario."""
        available_routes = SCENARIO_CONFLICT_ROUTES.get(scenario_id, [])
        has_ped_routes = any(
            ROUTE_CONFLICTS[r].actor_type == "ped"
            for r in available_routes if r in ROUTE_CONFLICTS
        )
        has_veh_routes = any(
            ROUTE_CONFLICTS[r].actor_type in ("veh", "cyc")
            for r in available_routes if r in ROUTE_CONFLICTS
        )

        # Build valid template list for this scenario
        valid_templates = []
        valid_probs = []
        for fam in TEMPLATE_FAMILIES:
            if fam in PED_TEMPLATES and not has_ped_routes:
                continue
            if fam not in PED_TEMPLATES and not has_veh_routes:
                if fam not in PED_TEMPLATES:
                    continue
            p = self.template_probs.get(fam, 0.0)
            if p > 0:
                valid_templates.append(fam)
                valid_probs.append(p)

        if not valid_templates:
            # Fallback: pothole-only or trivial
            return EpisodeTemplate(
                template_family="ego_first_clear",
                scenario_id=scenario_id,
                ego_target_eta_enter=3.0,
                has_pothole=has_pothole,
            )

        prob_arr = np.array(valid_probs, dtype=float)
        prob_arr /= prob_arr.sum()
        family = self.rng.choice(valid_templates, p=prob_arr)

        ego_target_eta = self.rng.uniform(3.0, 5.0)
        eta_band = self.eta_bands.get(family, (-1.0, 1.0))

        # Select appropriate actor route(s)
        actors: List[ActorSpec] = []

        if family in PED_TEMPLATES:
            ped_routes = [r for r in available_routes
                          if r in ROUTE_CONFLICTS and
                          ROUTE_CONFLICTS[r].actor_type == "ped"]
            if ped_routes:
                route_name = self.rng.choice(ped_routes)
                actors.append(self._make_actor_spec(
                    route_name, family, ego_target_eta, eta_band,
                ))
            # Add a secondary vehicle actor for mixed scenarios (2, 3, 4)
            veh_routes = [r for r in available_routes
                          if r in ROUTE_CONFLICTS and
                          ROUTE_CONFLICTS[r].actor_type != "ped"]
            if veh_routes and scenario_id in ("2", "3", "4"):
                sec_route = self.rng.choice(veh_routes)
                sec_band = (1.5, 4.0)  # secondary arrives after to reduce chaos
                actors.append(self._make_actor_spec(
                    sec_route, "ego_first_clear", ego_target_eta, sec_band,
                ))
        else:
            # Vehicle / motorcycle template
            veh_routes = [r for r in available_routes
                          if r in ROUTE_CONFLICTS and
                          ROUTE_CONFLICTS[r].actor_type != "ped"]
            if veh_routes:
                route_name = self.rng.choice(veh_routes)
                actors.append(self._make_actor_spec(
                    route_name, family, ego_target_eta, eta_band,
                ))
            # Add pedestrian as secondary if scenario has one
            ped_routes = [r for r in available_routes
                          if r in ROUTE_CONFLICTS and
                          ROUTE_CONFLICTS[r].actor_type == "ped"]
            if ped_routes and scenario_id in ("2", "3", "4"):
                sec_route = self.rng.choice(ped_routes)
                sec_band = (2.0, 5.0)
                actors.append(self._make_actor_spec(
                    sec_route, "ped_waiting", ego_target_eta, sec_band,
                ))
            # Add motorcycle as third actor for scenario 3, 4
            cyc_routes = [r for r in available_routes
                          if r in ROUTE_CONFLICTS and
                          ROUTE_CONFLICTS[r].actor_type == "cyc"]
            if cyc_routes and scenario_id in ("3", "4"):
                existing_routes = {a.route_name for a in actors}
                available_cyc = [r for r in cyc_routes if r not in existing_routes]
                if available_cyc:
                    sec_route = self.rng.choice(available_cyc)
                    sec_band = (1.0, 3.0)
                    actors.append(self._make_actor_spec(
                        sec_route, "actor_first_clear", ego_target_eta, sec_band,
                    ))

        pothole_x = float(self.rng.uniform(-2, 5)) if has_pothole else 0.0
        pothole_y = float(self.rng.uniform(-4, 2)) if has_pothole else 0.0

        return EpisodeTemplate(
            template_family=family,
            scenario_id=scenario_id,
            ego_target_eta_enter=ego_target_eta,
            actors=actors,
            has_pothole=has_pothole,
            pothole_x=pothole_x,
            pothole_y=pothole_y,
        )

    def _make_actor_spec(
        self,
        route_name: str,
        family: str,
        ego_target_eta: float,
        eta_band: Tuple[float, float],
    ) -> ActorSpec:
        """Build a single ActorSpec from the conflict map and template family."""
        rpc = ROUTE_CONFLICTS[route_name]
        delta_eta = float(self.rng.uniform(eta_band[0], eta_band[1]))
        actor_eta = ego_target_eta + delta_eta

        style_name = _style_for_template(family, self.rng)
        intent = _intent_for_template(family)

        style_table = _style_table(rpc.actor_type)
        style_params = style_table.get(style_name, style_table.get("nominal", {}))

        ped_hesitant = (family == "ped_hesitant_then_commit")
        ped_stop_midway = self.rng.random() < 0.3 if rpc.actor_type == "ped" else False
        ped_stop_dur = float(self.rng.uniform(1.5, 3.5)) if ped_stop_midway else 0.0

        return ActorSpec(
            route_name=route_name,
            route_edges=rpc.actor_edges,
            actor_type=rpc.actor_type,
            target_eta_enter=actor_eta,
            delta_eta=delta_eta,
            legal_priority=rpc.legal_priority,
            conflict_type=rpc.conflict_type,
            primary_cz=rpc.primary_cz,
            style=style_name,
            intent=intent,
            approach_speed=style_params.get("approach_speed",
                                            style_params.get("ped_speed", 1.2)),
            gap_accept_s=style_params.get("gap_accept_s", 3.0),
            assertiveness=style_params.get("assertiveness", 0.5),
            ped_speed=style_params.get("ped_speed", 1.2),
            ped_hesitant=ped_hesitant,
            ped_stop_midway=ped_stop_midway,
            ped_stop_duration=ped_stop_dur,
        )

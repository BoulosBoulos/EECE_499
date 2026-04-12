"""Conflict-zone metadata for the T-intersection behavioral benchmark.

Defines which route pairs actually conflict with the ego's fixed route
(stem_in -> right_out), the legal priority relation, and approximate
distances from each edge start to the shared conflict zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Conflict zone identifiers ───────────────────────────────────────────────

CZ_PREENTRY = "cz_preentry"
CZ_TURN_CORE = "cz_turn_core"
CZ_MERGE_RIGHT = "cz_merge_right"
CZ_EXIT_CROSSWALK = "cz_exit_crosswalk"

ALL_CZ = [CZ_PREENTRY, CZ_TURN_CORE, CZ_MERGE_RIGHT, CZ_EXIT_CROSSWALK]


@dataclass(frozen=True)
class ConflictZone:
    """Static description of a named conflict zone."""
    cz_id: str
    description: str
    # Rough center relative to the junction center (SUMO world coords added at
    # runtime once the net-offset is known).
    offset_x: float = 0.0
    offset_y: float = 0.0
    radius_m: float = 8.0


CONFLICT_ZONES: Dict[str, ConflictZone] = {
    CZ_PREENTRY: ConflictZone(
        cz_id=CZ_PREENTRY,
        description="Virtual decision boundary before ego enters the junction",
        offset_x=0.0, offset_y=-10.0, radius_m=5.0,
    ),
    CZ_TURN_CORE: ConflictZone(
        cz_id=CZ_TURN_CORE,
        description="Central turn region used by ego and turning/through vehicles",
        offset_x=3.0, offset_y=0.0, radius_m=10.0,
    ),
    CZ_MERGE_RIGHT: ConflictZone(
        cz_id=CZ_MERGE_RIGHT,
        description="Merge area where ego enters right_out",
        offset_x=8.0, offset_y=0.0, radius_m=6.0,
    ),
    CZ_EXIT_CROSSWALK: ConflictZone(
        cz_id=CZ_EXIT_CROSSWALK,
        description="Pedestrian crossing over right_out near junction",
        offset_x=10.0, offset_y=0.0, radius_m=5.0,
    ),
}


# ── Priority constants ──────────────────────────────────────────────────────
# 1.0 = ego has priority,  0.0 = actor has priority,  0.5 = ambiguous
PRIO_EGO = 1.0
PRIO_ACTOR = 0.0
PRIO_AMBIGUOUS = 0.5


@dataclass(frozen=True)
class RoutePairConflict:
    """Describes the conflict between the ego route and one actor route."""
    actor_route_name: str
    actor_edges: str                       # space-separated SUMO edges
    primary_cz: str                        # dominant conflict zone id
    legal_priority: float                  # PRIO_EGO / PRIO_ACTOR / PRIO_AMBIGUOUS
    conflict_type: str                     # "crossing" | "merging" | "blocking"
    actor_dist_to_cz_approx_m: float       # from edge start to CZ entry
    ego_dist_to_cz_approx_m: float         # from stem_in start to CZ entry
    actor_type: str = "veh"                # "veh" | "ped" | "cyc"


# The ego always travels  stem_in -> right_out  (priority = -1, minor road).
# Bar edges (left_in / right_in) have priority = +1 (major road).

ROUTE_CONFLICTS: Dict[str, RoutePairConflict] = {
    # ── Cars ────────────────────────────────────────────────────────────────
    "car_left_right": RoutePairConflict(
        actor_route_name="car_left_right",
        actor_edges="left_in right_out",
        primary_cz=CZ_MERGE_RIGHT,
        legal_priority=PRIO_ACTOR,         # major-road through traffic
        conflict_type="merging",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
    ),
    "car_right_left": RoutePairConflict(
        actor_route_name="car_right_left",
        actor_edges="right_in left_out",
        primary_cz=CZ_TURN_CORE,
        legal_priority=PRIO_ACTOR,         # major-road through traffic
        conflict_type="crossing",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
    ),
    "car_left_stem": RoutePairConflict(
        actor_route_name="car_left_stem",
        actor_edges="left_in stem_out",
        primary_cz=CZ_TURN_CORE,
        legal_priority=PRIO_ACTOR,
        conflict_type="crossing",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
    ),
    "car_right_stem": RoutePairConflict(
        actor_route_name="car_right_stem",
        actor_edges="right_in stem_out",
        primary_cz=CZ_TURN_CORE,
        legal_priority=PRIO_ACTOR,
        conflict_type="crossing",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
    ),

    # ── Motorcycles ─────────────────────────────────────────────────────────
    "moto_right_left": RoutePairConflict(
        actor_route_name="moto_right_left",
        actor_edges="right_in left_out",
        primary_cz=CZ_TURN_CORE,
        legal_priority=PRIO_ACTOR,
        conflict_type="crossing",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
        actor_type="cyc",
    ),
    "moto_left_right": RoutePairConflict(
        actor_route_name="moto_left_right",
        actor_edges="left_in right_out",
        primary_cz=CZ_MERGE_RIGHT,
        legal_priority=PRIO_ACTOR,
        conflict_type="merging",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
        actor_type="cyc",
    ),
    "moto_right_stem": RoutePairConflict(
        actor_route_name="moto_right_stem",
        actor_edges="right_in stem_out",
        primary_cz=CZ_TURN_CORE,
        legal_priority=PRIO_ACTOR,
        conflict_type="crossing",
        actor_dist_to_cz_approx_m=42.0,
        ego_dist_to_cz_approx_m=44.0,
        actor_type="cyc",
    ),

    # ── Pedestrians ─────────────────────────────────────────────────────────
    "ped_cross_right_out": RoutePairConflict(
        actor_route_name="ped_cross_right_out",
        actor_edges="left_in right_out",      # sidewalk route; from/to edges
        primary_cz=CZ_EXIT_CROSSWALK,
        legal_priority=PRIO_ACTOR,             # pedestrians on crosswalk
        conflict_type="blocking",
        actor_dist_to_cz_approx_m=42.0,       # sidewalk distance to crossing
        ego_dist_to_cz_approx_m=48.0,         # ego must pass through crosswalk
        actor_type="ped",
    ),
}


# Routes that are meaningfully conflicting with ego per scenario
SCENARIO_CONFLICT_ROUTES: Dict[str, List[str]] = {
    "1a": ["car_right_left", "car_left_right", "car_right_stem", "car_left_stem"],
    "1b": ["ped_cross_right_out"],
    "1c": ["moto_right_left", "moto_left_right", "moto_right_stem"],
    "1d": [],
    "2":  ["car_right_left", "car_left_right", "ped_cross_right_out"],
    "3":  ["car_right_left", "car_left_right", "ped_cross_right_out",
            "moto_right_left", "moto_left_right"],
    "4":  ["car_right_left", "car_left_right", "ped_cross_right_out",
            "moto_right_left", "moto_left_right"],
}


def get_conflict_for_route(route_name: str) -> RoutePairConflict | None:
    """Look up the conflict descriptor for an actor route name."""
    return ROUTE_CONFLICTS.get(route_name)


def get_routes_for_scenario(scenario: str) -> List[RoutePairConflict]:
    """Return all meaningful conflict routes for a given scenario id."""
    names = SCENARIO_CONFLICT_ROUTES.get(scenario, [])
    return [ROUTE_CONFLICTS[n] for n in names if n in ROUTE_CONFLICTS]


def ego_dist_to_cz(stem_len: float) -> float:
    """Approximate ego distance from stem start to the junction conflict."""
    return stem_len - 6.0  # junction internal area starts ~6m before edge end


def actor_dist_to_cz(edge_id: str, bar_len: float, stem_len: float) -> float:
    """Approximate actor distance from edge start to the junction conflict."""
    if "stem" in edge_id:
        return stem_len - 6.0
    return bar_len - 8.0


def legal_priority_for_ego_vs(route_name: str) -> float:
    """Return the legal priority value: 0 = actor has priority, 1 = ego has it."""
    rpc = ROUTE_CONFLICTS.get(route_name)
    if rpc is None:
        return PRIO_AMBIGUOUS
    return rpc.legal_priority

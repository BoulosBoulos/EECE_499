"""Conflict-based actor scheduling.

Given a target ETA at the conflict zone and an approach speed, solve for
the SUMO depart_time and depart_pos so that the actor actually arrives at
the right time.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from scenario.conflict_map import actor_dist_to_cz, ego_dist_to_cz


def solve_spawn_for_eta(
    target_eta_s: float,
    approach_speed_mps: float,
    edge_id: str,
    bar_len: float = 50.0,
    stem_len: float = 50.0,
    edge_length: float | None = None,
) -> Tuple[float, float]:
    """Compute (depart_time, depart_pos) so the actor reaches the CZ at *target_eta_s*.

    Parameters
    ----------
    target_eta_s : float
        Desired simulation time when the actor reaches the conflict zone entry.
    approach_speed_mps : float
        Cruising speed of the actor.
    edge_id : str
        First SUMO edge the actor departs from (e.g. "left_in", "right_in").
    bar_len, stem_len : float
        Network dimensions.
    edge_length : float | None
        Override the edge length (used when live TraCI length is available).

    Returns
    -------
    depart_time : float  (>= 0)
    depart_pos : float   (>= 0, clamped to edge_length − 1)
    """
    if edge_length is None:
        if "stem" in edge_id:
            edge_length = stem_len
        else:
            edge_length = bar_len

    dist_to_cz = actor_dist_to_cz(edge_id, bar_len, stem_len)
    dist_to_cz = min(dist_to_cz, edge_length - 1.0)

    speed = max(approach_speed_mps, 0.5)

    travel_time = dist_to_cz / speed
    depart_time = max(0.0, target_eta_s - travel_time)
    depart_pos = 0.0

    # If the actor should arrive very early (depart_time near zero but
    # travel_time still too long), place it closer to the CZ instead.
    if depart_time < 0.1 and travel_time > target_eta_s:
        needed_dist = speed * target_eta_s
        depart_pos = max(0.0, dist_to_cz - needed_dist)
        depart_time = 0.0

    depart_pos = float(np.clip(depart_pos, 0.0, max(0.0, edge_length - 2.0)))
    return round(depart_time, 2), round(depart_pos, 2)


def solve_ego_preroll(
    ego_target_eta_s: float,
    ego_approach_speed_mps: float,
    stem_len: float = 50.0,
) -> float:
    """Return the ego depart_time so it reaches the CZ at *ego_target_eta_s*.

    Ego always departs at pos=0 on stem_in.
    """
    dist = ego_dist_to_cz(stem_len)
    travel = dist / max(ego_approach_speed_mps, 0.5)
    return max(0.0, ego_target_eta_s - travel)


def solve_ped_spawn(
    target_eta_s: float,
    ped_speed_mps: float,
    from_edge: str,
    bar_len: float = 50.0,
    edge_length: float | None = None,
) -> Tuple[float, float]:
    """Compute pedestrian (depart_time, depart_pos) targeting crosswalk arrival.

    Pedestrians walk on the sidewalk lane (index 0) of their from_edge.
    The crosswalk is near the junction end of the edge, so the distance is
    approximately edge_length.
    """
    if edge_length is None:
        edge_length = bar_len

    # Pedestrian sidewalk edge is shorter than the vehicle edge; be conservative
    safe_edge = max(edge_length - 8.0, 10.0)
    dist_to_crosswalk = safe_edge
    speed = max(ped_speed_mps, 0.3)
    travel = dist_to_crosswalk / speed

    depart_time = max(0.0, target_eta_s - travel)
    depart_pos = 0.0

    if depart_time < 0.1 and travel > target_eta_s:
        needed_dist = speed * target_eta_s
        depart_pos = max(0.0, dist_to_crosswalk - needed_dist)
        depart_time = 0.0

    # Hard-clamp to avoid SUMO "Invalid departure position" errors
    depart_pos = float(np.clip(depart_pos, 0.0, max(0.0, safe_edge - 2.0)))
    return round(depart_time, 2), round(depart_pos, 2)

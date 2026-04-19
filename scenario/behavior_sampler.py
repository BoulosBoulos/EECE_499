"""Per-episode behavior sampler: diverse maneuvers, styles, and timing for all VRUs.

Each reset() call returns a BehaviorConfig that the env uses to spawn actors
with randomized routes, speeds, depart times, and TraCI-level control parameters.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Maneuver and style enums ────────────────────────────────────────────────

CAR_MANEUVERS = [
    "straight_left_right",   # left_in → right_out (default)
    "straight_right_left",   # right_in → left_out
    "turn_left",             # left_in → stem_out  (turns into stem)
    "turn_right",            # right_in → stem_out  (turns into stem)
]

CAR_STYLES = [
    "nominal",       # sigma=0.15, normal accel/decel, tau=0.5
    "aggressive",    # sigma=0.3,  higher accel, lower tau, higher jm
    "timid",         # sigma=0.05, lower accel, higher tau, jm=0
    "distracted",    # sigma=0.4,  higher tau, reaction delay
    "erratic",       # sigma=0.5,  oscillating speed
    "drunk",         # sigma=0.6,  very high sigma, swerving effect
    "rule_violating",# high jm, ignores right-of-way
]

PED_MANEUVERS = [
    "cross_left_right",      # left_in → right_out (sidewalk)
    "cross_right_left",      # right_out → left_in (reverse)
]

PED_STYLES = [
    "normal_walk",           # 1.2 m/s
    "running",               # 3.0 m/s
    "slow_elderly",          # 0.6 m/s
    "stop_midway",           # walks, stops in CZ for 2-5s, then continues
    "hesitant",              # starts, retreats, then goes
    "distracted_slow",       # 0.8 m/s, late start
    "jaywalking_fast",       # 2.5 m/s, departs early from unexpected position
]

MOTO_MANEUVERS = [
    "straight_right_left",   # right_in → left_out (default)
    "straight_left_right",   # left_in → right_out
    "turn_into_stem",        # right_in → stem_out
]

MOTO_STYLES = [
    "nominal",               # normal motorcycle behavior
    "aggressive_fast",       # high accel, high speed, low tau
    "cautious",              # lower speed, higher tau
    "late_brake",            # delays braking until very close
    "swerving",              # high sigma, erratic
    "yield_to_ego",          # high tau, low jm, lets ego go first
]


@dataclass
class ActorBehavior:
    """Sampled behavior for one actor."""
    maneuver: str
    style: str
    route_edges: str
    depart_time: float
    depart_pos: Optional[float] = None
    max_speed: float = 13.89
    accel: float = 2.5
    decel: float = 5.0
    sigma: float = 0.15
    tau: float = 0.5
    jm_ignore: float = 0.0
    speed_factor: float = 1.0
    ped_speed: float = 1.2
    # TraCI-level overrides applied during episode
    stop_midway: bool = False
    stop_duration: float = 0.0
    hesitant: bool = False
    color: str = "1,0,0"


@dataclass
class PotholeConfig:
    """Randomly placed pothole within the T-intersection conflict zone."""
    x: float = 0.0
    y: float = 0.0
    half_w: float = 3.0
    half_h: float = 1.5


@dataclass
class BehaviorConfig:
    """Full per-episode behavior configuration."""
    car: Optional[ActorBehavior] = None
    car2: Optional[ActorBehavior] = None
    car3: Optional[ActorBehavior] = None
    pedestrian: Optional[ActorBehavior] = None
    pedestrian2: Optional[ActorBehavior] = None
    pedestrian3: Optional[ActorBehavior] = None
    motorcycle: Optional[ActorBehavior] = None
    motorcycle2: Optional[ActorBehavior] = None
    pothole: Optional[PotholeConfig] = None
    # Ground-truth labels for intent model training
    car_intent_label: int = 0       # 0=yield/stop, 1=proceed, 2=turn
    car_style_label: int = 1        # 0=soft, 1=mid, 2=aggressive
    ped_intent_label: int = 1       # 0=wait, 1=cross, 2=retreat
    ped_style_label: int = 0        # 0=soft, 1=mid, 2=aggressive
    moto_intent_label: int = 1
    moto_style_label: int = 1


# ── Route definitions ───────────────────────────────────────────────────────

CAR_ROUTES = {
    "straight_left_right": "left_in right_out",
    "straight_right_left": "right_in left_out",
    "turn_left":           "left_in stem_out",
    "turn_right":          "right_in stem_out",
}

MOTO_ROUTES = {
    "straight_right_left": "right_in left_out",
    "straight_left_right": "left_in right_out",
    "turn_into_stem":      "right_in stem_out",
}


# ── Style parameter tables ─────────────────────────────────────────────────

CAR_STYLE_PARAMS = {
    "nominal":        {"sigma": 0.15, "accel": 2.5, "decel": 5.0, "tau": 0.5,  "jm": 0.0,  "max_speed": 13.89},
    "aggressive":     {"sigma": 0.3,  "accel": 4.0, "decel": 6.0, "tau": 0.3,  "jm": 0.15, "max_speed": 16.67},
    "timid":          {"sigma": 0.05, "accel": 1.5, "decel": 4.0, "tau": 1.0,  "jm": 0.0,  "max_speed": 11.11},
    "distracted":     {"sigma": 0.4,  "accel": 2.0, "decel": 4.0, "tau": 1.5,  "jm": 0.05, "max_speed": 13.89},
    "erratic":        {"sigma": 0.5,  "accel": 3.5, "decel": 7.0, "tau": 0.3,  "jm": 0.1,  "max_speed": 16.67},
    "drunk":          {"sigma": 0.6,  "accel": 2.0, "decel": 3.0, "tau": 1.2,  "jm": 0.2,  "max_speed": 11.11},
    "rule_violating": {"sigma": 0.2,  "accel": 3.0, "decel": 5.0, "tau": 0.3,  "jm": 0.2,  "max_speed": 16.67},
}

MOTO_STYLE_PARAMS = {
    "nominal":         {"sigma": 0.15, "accel": 4.0, "decel": 8.0, "tau": 0.4,  "jm": 0.0,  "max_speed": 16.67},
    "aggressive_fast": {"sigma": 0.3,  "accel": 6.0, "decel": 9.0, "tau": 0.2,  "jm": 0.15, "max_speed": 22.22},
    "cautious":        {"sigma": 0.05, "accel": 2.5, "decel": 6.0, "tau": 0.8,  "jm": 0.0,  "max_speed": 13.89},
    "late_brake":      {"sigma": 0.2,  "accel": 5.0, "decel": 9.0, "tau": 0.15, "jm": 0.1,  "max_speed": 19.44},
    "swerving":        {"sigma": 0.5,  "accel": 4.0, "decel": 8.0, "tau": 0.3,  "jm": 0.1,  "max_speed": 16.67},
    "yield_to_ego":    {"sigma": 0.05, "accel": 2.0, "decel": 6.0, "tau": 1.2,  "jm": 0.0,  "max_speed": 11.11},
}

PED_STYLE_PARAMS = {
    "normal_walk":      {"speed": 1.2,  "stop_midway": False, "hesitant": False},
    "running":          {"speed": 3.0,  "stop_midway": False, "hesitant": False},
    "slow_elderly":     {"speed": 0.6,  "stop_midway": False, "hesitant": False},
    "stop_midway":      {"speed": 1.2,  "stop_midway": True,  "hesitant": False},
    "hesitant":         {"speed": 1.0,  "stop_midway": False, "hesitant": True},
    "distracted_slow":  {"speed": 0.8,  "stop_midway": False, "hesitant": False},
    "jaywalking_fast":  {"speed": 2.5,  "stop_midway": False, "hesitant": False},
}

# ── Intent/style label mappings ─────────────────────────────────────────────

CAR_INTENT_LABELS = {
    "straight_left_right": 1, "straight_right_left": 1,
    "turn_left": 2, "turn_right": 2,
}
CAR_STYLE_LABELS = {
    "nominal": 1, "aggressive": 2, "timid": 0, "distracted": 0,
    "erratic": 2, "drunk": 2, "rule_violating": 2,
}
PED_INTENT_LABELS = {
    "cross_left_right": 1, "cross_right_left": 1,
}
PED_STYLE_LABELS = {
    "normal_walk": 1, "running": 2, "slow_elderly": 0,
    "stop_midway": 0, "hesitant": 0, "distracted_slow": 0,
    "jaywalking_fast": 2,
}
MOTO_INTENT_LABELS = {
    "straight_right_left": 1, "straight_left_right": 1,
    "turn_into_stem": 2,
}
MOTO_STYLE_LABELS = {
    "nominal": 1, "aggressive_fast": 2, "cautious": 0,
    "late_brake": 2, "swerving": 2, "yield_to_ego": 0,
}


# ── Style categories for behavioral robustness ablation ────────────────
NOMINAL_CAR_STYLES = ["nominal", "timid"]
ADVERSARIAL_CAR_STYLES = ["aggressive", "distracted", "erratic", "drunk", "rule_violating"]

NOMINAL_PED_STYLES = ["normal_walk", "slow_elderly"]
ADVERSARIAL_PED_STYLES = ["running", "stop_midway", "hesitant", "distracted_slow", "jaywalking_fast"]

NOMINAL_MOTO_STYLES = ["nominal", "cautious", "yield_to_ego"]
ADVERSARIAL_MOTO_STYLES = ["aggressive_fast", "late_brake", "swerving"]


class BehaviorSampler:
    """Sample diverse per-episode behavior configs."""

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def _compute_conflict_spawn(
        self,
        ego_approach_dist: float,
        ego_approach_speed: float,
        agent_approach_dist: float,
        agent_speed: float,
        bar_len: float,
    ) -> tuple[float, float]:
        """Compute depart_time and depart_pos so agent arrives at CZ near ego.

        Returns (depart_time, depart_pos) for the other agent.
        """
        ego_eta = ego_approach_dist / max(ego_approach_speed, 1.0)
        # Primary agent: tight window around competent ego arrival.
        # Insurance agent (separate method) covers slower ego speeds.
        conflict_offset = self.rng.uniform(-1.5, 1.5)
        agent_eta = ego_eta + conflict_offset

        depart_time = 0.5 + self.rng.uniform(0, 1.0)
        travel_time_needed = max(0.5, agent_eta - depart_time)
        travel_dist = agent_speed * travel_time_needed
        depart_pos = max(0, bar_len - travel_dist - 5.0)
        depart_pos = float(np.clip(depart_pos, 0.0, bar_len - 5.0))

        return depart_time, depart_pos

    def _compute_insurance_spawn(
        self,
        ego_approach_dist: float,
        ego_approach_speed: float,
        agent_approach_dist: float,
        agent_speed: float,
        bar_len: float,
    ) -> tuple[float, float]:
        """Compute depart_time/depart_pos for the INSURANCE agent.

        Timed for a slower ego (~1/3 competent speed) to catch episodes
        where the ego policy is early in training and moves at ~2 m/s.
        """
        slow_ego_eta = ego_approach_dist / max(ego_approach_speed / 3.0, 1.0)
        insurance_offset = self.rng.uniform(-2.0, 2.0)
        agent_eta = slow_ego_eta + insurance_offset
        depart_time = 2.0 + self.rng.uniform(0, 1.5)
        travel_time_needed = max(0.5, agent_eta - depart_time)
        travel_dist = agent_speed * travel_time_needed
        depart_pos = max(0, bar_len - travel_dist - 5.0)
        depart_pos = float(np.clip(depart_pos, 0.0, bar_len - 5.0))
        return depart_time, depart_pos

    def _conflicting_agent_routes(self, ego_maneuver: str) -> dict:
        """Return agent route keys that create meaningful conflict with the ego.

        For stem maneuvers, moto conflict routes include turn_into_stem
        (right_in -> stem_out) as a "linger" route — the moto turns into
        the stem and stays near the ego's path, instead of zooming straight
        through. Without it, 1c interaction rates fall to ~40%.
        """
        conflicts = {
            "stem_right": {
                "car": ["straight_left_right", "straight_right_left", "turn_left"],
                "moto": ["straight_right_left", "straight_left_right", "turn_into_stem"],
            },
            "stem_left": {
                "car": ["straight_right_left", "straight_left_right", "turn_right"],
                "moto": ["straight_right_left", "straight_left_right", "turn_into_stem"],
            },
            "right_left": {
                "car": ["turn_left", "turn_right"],
                "moto": ["turn_into_stem"],
            },
            "right_stem": {
                "car": ["straight_left_right", "turn_left"],
                "moto": ["straight_left_right"],
            },
            "left_right": {
                "car": ["turn_left", "turn_right"],
                "moto": ["turn_into_stem"],
            },
            "left_stem": {
                "car": ["straight_right_left", "turn_right"],
                "moto": ["straight_right_left"],
            },
        }
        return conflicts.get(ego_maneuver, conflicts["stem_right"])

    def sample(
        self,
        has_car: bool,
        has_ped: bool,
        has_moto: bool,
        has_pothole: bool,
        bar_len: float = 50.0,
        stem_len: float = 60.0,
        ego_maneuver: str = "stem_right",
        dense: bool = False,
        style_filter: str | None = None,
        jm_ignore_fixed: float | None = None,
    ) -> BehaviorConfig:
        cfg = BehaviorConfig()

        # Ego approach distance to CZ depends on starting arm
        if ego_maneuver.startswith("stem"):
            ego_approach_dist = stem_len - 10
            ego_approach_speed = 6.0
        else:
            ego_approach_dist = bar_len - 10
            ego_approach_speed = 10.0
        conflict_routes = self._conflicting_agent_routes(ego_maneuver)

        # Style pools based on filter
        if style_filter == "nominal":
            car_style_pool = NOMINAL_CAR_STYLES
            ped_style_pool = NOMINAL_PED_STYLES
            moto_style_pool = NOMINAL_MOTO_STYLES
        elif style_filter == "adversarial":
            car_style_pool = ADVERSARIAL_CAR_STYLES
            ped_style_pool = ADVERSARIAL_PED_STYLES
            moto_style_pool = ADVERSARIAL_MOTO_STYLES
        else:
            car_style_pool = CAR_STYLES
            ped_style_pool = PED_STYLES
            moto_style_pool = MOTO_STYLES

        if has_car:
            # 90% conflict route for higher interaction density
            if self.rng.uniform() < 0.9 and conflict_routes.get("car"):
                maneuver = self.rng.choice(conflict_routes["car"])
            else:
                maneuver = self.rng.choice(CAR_MANEUVERS)
            style = self.rng.choice(car_style_pool)
            sp = CAR_STYLE_PARAMS[style]
            depart, dep_pos = self._compute_conflict_spawn(
                ego_approach_dist=ego_approach_dist,
                ego_approach_speed=ego_approach_speed,
                agent_approach_dist=bar_len - 5.0,
                agent_speed=sp["max_speed"],
                bar_len=bar_len,
            )
            cfg.car = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=CAR_ROUTES[maneuver],
                depart_time=depart,
                depart_pos=dep_pos,
                max_speed=sp["max_speed"] * (0.9 + self.rng.uniform(0, 0.2)),
                accel=sp["accel"],
                decel=sp["decel"],
                sigma=sp["sigma"],
                tau=sp["tau"],
                jm_ignore=jm_ignore_fixed if jm_ignore_fixed is not None else sp["jm"],
                color="1,0,0",
            )
            cfg.car_intent_label = CAR_INTENT_LABELS.get(maneuver, 1)
            cfg.car_style_label = CAR_STYLE_LABELS.get(style, 1)

            # ALWAYS spawn insurance car (car2) — catches slow-ego episodes
            if len(conflict_routes.get("car", [])) > 1:
                man2_choices = [m for m in conflict_routes["car"] if m != maneuver]
                man2 = self.rng.choice(man2_choices) if man2_choices else self.rng.choice(CAR_MANEUVERS)
            else:
                man2 = maneuver
            style2 = self.rng.choice(car_style_pool)
            sp2 = CAR_STYLE_PARAMS[style2]
            dep2, dpos2 = self._compute_insurance_spawn(
                ego_approach_dist=ego_approach_dist,
                ego_approach_speed=ego_approach_speed,
                agent_approach_dist=bar_len - 5.0,
                agent_speed=sp2["max_speed"],
                bar_len=bar_len,
            )
            cfg.car2 = ActorBehavior(
                maneuver=man2, style=style2,
                route_edges=CAR_ROUTES[man2],
                depart_time=dep2,
                depart_pos=dpos2,
                max_speed=sp2["max_speed"] * (0.9 + self.rng.uniform(0, 0.2)),
                accel=sp2["accel"], decel=sp2["decel"],
                sigma=sp2["sigma"], tau=sp2["tau"],
                jm_ignore=jm_ignore_fixed if jm_ignore_fixed is not None else sp2["jm"],
                color="0.8,0,0",
            )

            # Dense: third car at medium-speed timing
            if dense:
                man3 = self.rng.choice(conflict_routes.get("car", CAR_MANEUVERS))
                style3 = self.rng.choice(car_style_pool)
                sp3 = CAR_STYLE_PARAMS[style3]
                dep3, dpos3 = self._compute_conflict_spawn(
                    ego_approach_dist=ego_approach_dist,
                    ego_approach_speed=ego_approach_speed * 0.6,
                    agent_approach_dist=bar_len - 5.0,
                    agent_speed=sp3["max_speed"],
                    bar_len=bar_len,
                )
                cfg.car3 = ActorBehavior(
                    maneuver=man3, style=style3,
                    route_edges=CAR_ROUTES[man3],
                    depart_time=dep3 + 0.5,
                    depart_pos=dpos3,
                    max_speed=sp3["max_speed"] * (0.9 + self.rng.uniform(0, 0.2)),
                    accel=sp3["accel"], decel=sp3["decel"],
                    sigma=sp3["sigma"], tau=sp3["tau"],
                    jm_ignore=jm_ignore_fixed if jm_ignore_fixed is not None else sp3["jm"],
                    color="0.6,0,0",
                )

        if has_ped:
            maneuver = self.rng.choice(PED_MANEUVERS)
            style = self.rng.choice(ped_style_pool)
            sp = PED_STYLE_PARAMS[style]
            # Ped should enter crosswalk when ego is 5-15m from CZ
            ego_close_dist = self.rng.uniform(5.0, 15.0)
            ego_approach_speed = 6.0
            ego_eta_to_close = (ego_approach_dist - ego_close_dist) / max(ego_approach_speed, 1.0)
            ped_depart_time = 0.5 + self.rng.uniform(0, 0.5)
            ped_travel_time = max(0.5, ego_eta_to_close - ped_depart_time)
            ped_travel_dist = sp["speed"] * ped_travel_time
            safe_len = min(bar_len, 38.0)
            depart_pos = max(0, safe_len - ped_travel_dist - 3.0)
            depart_pos = float(np.clip(depart_pos, 0.0, safe_len - 2.0))
            cfg.pedestrian = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=f"{'left_in' if 'left_right' in maneuver else 'right_out'} {'right_out' if 'left_right' in maneuver else 'left_in'}",
                depart_time=ped_depart_time,
                depart_pos=depart_pos,
                ped_speed=sp["speed"],
                stop_midway=sp["stop_midway"],
                stop_duration=self.rng.uniform(2, 5) if sp["stop_midway"] else 0,
                hesitant=sp["hesitant"],
                color="0,0.9,0",
            )
            cfg.ped_intent_label = PED_INTENT_LABELS.get(maneuver, 1)
            cfg.ped_style_label = PED_STYLE_LABELS.get(style, 1)

            # ALWAYS spawn insurance pedestrian (ped2) from opposite direction
            man2 = "cross_right_left" if maneuver == "cross_left_right" else "cross_left_right"
            style2 = self.rng.choice(ped_style_pool)
            sp2 = PED_STYLE_PARAMS[style2]
            # Insurance ped: timed for slower ego (~1/3 speed)
            slow_ego_eta = (ego_approach_dist - self.rng.uniform(5.0, 15.0)) / max(ego_approach_speed / 3.0, 1.0)
            dep_time2 = 2.0 + self.rng.uniform(0, 1.0)
            trav_time2 = max(0.5, slow_ego_eta - dep_time2)
            trav_dist2 = sp2["speed"] * trav_time2
            dep_pos2 = max(0, safe_len - trav_dist2 - 3.0)
            dep_pos2 = float(np.clip(dep_pos2, 0.0, safe_len - 2.0))
            cfg.pedestrian2 = ActorBehavior(
                maneuver=man2, style=style2,
                route_edges=f"{'left_in' if 'left_right' in man2 else 'right_out'} {'right_out' if 'left_right' in man2 else 'left_in'}",
                depart_time=dep_time2,
                depart_pos=dep_pos2,
                ped_speed=sp2["speed"],
                stop_midway=sp2["stop_midway"],
                stop_duration=self.rng.uniform(2, 5) if sp2["stop_midway"] else 0,
                hesitant=sp2["hesitant"],
                color="0,0.7,0",
            )

            # Dense: third pedestrian at medium timing
            if dense:
                man3 = self.rng.choice(PED_MANEUVERS)
                style3 = self.rng.choice(ped_style_pool)
                sp3 = PED_STYLE_PARAMS[style3]
                mid_ego_eta = (ego_approach_dist - self.rng.uniform(5.0, 15.0)) / max(ego_approach_speed * 0.6, 1.0)
                dep_time3 = 1.0 + self.rng.uniform(0, 1.0)
                trav_time3 = max(0.5, mid_ego_eta - dep_time3)
                trav_dist3 = sp3["speed"] * trav_time3
                dep_pos3 = max(0, safe_len - trav_dist3 - 3.0)
                dep_pos3 = float(np.clip(dep_pos3, 0.0, safe_len - 2.0))
                cfg.pedestrian3 = ActorBehavior(
                    maneuver=man3, style=style3,
                    route_edges=f"{'left_in' if 'left_right' in man3 else 'right_out'} {'right_out' if 'left_right' in man3 else 'left_in'}",
                    depart_time=dep_time3,
                    depart_pos=dep_pos3,
                    ped_speed=sp3["speed"],
                    stop_midway=sp3["stop_midway"],
                    stop_duration=self.rng.uniform(2, 5) if sp3["stop_midway"] else 0,
                    hesitant=sp3["hesitant"],
                    color="0,0.5,0",
                )

        if has_moto:
            if self.rng.uniform() < 0.9 and conflict_routes.get("moto"):
                maneuver = self.rng.choice(conflict_routes["moto"])
            else:
                maneuver = self.rng.choice(MOTO_MANEUVERS)
            style = self.rng.choice(moto_style_pool)
            sp = MOTO_STYLE_PARAMS[style]
            depart, dep_pos = self._compute_conflict_spawn(
                ego_approach_dist=ego_approach_dist,
                ego_approach_speed=ego_approach_speed,
                agent_approach_dist=bar_len - 5.0,
                agent_speed=sp["max_speed"],
                bar_len=bar_len,
            )
            cfg.motorcycle = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=MOTO_ROUTES[maneuver],
                depart_time=depart,
                depart_pos=dep_pos,
                max_speed=sp["max_speed"] * (0.9 + self.rng.uniform(0, 0.2)),
                accel=sp["accel"],
                decel=sp["decel"],
                sigma=sp["sigma"],
                tau=sp["tau"],
                jm_ignore=jm_ignore_fixed if jm_ignore_fixed is not None else sp["jm"],
                color="1,0.5,0",
            )
            cfg.moto_intent_label = MOTO_INTENT_LABELS.get(maneuver, 1)
            cfg.moto_style_label = MOTO_STYLE_LABELS.get(style, 1)

            # ALWAYS spawn insurance motorcycle (moto2) on a DIFFERENT conflict
            # route than the primary. Without this exclusion, 50% of episodes
            # have both motos on the same route (convoy, not conflict) and the
            # interaction rate ceilings at ~53%.
            moto_conflict = conflict_routes.get("moto", [])
            if len(moto_conflict) > 1:
                man2_choices = [m for m in moto_conflict if m != maneuver]
                man2_m = self.rng.choice(man2_choices) if man2_choices else self.rng.choice(MOTO_MANEUVERS)
            else:
                man2_m = maneuver if moto_conflict else self.rng.choice(MOTO_MANEUVERS)
            style2_m = self.rng.choice(moto_style_pool)
            sp2_m = MOTO_STYLE_PARAMS[style2_m]
            dep2_m, dpos2_m = self._compute_insurance_spawn(
                ego_approach_dist=ego_approach_dist,
                ego_approach_speed=ego_approach_speed,
                agent_approach_dist=bar_len - 5.0,
                agent_speed=sp2_m["max_speed"],
                bar_len=bar_len,
            )
            cfg.motorcycle2 = ActorBehavior(
                maneuver=man2_m, style=style2_m,
                route_edges=MOTO_ROUTES[man2_m],
                depart_time=dep2_m,
                depart_pos=dpos2_m,
                max_speed=sp2_m["max_speed"] * (0.9 + self.rng.uniform(0, 0.2)),
                accel=sp2_m["accel"], decel=sp2_m["decel"],
                sigma=sp2_m["sigma"], tau=sp2_m["tau"],
                jm_ignore=jm_ignore_fixed if jm_ignore_fixed is not None else sp2_m["jm"],
                color="0.8,0.4,0",
            )

        if has_pothole:
            cx = self.rng.uniform(-2, 5)
            cy = self.rng.uniform(-3, 3)
            hw = self.rng.uniform(1.5, 3.0)
            hh = self.rng.uniform(1.0, 2.0)
            cfg.pothole = PotholeConfig(x=cx, y=cy, half_w=hw, half_h=hh)

        return cfg

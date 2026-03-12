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
    pedestrian: Optional[ActorBehavior] = None
    motorcycle: Optional[ActorBehavior] = None
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


class BehaviorSampler:
    """Sample diverse per-episode behavior configs."""

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def sample(
        self,
        has_car: bool,
        has_ped: bool,
        has_moto: bool,
        has_pothole: bool,
        bar_len: float = 160.0,
        jm_ignore_fixed: float | None = None,
    ) -> BehaviorConfig:
        cfg = BehaviorConfig()

        if has_car:
            maneuver = self.rng.choice(CAR_MANEUVERS)
            style = self.rng.choice(CAR_STYLES)
            sp = CAR_STYLE_PARAMS[style]
            depart = 3.0 + self.rng.uniform(0, 5)
            cfg.car = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=CAR_ROUTES[maneuver],
                depart_time=depart,
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

        if has_ped:
            maneuver = self.rng.choice(PED_MANEUVERS)
            style = self.rng.choice(PED_STYLES)
            sp = PED_STYLE_PARAMS[style]
            if maneuver == "cross_left_right":
                depart_pos = bar_len - self.rng.uniform(15, 35)
            else:
                depart_pos = bar_len - self.rng.uniform(15, 35)
            cfg.pedestrian = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=f"{'left_in' if 'left_right' in maneuver else 'right_out'} {'right_out' if 'left_right' in maneuver else 'left_in'}",
                depart_time=0.5 + self.rng.uniform(0, 3),
                depart_pos=depart_pos,
                ped_speed=sp["speed"],
                stop_midway=sp["stop_midway"],
                stop_duration=self.rng.uniform(2, 5) if sp["stop_midway"] else 0,
                hesitant=sp["hesitant"],
                color="0,0.9,0",
            )
            cfg.ped_intent_label = PED_INTENT_LABELS.get(maneuver, 1)
            cfg.ped_style_label = PED_STYLE_LABELS.get(style, 1)

        if has_moto:
            maneuver = self.rng.choice(MOTO_MANEUVERS)
            style = self.rng.choice(MOTO_STYLES)
            sp = MOTO_STYLE_PARAMS[style]
            depart = 2.0 + self.rng.uniform(0, 6)
            cfg.motorcycle = ActorBehavior(
                maneuver=maneuver,
                style=style,
                route_edges=MOTO_ROUTES[maneuver],
                depart_time=depart,
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

        if has_pothole:
            # Random position within the junction conflict area
            cx = self.rng.uniform(-8, 8)
            cy = self.rng.uniform(-5, 5)
            hw = self.rng.uniform(1.5, 4.0)
            hh = self.rng.uniform(1.0, 2.5)
            cfg.pothole = PotholeConfig(x=cx, y=cy, half_w=hw, half_h=hh)

        return cfg

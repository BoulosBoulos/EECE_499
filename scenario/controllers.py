"""Reactive finite-state-machine controllers for non-ego actors.

Each controller observes the ego vehicle's state (via TraCI) and decides
its own speed/brake behaviour so the interaction feels realistic:
  - Cars and motorcycles can yield, commit, or violate expectations.
  - Pedestrians can wait, commit, hesitate, or retreat.

These are *not* full RL agents; they are interpretable reactive policies
that create genuine decision tension for the ego's RL framework.
"""

from __future__ import annotations

import enum
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import traci
except ImportError:
    traci = None


# ═════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_vehicle_state(vid: str) -> Optional[dict]:
    """Safely read a vehicle's position, speed, accel from TraCI."""
    try:
        if vid not in traci.vehicle.getIDList():
            return None
        pos = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        accel = traci.vehicle.getAcceleration(vid)
        edge = traci.vehicle.getRoadID(vid)
        lane_pos = traci.vehicle.getLanePosition(vid)
        lane_len = traci.lane.getLength(traci.vehicle.getLaneID(vid))
        return {
            "pos": np.array(pos, dtype=float),
            "speed": speed, "accel": accel,
            "edge": edge, "lane_pos": lane_pos, "lane_len": lane_len,
            "remaining": max(0.0, lane_len - lane_pos),
        }
    except Exception:
        return None


def _get_person_state(pid: str) -> Optional[dict]:
    """Safely read a pedestrian's position and speed from TraCI."""
    try:
        if pid not in traci.person.getIDList():
            return None
        pos = traci.person.getPosition(pid)
        speed = traci.person.getSpeed(pid)
        return {"pos": np.array(pos, dtype=float), "speed": speed}
    except Exception:
        return None


def _ego_eta_to_conflict(ego_state: dict, stem_len: float) -> float:
    """Estimate ego ETA to the junction conflict zone."""
    remaining = ego_state.get("remaining", 0.0)
    v = max(ego_state.get("speed", 0.0), 0.1)
    return remaining / v


def _ego_is_braking(ego_state: dict) -> bool:
    return ego_state.get("accel", 0.0) < -0.5


def _ego_is_committed(ego_state: dict) -> bool:
    """Ego has entered junction internal edges or right_out."""
    edge = ego_state.get("edge", "")
    return "center" in edge.lower() or "right_out" in edge


# ═════════════════════════════════════════════════════════════════════════════
#  Vehicle / Motorcycle controller
# ═════════════════════════════════════════════════════════════════════════════

class VehState(enum.Enum):
    APPROACH = "approach"
    ASSESS_GAP = "assess_gap"
    YIELDING = "yielding"
    COMMITTED = "committed"
    CLEARING = "clearing"
    DONE = "done"


@dataclass
class VehicleController:
    """Reactive FSM controller for a car or motorcycle actor."""

    vid: str
    approach_speed: float = 11.11
    gap_accept_s: float = 3.0
    assertiveness: float = 0.5
    intent: str = "proceed"           # "yield" | "proceed" | "violate"
    yield_decel: float = 3.0
    commit_speed_factor: float = 1.0
    assess_dist_m: float = 20.0
    stem_len: float = 50.0

    state: VehState = VehState.APPROACH
    _ticks_yielding: int = 0
    _max_yield_ticks: int = 80        # give up yielding after ~8s

    def step(self, ego_id: str = "ego", dt: float = 0.1) -> None:
        """Execute one controller tick (called every sim step)."""
        me = _get_vehicle_state(self.vid)
        if me is None:
            self.state = VehState.DONE
            return

        ego = _get_vehicle_state(ego_id)
        if ego is None:
            self._set_speed(self.approach_speed)
            return

        if self.state == VehState.APPROACH:
            self._handle_approach(me, ego)
        elif self.state == VehState.ASSESS_GAP:
            self._handle_assess(me, ego)
        elif self.state == VehState.YIELDING:
            self._handle_yielding(me, ego, dt)
        elif self.state == VehState.COMMITTED:
            self._handle_committed(me, ego)
        elif self.state == VehState.CLEARING:
            self._handle_clearing(me)

    # ── State handlers ──────────────────────────────────────────────────────

    def _handle_approach(self, me: dict, ego: dict) -> None:
        self._set_speed(self.approach_speed)
        if me["remaining"] < self.assess_dist_m:
            self.state = VehState.ASSESS_GAP

    def _handle_assess(self, me: dict, ego: dict) -> None:
        ego_eta = _ego_eta_to_conflict(ego, self.stem_len)
        my_eta = me["remaining"] / max(me["speed"], 0.5)

        if self.intent == "yield":
            self.state = VehState.YIELDING
            self._ticks_yielding = 0
            return

        if self.intent == "violate":
            self.state = VehState.COMMITTED
            self._set_speed(self.approach_speed * 1.1)
            return

        # Gap acceptance: if ego is far enough, proceed; else yield
        delta_eta = ego_eta - my_eta  # positive = I arrive first
        threshold = self.gap_accept_s * (1.0 - 0.4 * self.assertiveness)

        if _ego_is_braking(ego) and ego_eta > threshold:
            self.state = VehState.COMMITTED
            return

        if delta_eta > threshold:
            self.state = VehState.COMMITTED
            return

        if _ego_is_committed(ego):
            self.state = VehState.YIELDING
            self._ticks_yielding = 0
            return

        # Ambiguous: assertive actors commit, others yield
        if self.assertiveness > 0.6:
            self.state = VehState.COMMITTED
        else:
            self.state = VehState.YIELDING
            self._ticks_yielding = 0

    def _handle_yielding(self, me: dict, ego: dict, dt: float) -> None:
        self._ticks_yielding += 1
        target_v = max(0.0, me["speed"] - self.yield_decel * dt)
        self._set_speed(target_v)

        # Resume if ego has passed or ego is far behind
        if _ego_is_committed(ego) and ego.get("remaining", 0) < 5:
            self.state = VehState.COMMITTED
            return
        ego_eta = _ego_eta_to_conflict(ego, self.stem_len)
        if ego_eta < 0.5:
            pass
        elif self._ticks_yielding > self._max_yield_ticks:
            self.state = VehState.COMMITTED

    def _handle_committed(self, me: dict, ego: dict) -> None:
        target = self.approach_speed * self.commit_speed_factor
        self._set_speed(target)
        edge = me.get("edge", "")
        if "out" in edge and me["lane_pos"] > 5:
            self.state = VehState.CLEARING

    def _handle_clearing(self, me: dict) -> None:
        self._set_speed(self.approach_speed)

    # ── Actuator ────────────────────────────────────────────────────────────

    def _set_speed(self, v: float) -> None:
        try:
            if self.vid in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(self.vid, max(0.0, v))
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Pedestrian controller
# ═════════════════════════════════════════════════════════════════════════════

class PedState(enum.Enum):
    AT_CURB = "at_curb"
    PREPARE = "prepare"
    COMMIT = "commit"
    CROSSING = "crossing"
    PAUSE_MIDWAY = "pause_midway"
    RETREAT = "retreat"
    CLEAR = "clear"


@dataclass
class PedestrianController:
    """Reactive FSM controller for a pedestrian."""

    pid: str = "ped0"
    walk_speed: float = 1.2
    gap_accept_s: float = 4.0
    assertiveness: float = 0.4
    intent: str = "proceed"          # "yield" | "proceed" | "hesitate"
    hesitant: bool = False
    stop_midway: bool = False
    stop_duration_s: float = 2.0
    stem_len: float = 50.0
    junction_x: float = 50.0        # junction center x in SUMO coords
    junction_y: float = 50.0

    state: PedState = PedState.AT_CURB
    _ticks_waiting: int = 0
    _ticks_pause: int = 0
    _pause_target: int = 0
    _hesitant_phase: int = 0
    _hesitant_wait: int = 0
    _committed: bool = False

    def step(self, ego_id: str = "ego", dt: float = 0.1) -> None:
        """One controller tick."""
        me = _get_person_state(self.pid)
        if me is None:
            self.state = PedState.CLEAR
            return

        ego = _get_vehicle_state(ego_id)

        if self.state == PedState.AT_CURB:
            self._handle_at_curb(me, ego, dt)
        elif self.state == PedState.PREPARE:
            self._handle_prepare(me, ego, dt)
        elif self.state == PedState.COMMIT:
            self._handle_commit(me, ego)
        elif self.state == PedState.CROSSING:
            self._handle_crossing(me, ego, dt)
        elif self.state == PedState.PAUSE_MIDWAY:
            self._handle_pause(me, dt)
        elif self.state == PedState.RETREAT:
            self._handle_retreat(me, dt)

    @property
    def is_committed(self) -> bool:
        return self._committed

    def _near_crosswalk(self, me: dict) -> bool:
        dist = np.linalg.norm(me["pos"] - np.array([self.junction_x, self.junction_y]))
        return dist < 15.0

    def _on_crosswalk(self, me: dict) -> bool:
        dist = np.linalg.norm(me["pos"] - np.array([self.junction_x, self.junction_y]))
        return dist < 10.0

    # ── State handlers ──────────────────────────────────────────────────────

    def _handle_at_curb(self, me: dict, ego: Optional[dict], dt: float) -> None:
        self._set_speed(self.walk_speed)
        if self._near_crosswalk(me):
            self.state = PedState.PREPARE
            self._ticks_waiting = 0

    def _handle_prepare(self, me: dict, ego: Optional[dict], dt: float) -> None:
        self._ticks_waiting += 1

        if self.intent == "yield" and ego is not None:
            ego_eta = _ego_eta_to_conflict(ego, self.stem_len)
            if ego_eta > self.gap_accept_s or _ego_is_committed(ego):
                self._set_speed(0.0)
                return
            elif ego_eta < 1.0:
                self._set_speed(0.0)
                return
            else:
                self.state = PedState.COMMIT
                self._committed = True
                return

        if self.hesitant:
            self._handle_hesitant_prepare(me, ego, dt)
            return

        if ego is None:
            self.state = PedState.COMMIT
            self._committed = True
            return

        ego_eta = _ego_eta_to_conflict(ego, self.stem_len)
        threshold = self.gap_accept_s * (1.0 - 0.3 * self.assertiveness)

        if _ego_is_braking(ego) and ego_eta > 1.5:
            self.state = PedState.COMMIT
            self._committed = True
            return

        if ego_eta > threshold:
            self.state = PedState.COMMIT
            self._committed = True
            return

        if self.assertiveness > 0.7:
            self.state = PedState.COMMIT
            self._committed = True
            return

        self._set_speed(0.0)
        if self._ticks_waiting > 80:
            self.state = PedState.COMMIT
            self._committed = True

    def _handle_hesitant_prepare(
        self, me: dict, ego: Optional[dict], dt: float,
    ) -> None:
        if self._hesitant_phase == 0:
            self._set_speed(0.0)
            self._hesitant_wait += 1
            if self._hesitant_wait > int(1.5 / max(dt, 0.01)):
                self._hesitant_phase = 1
                self._hesitant_wait = 0
        elif self._hesitant_phase == 1:
            self._set_speed(self.walk_speed * 0.4)
            self._hesitant_wait += 1
            if self._hesitant_wait > int(1.0 / max(dt, 0.01)):
                self._hesitant_phase = 2
                self._hesitant_wait = 0
        elif self._hesitant_phase == 2:
            self._set_speed(0.0)
            self._hesitant_wait += 1
            if self._hesitant_wait > int(1.0 / max(dt, 0.01)):
                self.state = PedState.COMMIT
                self._committed = True

    def _handle_commit(self, me: dict, ego: Optional[dict]) -> None:
        self._set_speed(self.walk_speed)
        self._committed = True
        if self._on_crosswalk(me):
            self.state = PedState.CROSSING

    def _handle_crossing(self, me: dict, ego: Optional[dict], dt: float) -> None:
        self._set_speed(self.walk_speed)

        if self.stop_midway:
            dist = np.linalg.norm(
                me["pos"] - np.array([self.junction_x, self.junction_y])
            )
            if dist < 4.0 and self._ticks_pause == 0:
                self.state = PedState.PAUSE_MIDWAY
                self._ticks_pause = 1
                self._pause_target = int(self.stop_duration_s / max(dt, 0.01))
                self._set_speed(0.0)
                return

        if not self._on_crosswalk(me) and self._committed:
            self.state = PedState.CLEAR

    def _handle_pause(self, me: dict, dt: float) -> None:
        self._set_speed(0.0)
        self._ticks_pause += 1
        if self._ticks_pause >= self._pause_target:
            self.state = PedState.CROSSING
            self._set_speed(self.walk_speed)

    def _handle_retreat(self, me: dict, dt: float) -> None:
        self._set_speed(-self.walk_speed * 0.5)
        self._ticks_waiting += 1
        if self._ticks_waiting > 20:
            self.state = PedState.PREPARE
            self._ticks_waiting = 0

    # ── Actuator ────────────────────────────────────────────────────────────

    def _set_speed(self, v: float) -> None:
        try:
            if self.pid in traci.person.getIDList():
                traci.person.setSpeed(self.pid, max(0.0, v))
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Factory
# ═════════════════════════════════════════════════════════════════════════════

def make_vehicle_controller(
    vid: str,
    approach_speed: float,
    gap_accept_s: float,
    assertiveness: float,
    intent: str,
    stem_len: float = 50.0,
) -> VehicleController:
    return VehicleController(
        vid=vid,
        approach_speed=approach_speed,
        gap_accept_s=gap_accept_s,
        assertiveness=assertiveness,
        intent=intent,
        stem_len=stem_len,
    )


def make_pedestrian_controller(
    pid: str,
    walk_speed: float,
    gap_accept_s: float,
    assertiveness: float,
    intent: str,
    hesitant: bool = False,
    stop_midway: bool = False,
    stop_duration_s: float = 2.0,
    stem_len: float = 50.0,
    junction_x: float = 50.0,
    junction_y: float = 50.0,
) -> PedestrianController:
    return PedestrianController(
        pid=pid,
        walk_speed=walk_speed,
        gap_accept_s=gap_accept_s,
        assertiveness=assertiveness,
        intent=intent,
        hesitant=hesitant,
        stop_midway=stop_midway,
        stop_duration_s=stop_duration_s,
        stem_len=stem_len,
        junction_x=junction_x,
        junction_y=junction_y,
    )

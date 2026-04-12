"""Interaction-benchmark Gymnasium environment (interaction_v2).

This environment is purpose-built for testing **behavioural decision
making** at a T-intersection.  Key differences from the legacy SumoEnv:

  * Template-driven episode generation via TemplateSampler / Scheduler.
  * Pre-roll phase before RL takes control.
  * Action latching – each RL step spans 0.4–0.8 s of simulation.
  * Reactive non-ego actor controllers (FSM-based).
  * Conflict-centric state via InteractionStateBuilder.
  * Explicit event detectors for ROW violations, conflict intrusions,
    forced braking, deadlocks, and crosswalk blocking.
  * Rich reward aligned with the behavioural-decision research claim.
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

try:
    import traci
except ImportError:
    traci = None

try:
    import yaml
except ImportError:
    yaml = None

from state.builder_interaction import InteractionStateBuilder
from scenario.conflict_map import (
    CONFLICT_ZONES, ROUTE_CONFLICTS, SCENARIO_CONFLICT_ROUTES,
    ego_dist_to_cz, actor_dist_to_cz, CZ_EXIT_CROSSWALK,
    PRIO_ACTOR, PRIO_EGO,
)
from scenario.template_sampler import TemplateSampler, EpisodeTemplate, ActorSpec
from scenario.scheduler import solve_spawn_for_eta, solve_ego_preroll, solve_ped_spawn
from scenario.controllers import (
    VehicleController, PedestrianController,
    make_vehicle_controller, make_pedestrian_controller,
    _get_vehicle_state, _get_person_state,
)
from scenario.generator_v2 import InteractionScenarioGenerator, SCENARIO_SPEC_V2


ACTION_NAMES = ["STOP", "CREEP", "YIELD", "GO", "ABORT"]
N_ACTIONS = 5

# ── Config helpers ──────────────────────────────────────────────────────────

def _load_yaml(path: str | None) -> dict:
    if path is None or yaml is None:
        return {}
    try:
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, path)
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _make_gym():
    return gym.Env if gym else object


# ═════════════════════════════════════════════════════════════════════════════

class InteractionEnv(_make_gym()):
    """Behavioural-decision T-intersection benchmark environment."""

    EGO_ID = "ego"
    SCENARIO_TYPES = list(SCENARIO_SPEC_V2.keys())

    def __init__(
        self,
        scenario_name: str = "1a",
        use_gui: bool = False,
        config_path: str | None = "configs/interaction/benchmark.yaml",
        scenario_dir: str | None = None,
        seed: int | None = None,
    ):
        if gym:
            super().__init__()

        self.cfg = _load_yaml(config_path)
        self.use_gui = use_gui
        self.scenario_name = scenario_name if scenario_name in self.SCENARIO_TYPES else "1a"

        spec = SCENARIO_SPEC_V2.get(self.scenario_name, (True, False, False, False))
        self._has_car, self._has_ped, self._has_moto, self._has_pothole = spec

        # Geometry
        self._stem_len = float(self.cfg.get("stem_length", 50.0))
        self._bar_len = float(self.cfg.get("bar_half_length", 50.0))

        # Timing
        self._sim_dt = float(self.cfg.get("sim_step_s", 0.1))
        self._latch_s = float(self.cfg.get("action_latch_s", 0.5))
        self._latch_ticks = max(1, int(round(self._latch_s / self._sim_dt)))
        self._max_steps = int(self.cfg.get("max_decision_steps", 60))
        self._preroll_max = float(self.cfg.get("pre_roll_max_s", 8.0))
        self._preroll_ego_speed = float(self.cfg.get("pre_roll_ego_speed_mps", 8.0))
        self._decision_zone_m = float(self.cfg.get("decision_zone_m", 18.0))
        self._decision_eta_band = float(self.cfg.get("decision_eta_band_s", 3.0))
        self._deadlock_s = float(self.cfg.get("deadlock_timeout_s", 15.0))

        # Ego controllers
        ec = self.cfg.get("ego_controllers", {})
        self._stop_decel = float(ec.get("stop_decel_mps2", 4.0))
        self._creep_target = float(ec.get("creep_target_mps", 1.0))
        self._creep_accel = float(ec.get("creep_accel_mps2", 0.8))
        self._yield_decel = float(ec.get("yield_decel_mps2", 1.5))
        self._go_accel = float(ec.get("go_accel_mps2", 2.0))
        self._go_max_speed = float(ec.get("go_max_speed_mps", 11.11))
        self._abort_decel = float(ec.get("abort_decel_mps2", 6.0))

        # Reward
        rc = self.cfg.get("reward", {})
        self._r_success = float(rc.get("r_success", 25.0))
        self._r_collision = float(rc.get("r_collision", -100.0))
        self._r_row_violation = float(rc.get("r_row_violation", -20.0))
        self._r_conflict_intrusion = float(rc.get("r_conflict_intrusion", -15.0))
        self._r_forced_brake = float(rc.get("r_forced_other_brake", -8.0))
        self._r_deadlock = float(rc.get("r_deadlock", -10.0))
        self._r_crosswalk_block = float(rc.get("r_crosswalk_block", -12.0))
        self._r_progress = float(rc.get("r_progress", 0.2))
        self._r_time = float(rc.get("r_time", -0.03))
        self._r_comfort = float(rc.get("r_comfort", -0.01))
        self._r_unnecessary_wait = float(rc.get("r_unnecessary_wait", -0.5))
        self._r_pothole = float(rc.get("r_pothole", -5.0))

        # Detection thresholds
        dc = self.cfg.get("detection", {})
        self._coll_dist = float(dc.get("collision_dist_m", 2.5))
        self._row_eta_thr = float(dc.get("row_violation_eta_s", 1.5))
        self._cz_occ_dist = float(dc.get("conflict_occupied_dist_m", 8.0))
        self._forced_brake_thr = float(dc.get("forced_brake_threshold_mps2", 4.0))
        self._safe_gap_eta = float(dc.get("safe_gap_eta_s", 3.0))
        self._cw_zone_m = float(dc.get("crosswalk_zone_m", 5.0))

        # Scenario / state
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scenario_dir = scenario_dir or os.path.join(
            base, "scenarios", f"interaction_{self.scenario_name}",
        )
        self._state_builder = InteractionStateBuilder(top_n=3)
        self._state_dim = self._state_builder.state_dim
        if self._has_pothole:
            self._state_dim += 1

        if spaces:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(self._state_dim,), dtype=np.float32,
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

        # Samplers
        tp = self.cfg.get("template_probs")
        eb = self.cfg.get("eta_bands")
        self._rng = np.random.RandomState(seed)
        self._template_sampler = TemplateSampler(
            template_probs=tp, eta_bands=eb, rng=self._rng,
        )

        # Runtime state
        self._step_count = 0
        self._sim_time = 0.0
        self._phase = "approach"
        self._template: Optional[EpisodeTemplate] = None
        self._actor_controllers: List = []
        self._prev_ego_a = 0.0
        self._prev_ego_speed = 0.0
        self._prev_ego_psi = 0.0
        self._ego_progress = 0.0
        self._actor_first_seen: Dict[str, int] = {}
        self._pothole_box = np.array([[-4, 4], [-2, 2]], dtype=float)
        self._junction_xy = np.array([self._bar_len, self._stem_len], dtype=float)
        self._events: Dict[str, bool] = {}
        self._event_log_path: Optional[str] = None
        self._event_log_file = None
        self._episode_id = 0

    # ════════════════════════════════════════════════════════════════════════
    #  JSONL event trace logging
    # ════════════════════════════════════════════════════════════════════════

    def enable_event_logging(self, log_dir: str) -> None:
        """Enable JSONL event logging to *log_dir*/interaction_events.jsonl."""
        os.makedirs(log_dir, exist_ok=True)
        self._event_log_path = os.path.join(log_dir, "interaction_events.jsonl")

    def _log_event(self, event_type: str, data: dict) -> None:
        if self._event_log_path is None:
            return
        record = {
            "episode": self._episode_id,
            "step": self._step_count,
            "sim_time": round(self._sim_time, 3),
            "event_type": event_type,
            "scenario": self.scenario_name,
            "template": self._template.template_family if self._template else "",
            **data,
        }
        try:
            with open(self._event_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    #  Gym interface
    # ════════════════════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._template_sampler.rng = self._rng

        self._close_sumo()
        self._ensure_scenario()
        self._start_sumo()

        self._episode_id += 1
        self._step_count = 0
        self._sim_time = 0.0
        self._phase = "approach"
        self._prev_ego_a = 0.0
        self._prev_ego_speed = 0.0
        self._prev_ego_psi = 0.0
        self._ego_progress = 0.0
        self._actor_first_seen = {}
        self._events = {}
        self._actor_controllers = []

        self._template = self._template_sampler.sample(
            scenario_id=self.scenario_name,
            has_pothole=self._has_pothole,
            bar_len=self._bar_len,
            stem_len=self._stem_len,
        )

        self._spawn_ego()
        self._spawn_actors(self._template)
        self._pre_roll()

        self._log_event("onset", {
            "template_family": self._template.template_family,
            "n_actors": len(self._template.actors),
            "ego_target_eta": self._template.ego_target_eta_enter,
        })

        obs = self._build_obs()
        info = self._build_info(reward=0.0, events={})
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        prev_phase = self._phase

        for _ in range(self._latch_ticks):
            self._apply_ego_action(action)
            self._tick_actor_controllers()
            traci.simulationStep()
            self._sim_time += self._sim_dt

        self._update_phase()
        events = self._detect_events()
        reward = self._compute_reward(action, events)
        terminated, truncated = self._check_termination(events)

        self._log_event("latch_change", {
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < N_ACTIONS else "?",
            "from_phase": prev_phase,
            "to_phase": self._phase,
            "reward": round(reward, 4),
        })

        active_events = {k: v for k, v in events.items() if v}
        if active_events:
            for ev_name in active_events:
                self._log_event(ev_name, {"action": action})

        if terminated or truncated:
            outcome = "success" if events.get("success") else (
                "collision" if events.get("collision") else (
                    "deadlock" if events.get("deadlock") else "truncated"))
            self._log_event("terminate", {
                "outcome_code": outcome,
                "total_steps": self._step_count,
                "total_return": round(self._ego_progress, 2),
            })

        obs = self._build_obs()
        info = self._build_info(reward, events)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_sumo()

    # ════════════════════════════════════════════════════════════════════════
    #  SUMO lifecycle
    # ════════════════════════════════════════════════════════════════════════

    def _ensure_scenario(self):
        cfg_path = os.path.join(self.scenario_dir, "t.sumocfg")
        if not os.path.isfile(cfg_path):
            gen = InteractionScenarioGenerator(
                stem_len=self._stem_len, bar_len=self._bar_len,
            )
            gen.generate(self.scenario_dir, scenario_name=self.scenario_name)

    def _start_sumo(self):
        sumocfg = os.path.join(self.scenario_dir, "t.sumocfg")
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            binary = os.path.join(sumo_home, "bin", binary)
        cmd = [
            binary, "-c", sumocfg,
            "--step-length", str(self._sim_dt),
            "--no-step-log", "true",
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--intermodal-collision.action", "warn",
        ]
        if self.use_gui:
            gui_path = os.path.join(self.scenario_dir, "t_gui.xml")
            if os.path.isfile(gui_path):
                cmd.extend(["--gui-settings-file", gui_path])
        traci.start(cmd)

    def _close_sumo(self):
        try:
            if traci:
                traci.close()
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    #  Spawning
    # ════════════════════════════════════════════════════════════════════════

    def _spawn_ego(self):
        try:
            existing = traci.route.getIDList()
            if "ego_route" not in existing:
                traci.route.add("ego_route", ["stem_in", "right_out"])
        except Exception:
            pass
        traci.vehicle.add(self.EGO_ID, "ego_route", depart="0", typeID="Car")

    def _spawn_actors(self, template: EpisodeTemplate):
        actor_id_counter = 0
        for spec in template.actors:
            if spec.actor_type == "ped":
                self._spawn_ped(spec)
            else:
                actor_id_counter += 1
                vid = f"actor_{actor_id_counter}"
                self._spawn_vehicle(vid, spec)

        if template.has_pothole:
            self._pothole_box = np.array([
                [template.pothole_x - 3.0, template.pothole_x + 3.0],
                [template.pothole_y - 1.5, template.pothole_y + 1.5],
            ])

    def _spawn_vehicle(self, vid: str, spec: ActorSpec):
        first_edge = spec.route_edges.split()[0]
        try:
            existing = traci.route.getIDList()
            if spec.route_name not in existing:
                traci.route.add(spec.route_name, spec.route_edges.split())
        except Exception:
            pass

        depart_time, depart_pos = solve_spawn_for_eta(
            target_eta_s=spec.target_eta_enter,
            approach_speed_mps=spec.approach_speed,
            edge_id=first_edge,
            bar_len=self._bar_len,
            stem_len=self._stem_len,
        )

        type_id = "Motorcycle" if spec.actor_type == "cyc" else "CarOther"
        try:
            traci.vehicle.add(
                vid, spec.route_name,
                depart=str(depart_time),
                typeID=type_id,
                departPos=f"{depart_pos:.1f}",
            )
            traci.vehicle.setSpeedMode(vid, 0)
        except Exception:
            pass

        ctrl = make_vehicle_controller(
            vid=vid,
            approach_speed=spec.approach_speed,
            gap_accept_s=spec.gap_accept_s,
            assertiveness=spec.assertiveness,
            intent=spec.intent,
            stem_len=self._stem_len,
        )
        self._actor_controllers.append(ctrl)

    def _spawn_ped(self, spec: ActorSpec):
        pid = "ped0"
        edges = spec.route_edges.split()
        from_edge = edges[0] if edges else "left_in"
        to_edge = edges[-1] if len(edges) > 1 else "right_out"

        depart_time, depart_pos = solve_ped_spawn(
            target_eta_s=spec.target_eta_enter,
            ped_speed_mps=spec.ped_speed,
            from_edge=from_edge,
            bar_len=self._bar_len,
        )

        try:
            edge_len = traci.lane.getLength(f"{from_edge}_0")
            safe_pos = float(np.clip(depart_pos, 0.0, max(0.0, edge_len - 1.0)))
        except Exception:
            safe_pos = 0.0

        try:
            traci.person.add(pid, from_edge, pos=safe_pos, depart=depart_time)
            traci.person.appendWalkingStage(pid, [from_edge, to_edge], arrivalPos=-1)
            traci.person.setSpeed(pid, spec.ped_speed)
        except Exception:
            try:
                traci.person.add(pid, from_edge, pos=0.0, depart=0.5)
                traci.person.appendWalkingStage(pid, [from_edge, to_edge], arrivalPos=-1)
                traci.person.setSpeed(pid, spec.ped_speed)
            except Exception:
                pass

        ctrl = make_pedestrian_controller(
            pid=pid,
            walk_speed=spec.ped_speed,
            gap_accept_s=spec.gap_accept_s,
            assertiveness=spec.assertiveness,
            intent=spec.intent,
            hesitant=spec.ped_hesitant,
            stop_midway=spec.ped_stop_midway,
            stop_duration_s=spec.ped_stop_duration,
            stem_len=self._stem_len,
            junction_x=self._junction_xy[0],
            junction_y=self._junction_xy[1],
        )
        self._actor_controllers.append(ctrl)

    # ════════════════════════════════════════════════════════════════════════
    #  Pre-roll
    # ════════════════════════════════════════════════════════════════════════

    def _pre_roll(self):
        max_ticks = int(self._preroll_max / self._sim_dt)
        for _ in range(max_ticks):
            if self.EGO_ID in traci.vehicle.getIDList():
                traci.vehicle.setSpeedMode(self.EGO_ID, 0)
                traci.vehicle.setSpeed(self.EGO_ID, self._preroll_ego_speed)

            self._tick_actor_controllers()
            traci.simulationStep()
            self._sim_time += self._sim_dt

            if self._decision_onset_met():
                break

        if self.EGO_ID in traci.vehicle.getIDList():
            traci.vehicle.setSpeedMode(self.EGO_ID, 0)

        self._phase = "decision"

    def _decision_onset_met(self) -> bool:
        """Dual condition: ego within decision zone AND at least one actor
        has an ETA overlap within the configured band."""
        ego = _get_vehicle_state(self.EGO_ID)
        if ego is None:
            return True
        d_to_cz = ego.get("remaining", 999)
        if d_to_cz > self._decision_zone_m:
            return False
        # Check actor ETA overlap
        ego_eta = d_to_cz / max(ego.get("speed", 0.5), 0.5)
        for ctrl in self._actor_controllers:
            actor_eta = self._estimate_actor_eta(ctrl)
            if actor_eta is not None:
                if abs(actor_eta - ego_eta) <= self._decision_eta_band:
                    return True
        # If no actors at all (e.g. scenario 1d), just use distance
        if not self._actor_controllers:
            return True
        # If actors exist but none within band yet, keep pre-rolling
        # unless we've hit the max pre-roll time
        if self._sim_time >= self._preroll_max * 0.9:
            return True
        return False

    # ════════════════════════════════════════════════════════════════════════
    #  Ego action application (low-level controllers)
    # ════════════════════════════════════════════════════════════════════════

    def _apply_ego_action(self, action: int):
        if self.EGO_ID not in traci.vehicle.getIDList():
            return
        v = traci.vehicle.getSpeed(self.EGO_ID)
        dt = self._sim_dt

        if action == 0:     # STOP
            new_v = max(0.0, v - self._stop_decel * dt)
            traci.vehicle.slowDown(self.EGO_ID, new_v, dt)
        elif action == 1:   # CREEP
            if v < self._creep_target:
                new_v = min(self._creep_target, v + self._creep_accel * dt)
            else:
                new_v = max(self._creep_target, v - 1.0 * dt)
            traci.vehicle.slowDown(self.EGO_ID, max(0.0, new_v), dt)
        elif action == 2:   # YIELD
            new_v = max(0.0, v - self._yield_decel * dt)
            traci.vehicle.slowDown(self.EGO_ID, new_v, dt)
        elif action == 3:   # GO
            new_v = min(self._go_max_speed, v + self._go_accel * dt)
            traci.vehicle.setSpeed(self.EGO_ID, new_v)
        else:               # ABORT
            new_v = max(0.0, v - self._abort_decel * dt)
            traci.vehicle.slowDown(self.EGO_ID, new_v, dt)

    def _tick_actor_controllers(self):
        for ctrl in self._actor_controllers:
            ctrl.step(ego_id=self.EGO_ID, dt=self._sim_dt)

    # ════════════════════════════════════════════════════════════════════════
    #  Phase tracking
    # ════════════════════════════════════════════════════════════════════════

    def _update_phase(self):
        ego = _get_vehicle_state(self.EGO_ID)
        if ego is None:
            if self._phase not in ("clearing", "aborted"):
                self._phase = "clearing"
            return

        edge = ego.get("edge", "")
        if "right_out" in edge and ego.get("lane_pos", 0) > 10:
            self._phase = "clearing"
        elif ":center" in edge.lower() or "right_out" in edge:
            self._phase = "committed"
        elif self._phase == "approach":
            if ego.get("remaining", 999) < self._decision_zone_m:
                self._phase = "decision"

    # ════════════════════════════════════════════════════════════════════════
    #  Event detection
    # ════════════════════════════════════════════════════════════════════════

    def _detect_events(self) -> Dict[str, bool]:
        events: Dict[str, bool] = {
            "collision": False,
            "row_violation": False,
            "conflict_intrusion": False,
            "forced_other_brake": False,
            "crosswalk_block": False,
            "deadlock": False,
            "success": False,
            "unnecessary_wait": False,
            "pothole_hit": False,
        }

        ego = _get_vehicle_state(self.EGO_ID)
        ego_pos = ego["pos"] if ego else None

        # ── Collision ───────────────────────────────────────────────────────
        try:
            colliding = traci.simulation.getCollidingVehiclesIDList()
            if self.EGO_ID in colliding:
                events["collision"] = True
        except Exception:
            pass

        if ego_pos is not None:
            for ctrl in self._actor_controllers:
                a_state = None
                if isinstance(ctrl, VehicleController):
                    a_state = _get_vehicle_state(ctrl.vid)
                elif isinstance(ctrl, PedestrianController):
                    a_state = _get_person_state(ctrl.pid)
                if a_state and np.linalg.norm(a_state["pos"] - ego_pos) < self._coll_dist:
                    events["collision"] = True

        # ── ROW violation ───────────────────────────────────────────────────
        if ego is not None and self._phase == "committed":
            for spec in (self._template.actors if self._template else []):
                if spec.legal_priority <= 0.3:  # actor has priority
                    for ctrl in self._actor_controllers:
                        actor_eta = self._estimate_actor_eta(ctrl)
                        if actor_eta is not None and actor_eta < self._row_eta_thr:
                            ac_state = None
                            if isinstance(ctrl, VehicleController):
                                ac_state = _get_vehicle_state(ctrl.vid)
                            elif isinstance(ctrl, PedestrianController):
                                ac_state = _get_person_state(ctrl.pid)
                            if ac_state is not None:
                                d = np.linalg.norm(ac_state["pos"] - ego_pos)
                                if d < self._cz_occ_dist:
                                    events["row_violation"] = True

        # ── Conflict intrusion ──────────────────────────────────────────────
        if ego is not None and self._phase == "committed":
            for ctrl in self._actor_controllers:
                if isinstance(ctrl, PedestrianController) and ctrl.is_committed:
                    ps = _get_person_state(ctrl.pid)
                    if ps is not None:
                        d = np.linalg.norm(ps["pos"] - ego_pos)
                        if d < self._cz_occ_dist:
                            events["conflict_intrusion"] = True
                if isinstance(ctrl, VehicleController):
                    if ctrl.state.value == "committed":
                        vs = _get_vehicle_state(ctrl.vid)
                        if vs is not None:
                            d = np.linalg.norm(vs["pos"] - ego_pos)
                            if d < self._cz_occ_dist * 0.8:
                                events["conflict_intrusion"] = True

        # ── Forced other brake ──────────────────────────────────────────────
        for ctrl in self._actor_controllers:
            if isinstance(ctrl, VehicleController):
                vs = _get_vehicle_state(ctrl.vid)
                if vs and vs["accel"] < -self._forced_brake_thr:
                    events["forced_other_brake"] = True

        # ── Crosswalk blocking ──────────────────────────────────────────────
        if ego is not None:
            cw_center = self._junction_xy + np.array([10.0, 0.0])
            if np.linalg.norm(ego_pos - cw_center) < self._cw_zone_m:
                v = ego.get("speed", 0)
                if v < 0.3 and self._phase == "committed":
                    for ctrl in self._actor_controllers:
                        if isinstance(ctrl, PedestrianController) and ctrl.is_committed:
                            events["crosswalk_block"] = True

        # ── Deadlock ────────────────────────────────────────────────────────
        if self._sim_time > self._deadlock_s and self._phase in ("decision", "approach"):
            events["deadlock"] = True

        # ── Success ─────────────────────────────────────────────────────────
        if ego is not None:
            edge = ego.get("edge", "")
            if "right_out" in edge and ego.get("lane_pos", 0) > 15.0:
                events["success"] = True
        if ego is None and self._phase == "clearing":
            events["success"] = True

        # ── Unnecessary wait ────────────────────────────────────────────────
        if ego is not None and ego.get("speed", 0) < 0.3:
            all_safe = True
            for ctrl in self._actor_controllers:
                eta = self._estimate_actor_eta(ctrl)
                if eta is not None and eta < self._safe_gap_eta:
                    all_safe = False
            if all_safe and self._phase == "decision" and len(self._actor_controllers) > 0:
                events["unnecessary_wait"] = True

        # ── Pothole ─────────────────────────────────────────────────────────
        if self._has_pothole and ego_pos is not None:
            if (self._pothole_box[0, 0] <= ego_pos[0] <= self._pothole_box[0, 1] and
                    self._pothole_box[1, 0] <= ego_pos[1] <= self._pothole_box[1, 1]):
                events["pothole_hit"] = True

        self._events = events
        return events

    def _estimate_actor_eta(self, ctrl) -> Optional[float]:
        """Rough ETA of an actor controller to the junction conflict zone."""
        if isinstance(ctrl, VehicleController):
            vs = _get_vehicle_state(ctrl.vid)
            if vs is None:
                return None
            return vs["remaining"] / max(vs["speed"], 0.5)
        if isinstance(ctrl, PedestrianController):
            ps = _get_person_state(ctrl.pid)
            if ps is None:
                return None
            d = np.linalg.norm(ps["pos"] - self._junction_xy)
            spd = max(ps["speed"], 0.3)
            return d / spd
        return None

    # ════════════════════════════════════════════════════════════════════════
    #  Reward
    # ════════════════════════════════════════════════════════════════════════

    def _compute_reward(self, action: int, events: Dict[str, bool]) -> float:
        r = self._r_time

        ego = _get_vehicle_state(self.EGO_ID)
        if ego is not None:
            prog = ego.get("speed", 0) * self._latch_s
            r += self._r_progress * prog
            self._ego_progress += prog

            jerk = abs(ego.get("speed", 0) - self._prev_ego_speed) / self._latch_s
            r += self._r_comfort * jerk ** 2
            self._prev_ego_speed = ego.get("speed", 0)

        if events.get("success"):
            r += self._r_success
        if events.get("collision"):
            r += self._r_collision
        if events.get("row_violation"):
            r += self._r_row_violation
        if events.get("conflict_intrusion"):
            r += self._r_conflict_intrusion
        if events.get("forced_other_brake"):
            r += self._r_forced_brake
        if events.get("deadlock"):
            r += self._r_deadlock
        if events.get("crosswalk_block"):
            r += self._r_crosswalk_block
        if events.get("unnecessary_wait"):
            r += self._r_unnecessary_wait
        if events.get("pothole_hit"):
            r += self._r_pothole

        return float(r)

    # ════════════════════════════════════════════════════════════════════════
    #  Termination
    # ════════════════════════════════════════════════════════════════════════

    def _check_termination(self, events: Dict[str, bool]) -> Tuple[bool, bool]:
        terminated = False
        if events.get("collision"):
            terminated = True
        if events.get("success"):
            terminated = True
        if events.get("deadlock"):
            terminated = True

        truncated = (not terminated) and (self._step_count >= self._max_steps)
        return terminated, truncated

    # ════════════════════════════════════════════════════════════════════════
    #  Observation
    # ════════════════════════════════════════════════════════════════════════

    def _build_obs(self) -> np.ndarray:
        ego_raw = self._get_ego_obs()
        actors_raw = self._get_actors_obs()
        obs_dict = {
            "ego": ego_raw,
            "phase": self._phase,
            "actors": actors_raw,
            "step": self._step_count,
            "dt": self._latch_s,
        }
        state = self._state_builder.build(obs_dict)

        if self._has_pothole:
            ego = _get_vehicle_state(self.EGO_ID)
            if ego is not None:
                cx = (self._pothole_box[0, 0] + self._pothole_box[0, 1]) / 2
                cy = (self._pothole_box[1, 0] + self._pothole_box[1, 1]) / 2
                d_ph = float(np.linalg.norm(ego["pos"] - np.array([cx, cy])))
            else:
                d_ph = 100.0
            state = np.concatenate([state, [d_ph]])

        return state.astype(np.float32)

    def _get_ego_obs(self) -> dict:
        ego = _get_vehicle_state(self.EGO_ID)
        if ego is None:
            return {
                "v": 0, "a": 0, "jerk": 0, "yaw_rate": 0,
                "d_preentry": 0, "d_conflict_entry": 0,
                "d_conflict_exit": 0, "ego_eta_enter": 0,
                "ego_eta_exit": 0, "psi": 0,
                "p": np.array([0.0, 0.0]),
            }
        v = ego["speed"]
        a = ego["accel"]
        jerk = (a - self._prev_ego_a) / max(self._latch_s, 0.01)
        self._prev_ego_a = a

        try:
            angle = traci.vehicle.getAngle(self.EGO_ID)
            psi = np.radians(angle)
        except Exception:
            psi = 0.0
        yaw_rate = (psi - self._prev_ego_psi) / max(self._latch_s, 0.01)
        self._prev_ego_psi = psi

        remaining = ego.get("remaining", 50.0)
        d_conflict_entry = remaining
        d_preentry = max(0.0, remaining - 8.0)
        d_conflict_exit = remaining + 15.0
        ego_eta_enter = remaining / max(v, 0.5)
        ego_eta_exit = d_conflict_exit / max(v, 0.5)

        return {
            "v": v, "a": a, "jerk": jerk, "yaw_rate": yaw_rate,
            "d_preentry": d_preentry, "d_conflict_entry": d_conflict_entry,
            "d_conflict_exit": d_conflict_exit,
            "ego_eta_enter": ego_eta_enter, "ego_eta_exit": ego_eta_exit,
            "psi": psi, "p": ego["pos"],
        }

    def _get_actors_obs(self) -> List[dict]:
        actors = []
        ego = _get_vehicle_state(self.EGO_ID)
        ego_eta = 5.0
        if ego:
            ego_eta = ego.get("remaining", 50) / max(ego["speed"], 0.5)

        for ctrl in self._actor_controllers:
            if isinstance(ctrl, VehicleController):
                vs = _get_vehicle_state(ctrl.vid)
                if vs is None:
                    continue
                eta_enter = vs["remaining"] / max(vs["speed"], 0.5)

                spec = self._find_spec_for_ctrl(ctrl)
                try:
                    angle = traci.vehicle.getAngle(ctrl.vid)
                    psi = np.radians(angle)
                except Exception:
                    psi = 0.0

                aid = ctrl.vid
                if aid not in self._actor_first_seen:
                    self._actor_first_seen[aid] = self._step_count

                actors.append({
                    "id": aid, "p": vs["pos"], "psi": psi,
                    "v": vs["speed"], "a": vs["accel"],
                    "actor_type": "cyc" if spec and spec.actor_type == "cyc" else "veh",
                    "eta_enter": eta_enter,
                    "eta_exit": eta_enter + 3.0,
                    "legal_priority": spec.legal_priority if spec else 0.5,
                    "committed": float(ctrl.state.value == "committed"),
                    "yielding": float(ctrl.state.value == "yielding"),
                    "crosswalk_progress": 0.0,
                    "relevant": 1.0,
                    "uncertainty": max(0, 1.0 - 0.1 * (self._step_count - self._actor_first_seen.get(aid, 0))),
                    "in_conflict_zone": float(vs.get("remaining", 99) < 5.0),
                    "first_seen_step": self._actor_first_seen.get(aid, 0),
                })

            elif isinstance(ctrl, PedestrianController):
                ps = _get_person_state(ctrl.pid)
                if ps is None:
                    continue
                d = np.linalg.norm(ps["pos"] - self._junction_xy)
                eta_enter = d / max(ps["speed"], 0.3)

                spec = self._find_spec_for_ctrl(ctrl)
                aid = ctrl.pid
                if aid not in self._actor_first_seen:
                    self._actor_first_seen[aid] = self._step_count

                cw_progress = max(0.0, 1.0 - d / 15.0) if ctrl.is_committed else 0.0

                actors.append({
                    "id": aid, "p": ps["pos"], "psi": 0.0,
                    "v": ps["speed"], "a": 0.0,
                    "actor_type": "ped",
                    "eta_enter": eta_enter,
                    "eta_exit": eta_enter + 5.0,
                    "legal_priority": spec.legal_priority if spec else 0.0,
                    "committed": float(ctrl.is_committed),
                    "yielding": float(ctrl.state.value in ("at_curb", "prepare") and not ctrl.is_committed),
                    "crosswalk_progress": cw_progress,
                    "relevant": 1.0,
                    "uncertainty": max(0, 1.0 - 0.1 * (self._step_count - self._actor_first_seen.get(aid, 0))),
                    "in_conflict_zone": float(d < 8.0),
                    "first_seen_step": self._actor_first_seen.get(aid, 0),
                })

        return actors

    def _find_spec_for_ctrl(self, ctrl) -> Optional[ActorSpec]:
        if self._template is None:
            return None
        for spec in self._template.actors:
            if isinstance(ctrl, PedestrianController) and spec.actor_type == "ped":
                return spec
            if isinstance(ctrl, VehicleController) and spec.actor_type != "ped":
                if spec.route_name in (ctrl.vid, f"actor_{ctrl.vid}"):
                    return spec
        # Fallback: return first matching type
        for spec in self._template.actors:
            if isinstance(ctrl, VehicleController) and spec.actor_type in ("veh", "cyc"):
                return spec
            if isinstance(ctrl, PedestrianController) and spec.actor_type == "ped":
                return spec
        return None

    def _build_info(self, reward: float, events: Dict[str, bool]) -> dict:
        info: Dict[str, Any] = {
            "reward": reward,
            "events": dict(events),
            "phase": self._phase,
            "step": self._step_count,
            "sim_time": self._sim_time,
            "ego_progress": self._ego_progress,
            "template_family": self._template.template_family if self._template else "",
            "scenario_id": self.scenario_name,
        }
        ego = _get_vehicle_state(self.EGO_ID)
        if ego:
            info["ego_speed"] = ego["speed"]
            info["ego_accel"] = float(ego.get("accel", 0.0))
            info["d_conflict_entry"] = float(ego.get("remaining", 50.0))
            info["ego_pos"] = ego["pos"].tolist()
            info["ego_edge"] = ego["edge"]
        actors_obs = self._get_actors_obs()
        info["actors"] = actors_obs
        info["n_actors"] = len(actors_obs)
        return info

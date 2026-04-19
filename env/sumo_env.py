"""SUMO T-intersection Gymnasium environment with TraCI.
Supports scenarios 1a-1d, 2, 3, 4. Full behavior diversity.
- Per-episode sampled maneuvers, styles, and timing via BehaviorSampler
- Random pothole placement
- Proper collision detection via SUMO events + proximity
- Intent features aligned with state-builder agent ordering
- GRU hidden state carryover support
- Real right-of-way negotiation via SUMO junction logic
- Dynamic visibility, path tracking, per-agent uncertainty
"""

from __future__ import annotations

import os
import math
import numpy as np
from typing import Any, Optional

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

from state.builder import StateBuilder, _rot2d, _wrap
from scenario.behavior_sampler import BehaviorSampler, BehaviorConfig

ACTION_NAMES = ["STOP", "CREEP", "YIELD", "GO", "ABORT"]
N_ACTIONS = 5

EGO_MANEUVERS = {
    "stem_right": {"route": "ego_stem_right", "start_edge": "stem_in", "exit_edge": "right_out"},
    "stem_left":  {"route": "ego_stem_left",  "start_edge": "stem_in", "exit_edge": "left_out"},
    "right_left": {"route": "ego_right_left", "start_edge": "right_in", "exit_edge": "left_out"},
    "right_stem": {"route": "ego_right_stem", "start_edge": "right_in", "exit_edge": "stem_out"},
    "left_right": {"route": "ego_left_right", "start_edge": "left_in", "exit_edge": "right_out"},
    "left_stem":  {"route": "ego_left_stem",  "start_edge": "left_in", "exit_edge": "stem_out"},
}

SCENARIOS_WITH_SIDEWALKS = frozenset({
    "1b", "2", "3", "4", "2_dense", "3_dense", "4_dense",
})
INNER_MARGIN_NO_SIDEWALK = 8.0
INNER_MARGIN_WITH_SIDEWALK = 11.0
OUTER_EXTENT = 30.0
FAR_EXTENT = 20.0

_DEFAULT_REWARD_CONFIG = {
    "w_prog": 1.0, "w_time": -0.1, "w_risk": -3.0, "w_coll": -20.0,
    "ttc_thr": 3.0, "d_coll": 2.0, "w_pothole": -5.0,
    "w_abort_comfort": -0.5,
    "w_success": 10.0, "w_switch": -0.05, "w_rule": -2.0,
}


def _load_reward_config(path: str | None) -> dict:
    if path is None or yaml is None:
        return dict(_DEFAULT_REWARD_CONFIG)
    try:
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, path)
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        out = dict(_DEFAULT_REWARD_CONFIG)
        for k in out:
            if k in cfg:
                out[k] = float(cfg[k])
        return out
    except Exception:
        return dict(_DEFAULT_REWARD_CONFIG)


def _make_gym():
    return gym.Env if gym else object


def _clamp_depart_pos(dep_pos: float, edge_id: str, fallback_len: float) -> str:
    """Clamp departure position to actual SUMO edge length, return as string."""
    try:
        edge_len = traci.lane.getLength(f"{edge_id}_0")
    except Exception:
        edge_len = fallback_len
    clamped = float(np.clip(dep_pos, 0.0, max(0.0, edge_len - 1.0)))
    return f"{clamped:.1f}"


class SumoEnv(_make_gym()):
    """SUMO T-intersection: ego turns right. Full behavior diversity."""

    EGO_ID = "ego"
    OTHER_ID = "other"
    SCENARIO_TYPES = ["1a", "1b", "1c", "1d", "2", "3", "4",
                      "2_dense", "3_dense", "4_dense"]

    def __init__(
        self,
        scenario_dir: str | None = None,
        scenario_name: str = "1a",
        ego_maneuver: str = "stem_right",
        use_gui: bool = False,
        state_config: str | None = None,
        reward_config: str | None = "configs/reward/default.yaml",
        max_steps: int = 500,
        dt: float = 0.1,
        use_intent: bool = False,
        jm_ignore_fixed: float | None = None,
        buildings: bool = True,
        style_filter: str | None = None,
        state_ablation: str | None = None,
    ):
        if gym:
            super().__init__()
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.dt = dt
        self.use_intent = use_intent
        self.state_builder = StateBuilder(state_config)
        self.reward_cfg = _load_reward_config(reward_config)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from scenario.generator import SCENARIO_SPEC
        self.scenario_name = scenario_name if scenario_name in self.SCENARIO_TYPES else "1a"
        spec = SCENARIO_SPEC.get(self.scenario_name, (True, False, False, False))
        self._has_car, self._has_ped, self._has_moto, self._has_pothole = spec
        self._dense = "dense" in self.scenario_name
        base_scenario = self.scenario_name.replace("_dense", "")
        self.scenario_dir = scenario_dir or os.path.join(base, "scenarios", f"sumo_{base_scenario}")

        if ego_maneuver not in EGO_MANEUVERS:
            raise ValueError(f"Unknown ego_maneuver '{ego_maneuver}'. Must be one of: {list(EGO_MANEUVERS.keys())}")
        self.ego_maneuver = ego_maneuver
        self._ego_route_id = EGO_MANEUVERS[ego_maneuver]["route"]
        self._ego_start_edge = EGO_MANEUVERS[ego_maneuver]["start_edge"]
        self._ego_exit_edge = EGO_MANEUVERS[ego_maneuver]["exit_edge"]

        self._stem_len, self._bar_len = 60.0, 50.0
        self._load_dims()

        self._state_dim = 6 + 12 + 6 + 5 * 22
        if use_intent:
            self._state_dim += 5 * 6
        if self._has_pothole:
            self._state_dim += 1
        if spaces:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32)
        else:
            self.observation_space = type("ObsSpace", (), {"shape": (self._state_dim,)})()
        self.action_space = spaces.Discrete(N_ACTIONS) if spaces else type("ActSpace", (), {"n": N_ACTIONS})

        self._sumo_proc = None
        self._step_count = 0
        self._prev_ego = None
        self._jm_ignore_fixed = jm_ignore_fixed
        self._agent_history: dict[str, list] = {}
        self._agent_first_seen: dict[str, int] = {}
        self._gru_hidden = None
        self._behavior: Optional[BehaviorConfig] = None
        self._behavior_sampler = BehaviorSampler()
        self._buildings_enabled = buildings
        self._occlusion_templates = self._build_occlusion_templates()
        self._occlusion_polygons = []  # built in reset() after SUMO starts
        self._junction_offset = np.array([0.0, 0.0])  # set properly in _init_occlusion_geometry()
        self._style_filter = style_filter
        self._state_ablation = state_ablation
        self._pothole_box = np.array([[-4, 4], [-2, 2]])
        self._collision_flag = False
        self._ped_stopped = False
        self._ped_stop_counter = 0
        self._ped_hesitant_phase = 0
        self._ped_hesitant_counter = 0

        self._intent_predictor = None
        if use_intent:
            try:
                from models.intent_style import IntentStylePredictor
                import torch
                self._intent_predictor = IntentStylePredictor(input_dim=9, hidden_dim=64).eval()
                self._intent_device = "cuda" if torch.cuda.is_available() else "cpu"
                self._intent_predictor.to(self._intent_device)
                ckpt = os.path.join(base, "results", "intent_model.pt")
                if os.path.isfile(ckpt):
                    data = torch.load(ckpt, map_location=self._intent_device)
                    self._intent_predictor.load_state_dict(data["model"])
            except Exception:
                self.use_intent = False

    def _load_dims(self):
        dims_path = os.path.join(self.scenario_dir, "scenario_dims.yaml")
        if os.path.isfile(dims_path) and yaml:
            try:
                with open(dims_path) as f:
                    d = yaml.safe_load(f) or {}
                self._stem_len = float(d.get("stem_length", self._stem_len))
                self._bar_len = float(d.get("bar_half_length", self._bar_len))
            except Exception:
                pass

    def _build_occlusion_templates(self):
        """Return occlusion polygon templates in the (0,0)-centered scenario frame.

        Four corner buildings (NW, NE, SW, SE) flanking the T-intersection.
        Inner margin is scenario-dependent: 8.0m for non-pedestrian scenarios,
        11.0m for pedestrian scenarios (clears road + sidewalk).
        """
        if not self._buildings_enabled:
            return []

        inner = (INNER_MARGIN_WITH_SIDEWALK
                 if self.scenario_name in SCENARIOS_WITH_SIDEWALKS
                 else INNER_MARGIN_NO_SIDEWALK)

        return [
            {"name": "building_NW", "corners": np.array([
                [-inner, inner], [-OUTER_EXTENT, inner], [-OUTER_EXTENT, FAR_EXTENT], [-inner, FAR_EXTENT],
            ])},
            {"name": "building_NE", "corners": np.array([
                [inner, inner], [OUTER_EXTENT, inner], [OUTER_EXTENT, FAR_EXTENT], [inner, FAR_EXTENT],
            ])},
            {"name": "building_SW", "corners": np.array([
                [-OUTER_EXTENT, -FAR_EXTENT], [-inner, -FAR_EXTENT], [-inner, -inner], [-OUTER_EXTENT, -inner],
            ])},
            {"name": "building_SE", "corners": np.array([
                [inner, -FAR_EXTENT], [OUTER_EXTENT, -FAR_EXTENT], [OUTER_EXTENT, -inner], [inner, -inner],
            ])},
        ]

    def _init_occlusion_geometry(self):
        """Build runtime occlusion polygons in SUMO frame.

        MUST be called AFTER _start_sumo(). Queries junction center from
        TraCI and offsets template corners accordingly.
        """
        if not self._occlusion_templates:
            self._occlusion_polygons = []
            # Still query junction offset for CZ sampling even without buildings
            try:
                jx, jy = traci.junction.getPosition("center")
                self._junction_offset = np.array([jx, jy], dtype=np.float64)
            except Exception:
                self._junction_offset = np.array([0.0, 0.0])
            return
        try:
            jx, jy = traci.junction.getPosition("center")
        except Exception:
            jx, jy = 0.0, 0.0
        self._junction_offset = np.array([jx, jy], dtype=np.float64)
        offset = self._junction_offset
        self._occlusion_polygons = [
            {
                "name": tpl["name"],
                "corners": tpl["corners"] + offset,
            }
            for tpl in self._occlusion_templates
        ]

    def _line_intersects_polygon(self, p1: np.ndarray, p2: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if line segment p1-p2 intersects any edge of the polygon.

        Uses the standard 2D line segment intersection test.
        polygon: (N, 2) array of corner points.
        """
        n = len(polygon)
        for i in range(n):
            p3 = polygon[i]
            p4 = polygon[(i + 1) % n]
            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-10:
                continue
            t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
            u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
        return False

    def _ray_polygon_edge_distance(
        self, origin: np.ndarray, direction: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> float | None:
        """Return distance along ray from origin in direction to edge p3-p4, or None."""
        d2 = p4 - p3
        cross = direction[0] * d2[1] - direction[1] * d2[0]
        if abs(cross) < 1e-10:
            return None
        diff = p3 - origin
        t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
        u = (diff[0] * direction[1] - diff[1] * direction[0]) / cross
        if t > 0 and 0 <= u <= 1:
            return float(t)
        return None

    def _scenario_has_static_ped(self, sumocfg: str) -> bool:
        try:
            ped_path = os.path.join(self.scenario_dir, "t_ped.rou.xml")
            if os.path.isfile(ped_path):
                return True
            with open(sumocfg, encoding="utf-8") as f:
                return "t_ped.rou.xml" in f.read()
        except Exception:
            return False

    def _start_sumo(self):
        if traci is None:
            raise RuntimeError("traci not installed. Install SUMO and set SUMO_HOME.")
        sumocfg = os.path.join(self.scenario_dir, "t.sumocfg")
        if not os.path.isfile(sumocfg) or (self._has_ped and self._scenario_has_static_ped(sumocfg)):
            from scenario.generator import ScenarioGenerator
            gen = ScenarioGenerator()
            gen.generate(self.scenario_dir, scenario_name=self.scenario_name)
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            binary = os.path.join(sumo_home, "bin", binary)
        cmd = [
            binary, "-c", sumocfg,
            "--step-length", str(self.dt),
            "--no-step-log", "true",
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--intermodal-collision.action", "warn",
        ]
        if self.use_gui:
            gui_settings = os.path.join(self.scenario_dir, "t_gui.xml")
            if os.path.isfile(gui_settings):
                cmd.extend(["--gui-settings-file", gui_settings])
        traci.start(cmd)

    def _close_sumo(self):
        try:
            if traci:
                traci.close()
        except Exception:
            pass
        self._sumo_proc = None

    def _sample_behavior(self) -> BehaviorConfig:
        return self._behavior_sampler.sample(
            self._has_car, self._has_ped, self._has_moto, self._has_pothole,
            bar_len=self._bar_len,
            stem_len=self._stem_len,
            ego_maneuver=self.ego_maneuver,
            dense=self._dense,
            style_filter=self._style_filter,
            jm_ignore_fixed=self._jm_ignore_fixed,
        )

    @staticmethod
    def _jm_type_suffix(jm_value: float) -> str:
        probs = [0, 0.05, 0.1, 0.15, 0.2]
        closest = min(probs, key=lambda p: abs(p - jm_value))
        return f"p{int(closest * 100):02d}"

    def _ensure_route(self, route_id: str, edges_str: str) -> str:
        try:
            existing = traci.route.getIDList()
            if route_id in existing:
                return route_id
            traci.route.add(route_id, edges_str.split())
            return route_id
        except Exception:
            return route_id

    def _spawn_actors(self, bcfg: BehaviorConfig):
        """Spawn actors with sampled behavior using TraCI."""
        traci.vehicle.add(self.EGO_ID, self._ego_route_id, depart="0", typeID="Car")

        if bcfg.car and self._has_car:
            cb = bcfg.car
            route_id = self._ensure_route(f"car_{cb.maneuver}", cb.route_edges)
            type_id = "CarOther"
            try:
                jm_suffix = self._jm_type_suffix(cb.jm_ignore)
                desired_type = f"CarOther_{jm_suffix}"
                if desired_type in traci.vehicletype.getIDList():
                    type_id = desired_type
            except Exception:
                pass
            first_edge = cb.route_edges.split()[0]
            dep_pos_str = _clamp_depart_pos(
                cb.depart_pos if cb.depart_pos is not None else 0.0,
                first_edge, self._bar_len - 2.0,
            )
            traci.vehicle.add(self.OTHER_ID, route_id, depart=str(cb.depart_time),
                              typeID=type_id, departPos=dep_pos_str)
            try:
                traci.vehicle.setMaxSpeed(self.OTHER_ID, cb.max_speed)
                traci.vehicle.setAccel(self.OTHER_ID, cb.accel)
                traci.vehicle.setDecel(self.OTHER_ID, cb.decel)
                traci.vehicle.setTau(self.OTHER_ID, cb.tau)
                traci.vehicle.setImperfection(self.OTHER_ID, cb.sigma)
                traci.vehicle.setSpeedFactor(self.OTHER_ID, cb.speed_factor)
            except Exception:
                pass

        if bcfg.motorcycle and self._has_moto:
            mb = bcfg.motorcycle
            route_id = self._ensure_route(f"moto_{mb.maneuver}", mb.route_edges)
            type_id = "Motorcycle"
            try:
                jm_suffix = self._jm_type_suffix(mb.jm_ignore)
                desired_type = f"Motorcycle_{jm_suffix}"
                if desired_type in traci.vehicletype.getIDList():
                    type_id = desired_type
            except Exception:
                pass
            first_edge = mb.route_edges.split()[0]
            dep_pos_str = _clamp_depart_pos(
                mb.depart_pos if mb.depart_pos is not None else 0.0,
                first_edge, self._bar_len - 2.0,
            )
            traci.vehicle.add("motorcyclist", route_id, depart=str(mb.depart_time),
                              typeID=type_id, departPos=dep_pos_str)
            try:
                traci.vehicle.setMaxSpeed("motorcyclist", mb.max_speed)
                traci.vehicle.setAccel("motorcyclist", mb.accel)
                traci.vehicle.setDecel("motorcyclist", mb.decel)
                traci.vehicle.setTau("motorcyclist", mb.tau)
                traci.vehicle.setImperfection("motorcyclist", mb.sigma)
            except Exception:
                pass

        if bcfg.pedestrian and self._has_ped:
            pb = bcfg.pedestrian
            try:
                if "ped0" in traci.person.getIDList():
                    traci.person.remove("ped0")
            except Exception:
                pass
            try:
                edges = pb.route_edges.split()
                from_edge = edges[0] if edges else "left_in"
                to_edge = edges[-1] if len(edges) > 1 else "right_out"
                dep_pos = pb.depart_pos if pb.depart_pos is not None else 0.0
                try:
                    edge_len = traci.lane.getLength(f"{from_edge}_0")
                    dep_pos = float(np.clip(dep_pos, 0.0, max(0.0, edge_len - 1.0)))
                except Exception:
                    dep_pos = float(np.clip(dep_pos, 0.0, max(0.0, self._bar_len - 2.0)))
                traci.person.add("ped0", from_edge, pos=dep_pos, depart=pb.depart_time)
                traci.person.appendWalkingStage("ped0", [from_edge, to_edge], arrivalPos=-1)
                traci.person.setSpeed("ped0", pb.ped_speed)
            except Exception:
                try:
                    traci.person.add("ped0", "left_in", pos=0.0, depart=pb.depart_time)
                    traci.person.appendWalkingStage("ped0", ["left_in", "right_out"], arrivalPos=-1)
                    traci.person.setSpeed("ped0", pb.ped_speed)
                except Exception:
                    pass

        # Insurance car (always spawned when available)
        if bcfg.car2 and self._has_car:
            cb2 = bcfg.car2
            route_id = self._ensure_route(f"car2_{cb2.maneuver}", cb2.route_edges)
            first_edge = cb2.route_edges.split()[0]
            dep_pos_str = _clamp_depart_pos(
                cb2.depart_pos if cb2.depart_pos is not None else 0.0,
                first_edge, self._bar_len - 2.0,
            )
            traci.vehicle.add("other2", route_id, depart=str(cb2.depart_time),
                              typeID="CarOther", departPos=dep_pos_str)
            try:
                traci.vehicle.setMaxSpeed("other2", cb2.max_speed)
                traci.vehicle.setAccel("other2", cb2.accel)
                traci.vehicle.setDecel("other2", cb2.decel)
                traci.vehicle.setTau("other2", cb2.tau)
                traci.vehicle.setImperfection("other2", cb2.sigma)
            except Exception:
                pass

        # Insurance pedestrian (always spawned when available)
        if bcfg.pedestrian2 and self._has_ped:
            pb2 = bcfg.pedestrian2
            try:
                edges2 = pb2.route_edges.split()
                from_edge2 = edges2[0] if edges2 else "right_out"
                to_edge2 = edges2[-1] if len(edges2) > 1 else "left_in"
                dep_pos2 = pb2.depart_pos if pb2.depart_pos is not None else 0.0
                try:
                    edge_len = traci.lane.getLength(f"{from_edge2}_0")
                    dep_pos2 = float(np.clip(dep_pos2, 0.0, max(0.0, edge_len - 1.0)))
                except Exception:
                    dep_pos2 = float(np.clip(dep_pos2, 0.0, max(0.0, self._bar_len - 2.0)))
                traci.person.add("ped1", from_edge2, pos=dep_pos2, depart=pb2.depart_time)
                traci.person.appendWalkingStage("ped1", [from_edge2, to_edge2], arrivalPos=-1)
                traci.person.setSpeed("ped1", pb2.ped_speed)
            except Exception:
                pass

        # Third car (dense only)
        if getattr(bcfg, 'car3', None) and self._has_car:
            cb3 = bcfg.car3
            route_id = self._ensure_route(f"car3_{cb3.maneuver}", cb3.route_edges)
            first_edge = cb3.route_edges.split()[0]
            dep_pos_str = _clamp_depart_pos(
                cb3.depart_pos if cb3.depart_pos is not None else 0.0,
                first_edge, self._bar_len - 2.0,
            )
            traci.vehicle.add("other3", route_id, depart=str(cb3.depart_time),
                              typeID="CarOther", departPos=dep_pos_str)
            try:
                traci.vehicle.setMaxSpeed("other3", cb3.max_speed)
                traci.vehicle.setAccel("other3", cb3.accel)
                traci.vehicle.setDecel("other3", cb3.decel)
                traci.vehicle.setTau("other3", cb3.tau)
                traci.vehicle.setImperfection("other3", cb3.sigma)
            except Exception:
                pass

        # Third pedestrian (dense only)
        if getattr(bcfg, 'pedestrian3', None) and self._has_ped:
            pb3 = bcfg.pedestrian3
            try:
                edges3 = pb3.route_edges.split()
                from_edge3 = edges3[0] if edges3 else "left_in"
                to_edge3 = edges3[-1] if len(edges3) > 1 else "right_out"
                dep_pos3 = pb3.depart_pos if pb3.depart_pos is not None else 0.0
                try:
                    edge_len = traci.lane.getLength(f"{from_edge3}_0")
                    dep_pos3 = float(np.clip(dep_pos3, 0.0, max(0.0, edge_len - 1.0)))
                except Exception:
                    dep_pos3 = float(np.clip(dep_pos3, 0.0, max(0.0, self._bar_len - 2.0)))
                traci.person.add("ped2", from_edge3, pos=dep_pos3, depart=pb3.depart_time)
                traci.person.appendWalkingStage("ped2", [from_edge3, to_edge3], arrivalPos=-1)
                traci.person.setSpeed("ped2", pb3.ped_speed)
            except Exception:
                pass

        # Insurance motorcycle
        if getattr(bcfg, 'motorcycle2', None) and self._has_moto:
            mb2 = bcfg.motorcycle2
            route_id = self._ensure_route(f"moto2_{mb2.maneuver}", mb2.route_edges)
            first_edge = mb2.route_edges.split()[0]
            dep_pos_str = _clamp_depart_pos(
                mb2.depart_pos if mb2.depart_pos is not None else 0.0,
                first_edge, self._bar_len - 2.0,
            )
            traci.vehicle.add("motorcyclist2", route_id, depart=str(mb2.depart_time),
                              typeID="Motorcycle", departPos=dep_pos_str)
            try:
                traci.vehicle.setMaxSpeed("motorcyclist2", mb2.max_speed)
                traci.vehicle.setAccel("motorcyclist2", mb2.accel)
                traci.vehicle.setDecel("motorcyclist2", mb2.decel)
                traci.vehicle.setTau("motorcyclist2", mb2.tau)
                traci.vehicle.setImperfection("motorcyclist2", mb2.sigma)
            except Exception:
                pass

        if bcfg.pothole and self._has_pothole:
            ph = bcfg.pothole
            self._pothole_box = np.array([
                [ph.x - ph.half_w, ph.x + ph.half_w],
                [ph.y - ph.half_h, ph.y + ph.half_h],
            ])

    def _check_collision_sumo(self) -> bool:
        try:
            collisions = traci.simulation.getCollidingVehiclesIDList()
            if self.EGO_ID in collisions:
                return True
        except Exception:
            pass
        return False

    def _in_pothole(self, p: np.ndarray) -> bool:
        return bool(
            self._pothole_box[0, 0] <= p[0] <= self._pothole_box[0, 1]
            and self._pothole_box[1, 0] <= p[1] <= self._pothole_box[1, 1]
        )

    def _dist_to_pothole(self, p: np.ndarray) -> float:
        cx = (self._pothole_box[0, 0] + self._pothole_box[0, 1]) / 2
        cy = (self._pothole_box[1, 0] + self._pothole_box[1, 1]) / 2
        return float(np.linalg.norm(p - np.array([cx, cy])))

    def _randomize_pothole(self):
        """Randomize pothole size and position each episode."""
        length = np.random.uniform(4.0, 12.0)
        width = np.random.uniform(2.0, 4.0)
        half_l = length / 2
        half_w = width / 2

        if self._ego_start_edge == "stem_in":
            pot_x = np.random.uniform(-2.0, 2.0)
            pot_y = np.random.uniform(-40.0, -5.0)
        elif "right" in self._ego_start_edge:
            pot_x = np.random.uniform(10.0, 40.0)
            pot_y = np.random.uniform(-2.0, 2.0)
        else:
            pot_x = np.random.uniform(-40.0, -10.0)
            pot_y = np.random.uniform(-2.0, 2.0)

        self._pothole_box = np.array([
            [pot_x - half_l, pot_x + half_l],
            [pot_y - half_w, pot_y + half_w],
        ])

        try:
            traci.polygon.remove("pothole")
        except Exception:
            pass
        try:
            shape = [
                (pot_x - half_l, pot_y - half_w),
                (pot_x + half_l, pot_y - half_w),
                (pot_x + half_l, pot_y + half_w),
                (pot_x - half_l, pot_y + half_w),
            ]
            traci.polygon.add("pothole", shape, color=(51, 38, 25, 200),
                              fill=True, layer=1, polygonType="pothole")
        except Exception:
            pass

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self._behavior_sampler.rng = np.random.RandomState(seed)
        self._close_sumo()
        self._start_sumo()
        self._init_occlusion_geometry()
        if not self._buildings_enabled:
            for bname in ("building_NW", "building_NE", "building_SW", "building_SE"):
                try:
                    traci.polygon.remove(bname)
                except Exception:
                    pass
        self._load_dims()
        self._step_count = 0
        self._agent_history = {}
        self._agent_first_seen = {}
        self._gru_hidden = None
        self._collision_flag = False
        self._ped_stopped = False
        self._ped_stop_counter = 0
        self._ped_hesitant_phase = 0
        self._ped_hesitant_counter = 0
        self._prev_action = None
        self._prev_psi = None

        self._behavior = self._sample_behavior()
        self._spawn_actors(self._behavior)

        if self._has_pothole:
            self._randomize_pothole()

        for _ in range(30):
            traci.simulationStep()
            if self.EGO_ID in traci.vehicle.getIDList():
                break

        if self.EGO_ID in traci.vehicle.getIDList():
            try:
                traci.vehicle.setSpeedMode(self.EGO_ID, 0)
            except Exception:
                pass

        ego = self._get_ego()
        self._prev_psi = ego["psi"]
        self._prev_ego = {"a": 0.0, "psi_dot": 0.0}
        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        state = self._augment_state(built, raw)
        self._update_agent_history(raw, built)
        self._prev_ego = {"a": ego.get("a", 0), "psi_dot": ego.get("psi_dot", 0.0)}
        ttc_min = self._compute_ttc_min()
        info = {
            "raw_obs": raw, "built": built,
            "ttc_min": ttc_min,
            "collision": False,
            "behavior": self._behavior,
        }
        # Pothole position diagnostic (temporary — remove after verification)
        if self._has_pothole and hasattr(self, '_pothole_box'):
            box = self._pothole_box
            print(f"[POTHOLE DIAG] scenario={self.scenario_name}, "
                  f"box x=[{box[0, 0]:.1f}, {box[0, 1]:.1f}], y=[{box[1, 0]:.1f}, {box[1, 1]:.1f}]")

        return state.astype(np.float32), info

    def _get_ego(self) -> dict:
        if self.EGO_ID not in traci.vehicle.getIDList():
            return {"p": np.array([0.0, 0.0]), "psi": 0.0, "v": 0.0, "a": 0.0, "psi_dot": 0.0}
        pos = traci.vehicle.getPosition(self.EGO_ID)
        angle = traci.vehicle.getAngle(self.EGO_ID)
        speed = traci.vehicle.getSpeed(self.EGO_ID)
        accel = traci.vehicle.getAcceleration(self.EGO_ID)
        psi = np.radians(angle)
        psi_dot = 0.0
        if self._prev_psi is not None and self.dt > 0:
            dpsi = np.arctan2(np.sin(psi - self._prev_psi), np.cos(psi - self._prev_psi))
            psi_dot = dpsi / self.dt
        return {"p": np.array(pos, dtype=float), "psi": psi, "v": speed, "a": accel, "psi_dot": psi_dot}

    def _get_agent_lane_dist(self, vid: str) -> tuple[float, float]:
        """Compute route-based distances to junction for a vehicle agent."""
        try:
            edge = traci.vehicle.getRoadID(vid)
            lane_pos = traci.vehicle.getLanePosition(vid)
            lane_len = traci.lane.getLength(traci.vehicle.getLaneID(vid))
            remaining = max(0.0, lane_len - lane_pos)
            if "out" in edge:
                return 0.0, remaining
            return remaining, remaining + 10.0
        except Exception:
            return 0.0, 0.0

    def _get_agents(self) -> list[dict]:
        agents = []
        ego_pos = np.array([0.0, 0.0])
        if self.EGO_ID in traci.vehicle.getIDList():
            ego_pos = np.array(traci.vehicle.getPosition(self.EGO_ID), dtype=float)

        for vid in traci.vehicle.getIDList():
            if vid == self.EGO_ID:
                continue
            pos = traci.vehicle.getPosition(vid)
            angle = traci.vehicle.getAngle(vid)
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)
            if "motorcyclist" in vid:
                atype = "cyc"
            else:
                atype = "veh"

            d_cz_i, d_exit_i = self._get_agent_lane_dist(vid)
            dist = np.linalg.norm(np.array(pos, dtype=float) - ego_pos) + 1e-6
            sigma_i = float(np.clip(dist / 50.0, 0.05, 1.0))

            try:
                agent_route = traci.vehicle.getRoute(vid)
                has_in = any("_in" in e for e in agent_route)
                has_out = any("_out" in e for e in agent_route)
                chi_i = 1.0 if (has_in and has_out) else 0.0
            except Exception:
                chi_i = 0.5

            try:
                ego_edge = traci.vehicle.getRoadID(self.EGO_ID) if self.EGO_ID in traci.vehicle.getIDList() else ""
                ag_edge = traci.vehicle.getRoadID(vid)
                ego_prio = -1 if "stem" in ego_edge else 1
                ag_prio = -1 if "stem" in ag_edge else 1
                pi_row_i = 0.2 if ag_prio > ego_prio else 0.8
            except Exception:
                pi_row_i = 0.5

            nu_i = self._compute_los(ego_pos, np.array(pos, dtype=float), vid)

            agents.append({
                "id": vid, "p": np.array(pos, dtype=float),
                "psi": np.radians(angle), "v": speed, "a": accel,
                "type": atype, "nu": nu_i, "sigma": sigma_i,
                "d_cz": d_cz_i, "d_exit": d_exit_i,
                "chi": chi_i, "pi_row": pi_row_i,
            })

        for pid in traci.person.getIDList():
            pos = traci.person.getPosition(pid)
            speed = traci.person.getSpeed(pid)
            try:
                ped_angle = traci.person.getAngle(pid)
                ped_psi = np.radians(ped_angle)
            except Exception:
                ped_psi = 0.0

            ped_pos = np.array(pos, dtype=float)
            dist = np.linalg.norm(ped_pos - ego_pos) + 1e-6
            # Pedestrian d_cz: distance along bar to crosswalk region
            if abs(ped_pos[0]) < 5.0 and abs(ped_pos[1]) < 10.0:
                ped_d_cz = 0.0  # already in crosswalk
            else:
                ped_d_cz = max(0.0, abs(ped_pos[0]) - 5.0)
            sigma_i = float(np.clip(dist / 50.0, 0.05, 1.0))
            nu_i = self._compute_los(ego_pos, ped_pos, pid)
            # Pedestrian ROW: absolute priority when in crosswalk
            in_crosswalk = abs(ped_pos[0]) < 8.0 and abs(ped_pos[1]) < 10.0
            pi_row_ped = 1.0 if in_crosswalk else 0.5

            agents.append({
                "id": pid, "p": ped_pos,
                "psi": ped_psi, "v": speed, "a": 0.0,
                "type": "ped", "nu": nu_i, "sigma": sigma_i,
                "d_cz": ped_d_cz, "d_exit": max(0, ped_d_cz - 5),
                "chi": 1.0, "pi_row": pi_row_ped,
            })
        return agents

    def _compute_los(self, ego_pos: np.ndarray, agent_pos: np.ndarray, agent_id: str) -> float:
        """Compute line-of-sight visibility. Returns 1.0 if clear, decays if occluded."""
        # Check static occlusion (buildings) first
        for occ in self._occlusion_polygons:
            if self._line_intersects_polygon(ego_pos, agent_pos, occ["corners"]):
                return 0.05  # almost fully occluded by building

        all_positions = []
        try:
            for vid in traci.vehicle.getIDList():
                if vid == self.EGO_ID or vid == agent_id:
                    continue
                all_positions.append(np.array(traci.vehicle.getPosition(vid), dtype=float))
        except Exception:
            pass

        if not all_positions:
            return 1.0

        ego_to_agent = agent_pos - ego_pos
        dist_ea = np.linalg.norm(ego_to_agent) + 1e-6

        for occ_pos in all_positions:
            ego_to_occ = occ_pos - ego_pos
            proj = np.dot(ego_to_occ, ego_to_agent) / (dist_ea ** 2)
            if proj < 0.05 or proj > 0.95:
                continue
            closest = ego_pos + proj * ego_to_agent
            lateral_dist = np.linalg.norm(occ_pos - closest)
            if lateral_dist < 3.0:
                return max(0.1, lateral_dist / 3.0)

        return 1.0

    def _get_geom_vis(self, ego: dict) -> tuple[dict, dict]:
        ego_present = self.EGO_ID in traci.vehicle.getIDList()
        edge = traci.vehicle.getRoadID(self.EGO_ID) if ego_present else ""
        lane_pos = traci.vehicle.getLanePosition(self.EGO_ID) if ego_present else 0.0

        on_start_edge = self._ego_start_edge in edge if edge else False
        on_exit_edge = self._ego_exit_edge in edge if edge else False
        in_junction = (edge == "" or ":center" in edge or ":J" in edge) if edge else False

        if on_start_edge:
            edge_len = self._stem_len if "stem" in edge else self._bar_len
            remaining = max(0, edge_len - lane_pos)
            d_cz = max(0, remaining - 10)
            d_stop = max(0, remaining - 5)
            d_exit = remaining
        elif in_junction:
            d_cz = 0.0
            d_stop = 0.0
            d_exit = max(0, 10.0 - lane_pos) if lane_pos > 0 else 5.0
        elif on_exit_edge:
            d_cz = 0.0
            d_stop = 0.0
            exit_edge_len = self._stem_len if "stem" in edge else self._bar_len
            d_exit = max(0, exit_edge_len - lane_pos)
        else:
            d_cz = 0.0
            d_stop = 0.0
            d_exit = 0.0

        v_ego = float(ego.get("v", 0.0))
        psi_dot_ego = float(ego.get("psi_dot", 0.0))
        kappa = psi_dot_ego / max(v_ego, 0.5) if v_ego > 0.5 else 0.0

        e_y = 0.0
        e_psi = 0.0
        try:
            if self.EGO_ID in traci.vehicle.getIDList():
                e_y = traci.vehicle.getLateralLanePosition(self.EGO_ID)
                lane_id = traci.vehicle.getLaneID(self.EGO_ID)
                lane_angle = traci.lane.getShape(lane_id)
                if len(lane_angle) >= 2:
                    seg = np.array(lane_angle[-1]) - np.array(lane_angle[-2])
                    lane_heading = math.atan2(seg[1], seg[0])
                    e_psi = _wrap(ego.get("psi", 0.0) - lane_heading)
        except Exception:
            pass

        if self._ego_start_edge == "stem_in":
            rho_ego_priority = 0.0
            rho_ego_must_yield = 1.0
        elif self.ego_maneuver in ("right_stem", "left_stem"):
            rho_ego_priority = 0.5
            rho_ego_must_yield = 0.5
        else:
            rho_ego_priority = 1.0
            rho_ego_must_yield = 0.0

        geom = {
            "d_stop": d_stop, "d_cz": d_cz, "d_exit": d_exit,
            "kappa": kappa, "e_y": e_y, "e_psi": e_psi,
            "w_lane": 3.5, "g_turn": [0, 0, 1],
            "rho": [rho_ego_priority, rho_ego_must_yield],
        }

        agents = self._get_agents() if not hasattr(self, '_cached_agents') else self._cached_agents
        ego_pos = np.array(ego["p"])

        # alpha_cz: geometric visibility of cross-traffic along the bar road.
        # Sampled as a rectangle along the bar (where conflicting traffic lives),
        # not a square at the junction center. This correctly reflects corner
        # buildings' occluding effect on ego's view of bar traffic.
        cz_cx, cz_cy = self._junction_offset
        CZ_BAR_HALF_LENGTH = 30.0   # extent along bar road (east + west of junction)
        CZ_BAR_HALF_WIDTH = 7.0     # bar road half-width
        n_samples = 20
        visible_count = 0
        for _ in range(n_samples):
            sx = cz_cx + np.random.uniform(-CZ_BAR_HALF_LENGTH, CZ_BAR_HALF_LENGTH)
            sy = cz_cy + np.random.uniform(-CZ_BAR_HALF_WIDTH, CZ_BAR_HALF_WIDTH)
            sample = np.array([sx, sy])
            if not any(self._line_intersects_polygon(ego_pos, sample, occ["corners"])
                       for occ in self._occlusion_polygons):
                visible_count += 1
        alpha_cz = visible_count / n_samples
        alpha_cross = alpha_cz

        # d_occ: distance from ego to nearest static occlusion boundary toward CZ
        d_occ = 200.0
        cz_center = self._junction_offset
        ego_to_cz = cz_center - ego_pos
        ego_to_cz_norm = ego_to_cz / (np.linalg.norm(ego_to_cz) + 1e-6)
        for occ in self._occlusion_polygons:
            corners = occ["corners"]
            n_corners = len(corners)
            for i in range(n_corners):
                p3 = corners[i]
                p4 = corners[(i + 1) % n_corners]
                d = self._ray_polygon_edge_distance(ego_pos, ego_to_cz_norm, p3, p4)
                if d is not None and d < d_occ:
                    d_occ = d

        dt_seen = 0.0
        if agents:
            first_seen_times = [self._agent_first_seen.get(ag["id"], self._step_count) for ag in agents]
            if first_seen_times:
                earliest = min(first_seen_times)
                dt_seen = (self._step_count - earliest) * self.dt

        # sigma_percep: dynamic perception uncertainty based on occlusion
        n_agents_total = len(agents)
        n_occluded = sum(1 for ag in agents if ag.get("nu", 1.0) < 0.5)
        sigma_percep = 0.05 + 0.15 * (n_occluded / max(n_agents_total, 1))

        vis = {
            "alpha_cz": alpha_cz,
            "alpha_cross": alpha_cross,
            "d_occ": d_occ,
            "dt_seen": dt_seen,
            "sigma_percep": sigma_percep,
            "n_occ": sum(1 for ag in agents if ag.get("nu", 1.0) < 0.8),
        }

        # State ablation: zero out visibility features while keeping physical occlusion
        if self._state_ablation == "no_visibility":
            vis = {
                "alpha_cz": 1.0,
                "alpha_cross": 1.0,
                "d_occ": 200.0,
                "dt_seen": 0.0,
                "sigma_percep": 0.05,
                "n_occ": 0.0,
            }

        return geom, vis

    def _get_raw_obs(self) -> dict:
        ego = self._get_ego()
        agents = self._get_agents()
        self._cached_agents = agents

        for ag in agents:
            aid = ag.get("id", "?")
            if aid not in self._agent_first_seen:
                self._agent_first_seen[aid] = self._step_count

        geom, vis = self._get_geom_vis(ego)
        if hasattr(self, '_cached_agents'):
            del self._cached_agents

        raw = {"ego": ego, "agents": agents, "geom": geom, "vis": vis}
        if self._has_pothole:
            raw["d_pothole"] = self._dist_to_pothole(ego["p"])
            raw["in_pothole"] = self._in_pothole(ego["p"])
        return raw

    def _update_agent_history(self, raw: dict, built: dict):
        ego = raw["ego"]
        p_e = np.array(ego["p"])
        psi_e = float(ego.get("psi", 0))
        v_e = float(ego.get("v", 0))
        v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])
        for ag in raw.get("agents", []):
            aid = ag.get("id", "?")
            if aid not in self._agent_history:
                self._agent_history[aid] = []
            p_i = np.array(ag["p"])
            psi_i = float(ag.get("psi", 0))
            v_i = float(ag.get("v", 0))
            d_cz = float(ag.get("d_cz", 1e6))
            v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])
            R = _rot2d(-psi_e)
            dp = p_i - p_e
            delta_xy = R @ dp
            delta_v = R @ (v_i_vec - v_e_vec)
            delta_psi = _wrap(psi_i - psi_e)
            t_cpa = np.clip(-np.dot(delta_xy, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
            p_cpa = delta_xy + t_cpa * delta_v
            d_cpa = np.linalg.norm(p_cpa)
            z = [delta_xy[0], delta_xy[1], delta_v[0], delta_v[1], delta_psi, d_cz, d_cpa,
                 ag.get("nu", 1.0), ag.get("sigma", 0.1)]
            self._agent_history[aid].append(z)
            if len(self._agent_history[aid]) > 20:
                self._agent_history[aid].pop(0)

    def _get_intent_features(self, built: dict, raw: dict) -> np.ndarray:
        if not self.use_intent or self._intent_predictor is None:
            return np.array([], dtype=np.float32)

        agent_list = raw.get("agents", [])
        ego = raw["ego"]
        p_e = np.array(ego.get("p", [0, 0]))
        v_e = float(ego.get("v", 0))
        psi_e = float(ego.get("psi", 0))
        d_cz_e = float(built.get("s_geom", np.zeros(12))[1])
        eps = 1e-6
        tau_e = d_cz_e / max(v_e, eps)

        scores = []
        for i, ag in enumerate(agent_list):
            v_i = float(ag.get("v", 0))
            d_cz_i = float(ag.get("d_cz", 1e6))
            tau_i = d_cz_i / max(v_i, eps)
            dist = np.linalg.norm(np.array(ag.get("p", [0, 0])) - p_e) + eps
            scores.append((tau_i, dist, i))
        scores.sort(key=lambda x: (x[0], x[1]))
        sorted_indices = [x[2] for x in scores[:5]]

        intent_feats = []
        for idx in sorted_indices:
            ag = agent_list[idx]
            aid = ag.get("id", str(idx))
            hist = self._agent_history.get(aid, [])
            if len(hist) < 2:
                intent_feats.append(np.zeros(6, dtype=np.float32))
                continue
            h = np.array(hist[-min(10, len(hist)):], dtype=np.float32)
            try:
                import torch
                with torch.no_grad():
                    z = torch.FloatTensor(h).unsqueeze(0).to(self._intent_device)
                    ip, sp, _, _ = self._intent_predictor(z, None)
                    ip_last = ip[0, -1].cpu().numpy()
                    sp_last = sp[0, -1].cpu().numpy()
                    intent_feats.append(np.concatenate([ip_last, sp_last]))
            except Exception:
                intent_feats.append(np.zeros(6, dtype=np.float32))
        while len(intent_feats) < 5:
            intent_feats.append(np.zeros(6, dtype=np.float32))
        return np.concatenate(intent_feats[:5])

    def _augment_state(self, built: dict, raw: dict) -> np.ndarray:
        state = built["state"].astype(np.float32)
        extras = []
        if self.use_intent:
            extras.append(self._get_intent_features(built, raw))
        if self._has_pothole:
            extras.append(np.array([raw.get("d_pothole", 100.0)], dtype=np.float32))
        if extras:
            state = np.concatenate([state] + extras)
        return state

    def _compute_ttc_min(self) -> float:
        ego = self._get_ego()
        agents = self._get_agents()
        ttc = 1e6
        v_e = ego["v"] * np.array([np.cos(ego["psi"]), np.sin(ego["psi"])])
        for ag in agents:
            dp = np.array(ag["p"]) - ego["p"]
            v_i = ag["v"] * np.array([np.cos(ag["psi"]), np.sin(ag["psi"])])
            dv = v_i - v_e
            dv_norm = np.linalg.norm(dv) + 1e-6
            t_cpa = np.clip(-np.dot(dp, dv) / (dv_norm ** 2), 0, 3.0)
            p_cpa = dp + t_cpa * dv
            d_cpa = np.linalg.norm(p_cpa)
            ttc_i = max(d_cpa - 2.0, 0) / dv_norm
            ttc = min(ttc, ttc_i)
        return ttc if ttc < 1e5 else 10.0

    def _apply_ped_behavior(self):
        """Apply pedestrian behavior overrides (stop_midway, hesitant)."""
        if not self._behavior or not self._behavior.pedestrian:
            return
        pb = self._behavior.pedestrian
        if "ped0" not in traci.person.getIDList():
            return

        ped_pos = traci.person.getPosition("ped0")
        near_center = abs(ped_pos[0]) < 12 and abs(ped_pos[1]) < 12

        if pb.stop_midway:
            if near_center and not self._ped_stopped:
                self._ped_stopped = True
                self._ped_stop_counter = int(pb.stop_duration / self.dt)
                try:
                    traci.person.setSpeed("ped0", 0.0)
                except Exception:
                    pass
            if self._ped_stopped and self._ped_stop_counter > 0:
                self._ped_stop_counter -= 1
            elif self._ped_stopped and self._ped_stop_counter <= 0:
                self._ped_stopped = False
                try:
                    traci.person.setSpeed("ped0", pb.ped_speed)
                except Exception:
                    pass

        elif pb.hesitant:
            if self._ped_hesitant_phase == 0:
                if near_center:
                    self._ped_hesitant_phase = 1
                    self._ped_hesitant_counter = int(1.5 / self.dt)
                    try:
                        traci.person.setSpeed("ped0", 0.0)
                    except Exception:
                        pass
            elif self._ped_hesitant_phase == 1:
                self._ped_hesitant_counter -= 1
                if self._ped_hesitant_counter <= 0:
                    self._ped_hesitant_phase = 2
                    self._ped_hesitant_counter = int(1.0 / self.dt)
                    try:
                        traci.person.setSpeed("ped0", pb.ped_speed * 0.3)
                    except Exception:
                        pass
            elif self._ped_hesitant_phase == 2:
                self._ped_hesitant_counter -= 1
                if self._ped_hesitant_counter <= 0:
                    self._ped_hesitant_phase = 3
                    try:
                        traci.person.setSpeed("ped0", pb.ped_speed)
                    except Exception:
                        pass

    def _apply_action(self, action: int):
        """Apply RL action to ego vehicle. Actions are behavioral modes."""
        if self.EGO_ID not in traci.vehicle.getIDList():
            return
        v = traci.vehicle.getSpeed(self.EGO_ID)
        if action == 0:     # STOP: controlled braking
            traci.vehicle.slowDown(self.EGO_ID, max(0, v - 5.0 * self.dt), self.dt)
        elif action == 1:   # CREEP: regulate toward ~1 m/s
            target_v = 1.0
            if v < target_v:
                new_v = min(target_v, v + 0.5 * self.dt)
            else:
                new_v = max(target_v, v - 1.0 * self.dt)
            traci.vehicle.slowDown(self.EGO_ID, max(0, new_v), self.dt)
        elif action == 2:   # YIELD: gentle deceleration
            traci.vehicle.slowDown(self.EGO_ID, max(0, v - 0.5 * self.dt), self.dt)
        elif action == 3:   # GO: accelerate at 2.0 m/s^2
            traci.vehicle.setSpeed(self.EGO_ID, min(13.89, v + 2.0 * self.dt))
        else:               # ABORT: emergency braking (harder than STOP)
            traci.vehicle.slowDown(self.EGO_ID, max(0, v - 8.0 * self.dt), self.dt)

    def _ego_must_yield(self, current_edge: str) -> bool:
        """Determine if the ego must yield based on maneuver and current position."""
        # Stem-origin: ego always yields when approaching from minor road
        if self._ego_start_edge == "stem_in":
            if "stem" in current_edge or current_edge == "" or ":center" in current_edge:
                return True
        # Bar-origin turns into stem: yield to oncoming bar traffic
        if self.ego_maneuver == "right_stem":
            if "right" in current_edge or current_edge == "" or ":center" in current_edge:
                return True
        if self.ego_maneuver == "left_stem":
            if "left" in current_edge or current_edge == "" or ":center" in current_edge:
                return True
        return False

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(action)
        self._apply_ped_behavior()
        n_steps = max(1, int(self.dt / 0.1))
        for _ in range(n_steps):
            traci.simulationStep()
        self._step_count += 1

        sumo_collision = self._check_collision_sumo()
        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        self._update_agent_history(raw, built)
        state = self._augment_state(built, raw)
        ego = raw["ego"]
        agents = raw["agents"]
        self._prev_ego = {"a": ego.get("a", 0), "psi_dot": ego.get("psi_dot", 0.0)}
        self._prev_psi = ego.get("psi", self._prev_psi)

        ttc_min = self._compute_ttc_min()
        d_coll = self.reward_cfg.get("d_coll", 2.0)
        ego_present = self.EGO_ID in traci.vehicle.getIDList()

        proximity_collision = False
        if ego_present:
            for ag in agents:
                if np.linalg.norm(np.array(ag["p"]) - ego["p"]) < d_coll:
                    proximity_collision = True
                    break
        collision = sumo_collision or proximity_collision
        if collision:
            self._collision_flag = True

        # --- Termination & success detection (before reward so success bonus applies) ---
        success = False
        terminated = collision
        if not terminated and ego_present:
            current_edge = traci.vehicle.getRoadID(self.EGO_ID)
            if current_edge == self._ego_exit_edge:
                lane_pos = traci.vehicle.getLanePosition(self.EGO_ID)
                exit_len = self._bar_len if "left" in self._ego_exit_edge or "right" in self._ego_exit_edge else self._stem_len
                if lane_pos > exit_len - 10:
                    terminated = True
                    success = True
        ego_missing_success = (not collision) and (not ego_present) and (self._step_count > 0)
        if ego_missing_success:
            terminated = True
            success = True
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        # --- Reward computation ---
        prog = 0.1
        if ego_present:
            prog = traci.vehicle.getSpeed(self.EGO_ID) * self.dt
        r = self.reward_cfg["w_prog"] * prog + self.reward_cfg["w_time"] * self.dt
        if ttc_min < self.reward_cfg["ttc_thr"]:
            r += self.reward_cfg["w_risk"]
        if collision:
            r += self.reward_cfg["w_coll"]
        if self._has_pothole and raw.get("in_pothole", False):
            r += self.reward_cfg["w_pothole"]
        if action == 4:
            r += self.reward_cfg.get("w_abort_comfort", -0.5)
        if success:
            r += self.reward_cfg.get("w_success", 10.0)

        # Action switching penalty
        if self._prev_action is not None and action != self._prev_action:
            r += self.reward_cfg.get("w_switch", -0.05)

        # ROW violation penalty
        if ego_present and not collision:
            ego_edge = traci.vehicle.getRoadID(self.EGO_ID)
            if self._ego_must_yield(ego_edge):
                for ag in agents:
                    ag_dist = np.linalg.norm(np.array(ag["p"]) - ego["p"])
                    if ag_dist < 15.0 and ag.get("pi_row", 0) > 0.5:
                        d_cz_ego = built["s_geom"][1]
                        if d_cz_ego < 3.0 and ego["v"] > 1.0:
                            r += self.reward_cfg.get("w_rule", -2.0)
                            break

        self._prev_action = action

        # --- Info dict ---
        ego_action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "UNKNOWN"
        nearest_agent_dist = float("inf")
        if ego_present:
            for ag in agents:
                d = np.linalg.norm(np.array(ag["p"]) - ego["p"])
                nearest_agent_dist = min(nearest_agent_dist, d)

        info = {
            "raw_obs": raw, "built": built,
            "ttc_min": ttc_min,
            "collision": collision,
            "sumo_collision": sumo_collision,
            "proximity_collision": proximity_collision,
            "in_pothole": raw.get("in_pothole", False),
            "behavior": self._behavior,
            "action_name": ego_action_name,
            "ego_speed": ego.get("v", 0.0),
            "nearest_agent_dist": nearest_agent_dist,
            "ego_missing": not ego_present,
            "ego_missing_success": ego_missing_success,
            "success": success,
            "prev_action": self._prev_action,
        }
        return state.astype(np.float32), float(r), terminated, truncated, info

    def close(self):
        self._close_sumo()

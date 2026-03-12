"""SUMO T-intersection Gymnasium environment with TraCI.
Supports scenarios 1a-1d, 2, 3, 4. Full behavior diversity.
- Per-episode sampled maneuvers, styles, and timing via BehaviorSampler
- Random pothole placement
- Proper collision detection via SUMO events + proximity
- Intent features aligned with state-builder agent ordering
- GRU hidden state carryover support
"""

from __future__ import annotations

import os
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

_DEFAULT_REWARD_CONFIG = {
    "w_prog": 1.0, "w_time": -0.1, "w_risk": -3.0, "w_coll": -20.0,
    "ttc_thr": 3.0, "d_coll": 2.0, "w_pothole": -5.0,
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


class SumoEnv(_make_gym()):
    """SUMO T-intersection: ego turns right. Full behavior diversity."""

    EGO_ID = "ego"
    OTHER_ID = "other"
    SCENARIO_TYPES = ["1a", "1b", "1c", "1d", "2", "3", "4"]

    def __init__(
        self,
        scenario_dir: str | None = None,
        scenario_name: str = "1a",
        use_gui: bool = False,
        state_config: str | None = None,
        reward_config: str | None = "configs/reward/default.yaml",
        max_steps: int = 500,
        dt: float = 0.1,
        use_intent: bool = False,
        jm_ignore_fixed: float | None = None,
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
        self.scenario_dir = scenario_dir or os.path.join(base, "scenarios", f"sumo_{self.scenario_name}")

        self._stem_len, self._bar_len = 200.0, 160.0
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
        self._gru_hidden = None  # GRU hidden state for recurrence
        self._behavior: Optional[BehaviorConfig] = None
        self._behavior_sampler = BehaviorSampler()
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

    def _start_sumo(self):
        if traci is None:
            raise RuntimeError("traci not installed. Install SUMO and set SUMO_HOME.")
        sumocfg = os.path.join(self.scenario_dir, "t.sumocfg")
        if not os.path.isfile(sumocfg):
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
            jm_ignore_fixed=self._jm_ignore_fixed,
        )

    @staticmethod
    def _jm_type_suffix(jm_value: float) -> str:
        probs = [0, 0.05, 0.1, 0.15, 0.2]
        closest = min(probs, key=lambda p: abs(p - jm_value))
        return f"p{int(closest * 100):02d}"

    def _ensure_route(self, route_id: str, edges_str: str) -> str:
        """Return route_id if it exists, otherwise create it from edge string."""
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
        traci.vehicle.add(self.EGO_ID, "ego_route", depart="0", typeID="Car")

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
            traci.vehicle.add(self.OTHER_ID, route_id, depart=str(cb.depart_time), typeID=type_id)
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
            traci.vehicle.add("motorcyclist", route_id, depart=str(mb.depart_time), typeID=type_id)
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
                traci.person.add("ped0", from_edge, pos=dep_pos, depart=pb.depart_time)
                traci.person.appendWalkingStage("ped0", [from_edge, to_edge], arrivalPos=-1)
                traci.person.setSpeed("ped0", pb.ped_speed)
            except Exception:
                pass

        if bcfg.pothole and self._has_pothole:
            ph = bcfg.pothole
            self._pothole_box = np.array([
                [ph.x - ph.half_w, ph.x + ph.half_w],
                [ph.y - ph.half_h, ph.y + ph.half_h],
            ])

    def _check_collision_sumo(self) -> bool:
        """Check SUMO collision list for actual collision events."""
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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self._behavior_sampler.rng = np.random.RandomState(seed)
        self._close_sumo()
        self._start_sumo()
        self._load_dims()
        self._step_count = 0
        self._agent_history = {}
        self._gru_hidden = None
        self._collision_flag = False
        self._ped_stopped = False
        self._ped_stop_counter = 0
        self._ped_hesitant_phase = 0
        self._ped_hesitant_counter = 0
        self._prev_psi = None

        self._behavior = self._sample_behavior()
        self._spawn_actors(self._behavior)

        for _ in range(20):
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

    def _get_agents(self) -> list[dict]:
        agents = []
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
            agents.append({
                "id": vid, "p": np.array(pos, dtype=float),
                "psi": np.radians(angle), "v": speed, "a": accel,
                "type": atype, "nu": 1.0, "sigma": 0.1,
                "d_cz": 0.0, "d_exit": 0.0, "chi": 0.5, "pi_row": 0.5,
            })
        for pid in traci.person.getIDList():
            pos = traci.person.getPosition(pid)
            agents.append({
                "id": pid, "p": np.array(pos, dtype=float),
                "psi": 0.0, "v": traci.person.getSpeed(pid), "a": 0.0,
                "type": "ped", "nu": 1.0, "sigma": 0.1,
                "d_cz": 0.0, "d_exit": 0.0, "chi": 0.5, "pi_row": 0.5,
            })
        return agents

    def _get_geom_vis(self, ego: dict) -> tuple[dict, dict]:
        edge = traci.vehicle.getRoadID(self.EGO_ID) if self.EGO_ID in traci.vehicle.getIDList() else ""
        lane_pos = traci.vehicle.getLanePosition(self.EGO_ID) if self.EGO_ID in traci.vehicle.getIDList() else 0.0
        on_right_out = "right_out" in edge
        if "stem_in" in edge:
            lane_len = self._stem_len
        elif on_right_out:
            lane_len = self._bar_len
        else:
            lane_len = min(self._stem_len, self._bar_len)
        d_stop = max(0, lane_len - lane_pos - 5)
        d_exit = max(0, lane_len - lane_pos)
        # d_cz: distance to conflict zone; 0 once past the junction
        d_cz = 0.0 if on_right_out else max(0, lane_len - lane_pos - 10)
        # kappa: real curvature from ego yaw rate and speed (psi_dot / v)
        v_ego = float(ego.get("v", 0.0))
        psi_dot_ego = float(ego.get("psi_dot", 0.0))
        kappa = psi_dot_ego / max(v_ego, 0.5) if v_ego > 0.5 else 0.0
        geom = {
            "d_stop": d_stop, "d_cz": d_cz, "d_exit": d_exit,
            "kappa": kappa, "e_y": 0.0, "e_psi": 0.0,
            "w_lane": 3.5, "g_turn": [0, 0, 1], "rho": [0.5, 0.5],
        }
        vis = {"alpha_cz": 1.0, "alpha_cross": 1.0, "d_occ": 50.0, "dt_seen": 0.0, "sigma_percep": 0.05, "n_occ": 0.0}
        return geom, vis

    def _get_raw_obs(self) -> dict:
        ego = self._get_ego()
        agents = self._get_agents()
        for ag in agents:
            ag["d_cz"] = np.linalg.norm(ag["p"] - ego["p"]) * 0.5
            ag["d_exit"] = ag["d_cz"] * 0.8
        geom, vis = self._get_geom_vis(ego)
        raw = {"ego": ego, "agents": agents, "geom": geom, "vis": vis}
        if self._has_pothole:
            raw["d_pothole"] = self._dist_to_pothole(ego["p"])
            raw["in_pothole"] = self._in_pothole(ego["p"])
        return raw

    def _update_agent_history(self, raw: dict, built: dict):
        """Track agent history keyed by agent ID for intent prediction."""
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
            t_cpa = np.clip(-np.dot(dp, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
            p_cpa = dp + t_cpa * delta_v
            d_cpa = np.linalg.norm(p_cpa)
            z = [delta_xy[0], delta_xy[1], delta_v[0], delta_v[1], delta_psi, d_cz, d_cpa, 1.0, 0.1]
            self._agent_history[aid].append(z)
            if len(self._agent_history[aid]) > 20:
                self._agent_history[aid].pop(0)

    def _get_intent_features(self, built: dict, raw: dict) -> np.ndarray:
        """Get intent features ALIGNED with the state-builder agent ordering."""
        if not self.use_intent or self._intent_predictor is None:
            return np.array([], dtype=np.float32)

        # Get the sorted indices that state builder used
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
        """Apply pedestrian behavior overrides (stop_midway, hesitant).

        stop_midway: walk normally, stop in the conflict zone for stop_duration, resume.
        hesitant: 3-phase pattern -- walk toward CZ, stop and retreat slightly, then go.
          phase 0: approach normally
          phase 1: stop near CZ entrance for ~1.5s
          phase 2: slow to 0.3*speed (simulating retreat/hesitation) for ~1s
          phase 3: resume at full speed
        """
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
        if self.EGO_ID not in traci.vehicle.getIDList():
            return
        v = traci.vehicle.getSpeed(self.EGO_ID)
        if action == 0:     # STOP
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 5.0 * self.dt))
        elif action == 1:   # CREEP
            target = 1.0 if v < 1.0 else v - 0.2 * self.dt
            traci.vehicle.setSpeed(self.EGO_ID, max(0, min(13.89, target + 0.5 * self.dt)))
        elif action == 2:   # YIELD
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 0.5 * self.dt))
        elif action == 3:   # GO
            traci.vehicle.setSpeed(self.EGO_ID, min(13.89, v + 1.0 * self.dt))
        else:               # ABORT
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 5.0 * self.dt))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(action)
        self._apply_ped_behavior()
        n_steps = max(1, int(self.dt / 0.1))
        for _ in range(n_steps):
            traci.simulationStep()
        self._step_count += 1

        # Collision detection: SUMO events + proximity
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

        # Proximity collision check
        proximity_collision = False
        for ag in agents:
            if np.linalg.norm(np.array(ag["p"]) - ego["p"]) < d_coll:
                proximity_collision = True
                break
        collision = sumo_collision or proximity_collision
        if collision:
            self._collision_flag = True

        # Reward
        prog = 0.1
        if self.EGO_ID in traci.vehicle.getIDList():
            prog = traci.vehicle.getSpeed(self.EGO_ID) * self.dt
        r = self.reward_cfg["w_prog"] * prog + self.reward_cfg["w_time"] * self.dt
        if ttc_min < self.reward_cfg["ttc_thr"]:
            r += self.reward_cfg["w_risk"]
        if collision:
            r += self.reward_cfg["w_coll"]
        if self._has_pothole and raw.get("in_pothole", False):
            r += self.reward_cfg["w_pothole"]

        terminated = collision
        if not terminated and self.EGO_ID in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(self.EGO_ID) == "right_out":
                lane_pos = traci.vehicle.getLanePosition(self.EGO_ID)
                if lane_pos > self._bar_len - 25:
                    terminated = True
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        info = {
            "raw_obs": raw, "built": built,
            "ttc_min": ttc_min,
            "collision": collision,
            "sumo_collision": sumo_collision,
            "proximity_collision": proximity_collision,
            "in_pothole": raw.get("in_pothole", False),
            "behavior": self._behavior,
        }
        return state.astype(np.float32), float(r), terminated, truncated, info

    def close(self):
        self._close_sumo()

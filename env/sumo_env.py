"""SUMO T-intersection Gymnasium environment with TraCI.
Supports scenarios 1a-1d, 2, 3, 4. Collisions enabled. Pothole support."""

from __future__ import annotations

import os
import numpy as np
from typing import Any

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

ACTION_NAMES = ["STOP", "CREEP", "YIELD", "GO", "ABORT"]
N_ACTIONS = 5

_DEFAULT_REWARD_CONFIG = {
    "w_prog": 1.0, "w_time": -0.1, "w_risk": -1.0, "w_coll": -10.0,
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


# Pothole: axis-aligned rectangle at junction center (used when scenario has pothole)
POTHOLE_BOX = np.array([[-4, 4], [-1.5, 1.5]])  # xmin, xmax; ymin, ymax


def _in_pothole(p: np.ndarray) -> bool:
    return bool(
        POTHOLE_BOX[0, 0] <= p[0] <= POTHOLE_BOX[0, 1]
        and POTHOLE_BOX[1, 0] <= p[1] <= POTHOLE_BOX[1, 1]
    )


def _dist_to_pothole(p: np.ndarray) -> float:
    cx = (POTHOLE_BOX[0, 0] + POTHOLE_BOX[0, 1]) / 2
    cy = (POTHOLE_BOX[1, 0] + POTHOLE_BOX[1, 1]) / 2
    return float(np.linalg.norm(p - np.array([cx, cy])))


class SumoEnv(_make_gym()):
    """SUMO T-intersection: ego turns right. Scenarios 1a-1d, 2, 3, 4. Collisions enabled."""

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

        # State dim: base 134 + intent (5*6=30) + pothole (1)
        self._state_dim = 6 + 12 + 6 + 5 * 22
        if use_intent:
            self._state_dim += 5 * 6
        if self._has_pothole:
            self._state_dim += 1
        if spaces:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
            )
        else:
            obs_shape = (self._state_dim,)
            self.observation_space = type("ObsSpace", (), {"shape": obs_shape})()
        self.action_space = spaces.Discrete(N_ACTIONS) if spaces else type("ActSpace", (), {"n": N_ACTIONS})

        self._sumo_proc = None
        self._step_count = 0
        self._prev_ego = None
        self._jm_ignore_fixed = jm_ignore_fixed
        self._agent_history: dict[str, list] = {}
        self._intent_predictor = None
        if use_intent:
            try:
                from models.intent_style import IntentStylePredictor
                import torch
                self._intent_predictor = IntentStylePredictor(input_dim=9, hidden_dim=64).eval()
                self._intent_device = "cuda" if torch.cuda.is_available() else "cpu"
                self._intent_predictor.to(self._intent_device)
            except Exception:
                self.use_intent = False

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
        traci.start(cmd)

    def _close_sumo(self):
        try:
            if traci:
                traci.close()
        except Exception:
            pass
        self._sumo_proc = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._close_sumo()
        self._start_sumo()
        self._step_count = 0
        self._agent_history = {}

        opts = options or {}
        dep_other = float(opts.get("other_depart", 5))
        from scenario.generator import JM_IGNORE_PROBS, jm_type_suffix
        if self._jm_ignore_fixed is not None:
            jm_prob = float(np.clip(self._jm_ignore_fixed, 0, 0.2))
            jm_nearest = min(JM_IGNORE_PROBS, key=lambda x: abs(x - jm_prob))
        else:
            jm_nearest = float(np.random.choice(JM_IGNORE_PROBS))
        suf = jm_type_suffix(jm_nearest)
        traci.vehicle.add(self.EGO_ID, "ego_route", depart="0", typeID="Car")
        if self._has_car:
            traci.vehicle.add(self.OTHER_ID, "other_route", depart=str(dep_other), typeID=f"CarOther_{suf}")
        if self._has_moto:
            traci.vehicle.add("motorcyclist", "moto_route", depart=str(dep_other + 1), typeID=f"Motorcycle_{suf}")

        for _ in range(20):
            traci.simulationStep()
            if self.EGO_ID in traci.vehicle.getIDList():
                break

        if self.EGO_ID in traci.vehicle.getIDList():
            try:
                traci.vehicle.setSpeedMode(self.EGO_ID, 0)
            except Exception:
                pass

        self._prev_ego = {"a": 0.0, "psi_dot": 0.0}
        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        state = self._augment_state(built, raw)
        self._update_agent_history(raw, built)
        self._prev_ego = {"a": self._get_ego().get("a", 0), "psi_dot": 0.0}
        return state.astype(np.float32), {"raw_obs": raw, "built": built}

    def _get_ego(self) -> dict:
        if self.EGO_ID not in traci.vehicle.getIDList():
            return {"p": np.array([0.0, 0.0]), "psi": 0.0, "v": 0.0, "a": 0.0}
        pos = traci.vehicle.getPosition(self.EGO_ID)
        angle = traci.vehicle.getAngle(self.EGO_ID)
        speed = traci.vehicle.getSpeed(self.EGO_ID)
        accel = traci.vehicle.getAcceleration(self.EGO_ID)
        return {
            "p": np.array(pos, dtype=float),
            "psi": np.radians(angle),
            "v": speed,
            "a": accel,
        }

    def _get_agents(self) -> list[dict]:
        agents = []
        for vid in traci.vehicle.getIDList():
            if vid == self.EGO_ID:
                continue
            pos = traci.vehicle.getPosition(vid)
            angle = traci.vehicle.getAngle(vid)
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)
            vclass = traci.vehicle.getVehicleClass(vid) if hasattr(traci.vehicle, "getVehicleClass") else "passenger"
            atype = "cyc" if "bicycle" in str(vclass).lower() else "veh"
            agents.append({
                "id": vid,
                "p": np.array(pos, dtype=float),
                "psi": np.radians(angle),
                "v": speed,
                "a": accel,
                "type": atype,
                "nu": 1.0,
                "sigma": 0.1,
                "d_cz": 0.0,
                "d_exit": 0.0,
                "chi": 0.5,
                "pi_row": 0.5,
            })
        for pid in traci.person.getIDList():
            pos = traci.person.getPosition(pid)
            agents.append({
                "id": pid,
                "p": np.array(pos, dtype=float),
                "psi": 0.0,
                "v": traci.person.getSpeed(pid),
                "a": 0.0,
                "type": "ped",
                "nu": 1.0,
                "sigma": 0.1,
                "d_cz": 0.0,
                "d_exit": 0.0,
                "chi": 0.5,
                "pi_row": 0.5,
            })
        return agents

    def _get_geom_vis(self, ego: dict) -> tuple[dict, dict]:
        edge = traci.vehicle.getRoadID(self.EGO_ID) if self.EGO_ID in traci.vehicle.getIDList() else ""
        lane_pos = traci.vehicle.getLanePosition(self.EGO_ID) if self.EGO_ID in traci.vehicle.getIDList() else 0.0
        if "stem_in" in edge:
            lane_len = 100.0
        elif "right_out" in edge:
            lane_len = 80.0
        else:
            lane_len = 50.0
        d_stop = max(0, lane_len - lane_pos - 5)
        d_cz = max(0, lane_len - lane_pos - 10)
        d_exit = max(0, lane_len - lane_pos)
        geom = {
            "d_stop": d_stop, "d_cz": d_cz, "d_exit": d_exit,
            "kappa": 0.0, "e_y": 0.0, "e_psi": 0.0,
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
            raw["d_pothole"] = _dist_to_pothole(ego["p"])
            raw["in_pothole"] = _in_pothole(ego["p"])
        return raw

    def _update_agent_history(self, raw: dict, built: dict):
        if not self.use_intent or self._intent_predictor is None:
            return
        ego = raw["ego"]
        p_e = np.array(ego["p"])
        psi_e = float(ego.get("psi", 0))
        v_e = float(ego.get("v", 0))
        v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])
        f_agents = built.get("f_agents", np.zeros((5, 22)))
        agent_list = raw.get("agents", [])
        for i, ag in enumerate(agent_list[:5]):
            aid = ag.get("id", str(i))
            if aid not in self._agent_history:
                self._agent_history[aid] = []
            p_i = np.array(ag["p"])
            psi_i = float(ag.get("psi", 0))
            v_i = float(ag.get("v", 0))
            d_cz = float(ag.get("d_cz", 1e6))
            v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])
            from state.builder import _rot2d, _wrap
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
        if not self.use_intent or self._intent_predictor is None:
            return np.array([], dtype=np.float32)
        agent_list = raw.get("agents", [])
        intent_feats = []
        scores = []
        for i, ag in enumerate(agent_list):
            aid = ag.get("id", str(i))
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
            intent_feats = self._get_intent_features(built, raw)
            extras.append(intent_feats)
        if self._has_pothole:
            d_pot = raw.get("d_pothole", 100.0)
            extras.append(np.array([d_pot], dtype=np.float32))
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

    def _apply_action(self, action: int):
        if self.EGO_ID not in traci.vehicle.getIDList():
            return
        v = traci.vehicle.getSpeed(self.EGO_ID)
        if action == 0:
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 5.0 * self.dt))
        elif action == 1:
            target = 1.0 if v < 1.0 else v - 0.2 * self.dt
            traci.vehicle.setSpeed(self.EGO_ID, max(0, min(13.89, target + 0.5 * self.dt)))
        elif action == 2:
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 0.5 * self.dt))
        elif action == 3:
            traci.vehicle.setSpeed(self.EGO_ID, min(13.89, v + 1.0 * self.dt))
        else:
            traci.vehicle.setSpeed(self.EGO_ID, max(0, v - 5.0 * self.dt))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(action)
        n_steps = max(1, int(self.dt / 0.1))
        for _ in range(n_steps):
            traci.simulationStep()
        self._step_count += 1

        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        self._update_agent_history(raw, built)
        state = self._augment_state(built, raw)
        ego = raw["ego"]
        agents = raw["agents"]
        self._prev_ego = {"a": ego.get("a", 0), "psi_dot": 0.0}

        ttc_min = self._compute_ttc_min()
        prog = 0.1
        if self.EGO_ID in traci.vehicle.getIDList():
            prog = traci.vehicle.getSpeed(self.EGO_ID) * self.dt
        w_prog = self.reward_cfg.get("w_prog", 1.0)
        w_time = self.reward_cfg.get("w_time", -0.1)
        w_risk = self.reward_cfg.get("w_risk", -1.0)
        w_coll = self.reward_cfg.get("w_coll", -10.0)
        w_pothole = self.reward_cfg.get("w_pothole", -5.0)
        ttc_thr = self.reward_cfg.get("ttc_thr", 3.0)
        d_coll = self.reward_cfg.get("d_coll", 2.0)

        r = w_prog * prog + w_time * self.dt
        if ttc_min < ttc_thr:
            r += w_risk
        for ag in agents:
            if np.linalg.norm(np.array(ag["p"]) - ego["p"]) < d_coll:
                r += w_coll
        if self._has_pothole and raw.get("in_pothole", False):
            r += w_pothole

        done = self._step_count >= self.max_steps
        if self.EGO_ID in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(self.EGO_ID) == "right_out":
                lane_pos = traci.vehicle.getLanePosition(self.EGO_ID)
                if lane_pos > 70:
                    done = True

        return (
            state.astype(np.float32),
            float(r),
            False,
            done,
            {"raw_obs": raw, "built": built, "ttc_min": ttc_min, "in_pothole": raw.get("in_pothole", False)},
        )

    def close(self):
        self._close_sumo()

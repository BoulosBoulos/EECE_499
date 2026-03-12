"""SUMO T-intersection scenario generation. Eight scenarios:
  1a: ego + other car | 1b: ego + pedestrian | 1c: ego + motorcyclist | 1d: ego + pothole
  2: ego + car + pedestrian | 3: ego + car + pedestrian + motorcyclist
  4: ego + car + pedestrian + motorcyclist + pothole
  Network: doubled size, bidirectional (car coming + going on all arms).
  Other car: left->right. Motorcycle: right->left (separate from other car).
"""

from __future__ import annotations

import os
import subprocess
import shutil

try:
    import yaml
except ImportError:
    yaml = None

# Scenario layout: (has_car, has_ped, has_moto, has_pothole)
SCENARIO_SPEC = {
    "1a": (True, False, False, False),   # ego + other car
    "1b": (False, True, False, False),   # ego + pedestrian only
    "1c": (False, False, True, False),   # ego + motorcyclist only
    "1d": (False, False, False, True),    # ego + pothole only
    "2": (True, True, False, False),     # car + pedestrian
    "3": (True, True, True, False),      # car + pedestrian + motorcyclist
    "4": (True, True, True, True),       # car + pedestrian + motorcyclist + pothole
}
SCENARIO_TYPES = list(SCENARIO_SPEC.keys())

# jmIgnoreJunctionFoeProb values for per-episode sampling
JM_IGNORE_PROBS = [0, 0.05, 0.1, 0.15, 0.2]


def jm_type_suffix(prob: float) -> str:
    """Return vType suffix for given jmIgnoreJunctionFoeProb, e.g. 0.1 -> 'p10'."""
    return "p" + str(int(prob * 100)).zfill(2)


def _load_config(path: str | None) -> dict:
    if path is None or yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_netconvert() -> str:
    nc = shutil.which("netconvert")
    if nc:
        return nc
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    nc = os.path.join(sumo_home, "bin", "netconvert")
    if os.path.isfile(nc):
        return nc
    raise RuntimeError("netconvert not found. Install SUMO and set SUMO_HOME.")


class ScenarioGenerator:
    """Generate T-intersection scenarios with collisions enabled.
    Bidirectional network: stem_in/out, left_in/out, right_in/out.
    """

    def __init__(self, config_path: str | None = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = config_path or os.path.join(base, "configs/scenario/default.yaml")
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        self.cfg = _load_config(path)
        self.stem_len = float(self.cfg.get("stem_length", 200))
        self.bar_len = float(self.cfg.get("bar_half_length", 160))
        self.junction_type = self.cfg.get("junction_type", "priority")

    def generate(self, output_dir: str, scenario_name: str = "1a") -> dict[str, str]:
        """Generate scenario. scenario_name: 1a, 1b, 1c, 1d, 2, 3, 4."""
        if scenario_name not in SCENARIO_SPEC:
            scenario_name = "1a"
        has_car, has_ped, has_moto, has_pothole = SCENARIO_SPEC[scenario_name]
        os.makedirs(output_dir, exist_ok=True)

        sw = ' sidewalkWidth="2"' if has_ped else ""
        # Bidirectional: 2 lanes per edge (car coming + going). All 6 edges.
        edg_path = os.path.join(output_dir, "t.edg.xml")
        with open(edg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<edges>\n')
            f.write(f'  <edge from="stem" id="stem_in" to="center" numLanes="2" speed="13.89" priority="-1"{sw}/>\n')
            f.write(f'  <edge from="center" id="stem_out" to="stem" numLanes="2" speed="13.89" priority="-1"{sw}/>\n')
            f.write(f'  <edge from="left" id="left_in" to="center" numLanes="2" speed="13.89" priority="1"{sw}/>\n')
            f.write(f'  <edge from="center" id="left_out" to="left" numLanes="2" speed="13.89" priority="1"{sw}/>\n')
            f.write(f'  <edge from="right" id="right_in" to="center" numLanes="2" speed="13.89" priority="1"{sw}/>\n')
            f.write(f'  <edge from="center" id="right_out" to="right" numLanes="2" speed="13.89" priority="1"{sw}/>\n')
            f.write("</edges>\n")

        nod_path = os.path.join(output_dir, "t.nod.xml")
        with open(nod_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<nodes>\n')
            f.write(f'  <node id="center" x="0" y="0" type="{self.junction_type}"/>\n')
            f.write(f'  <node id="stem" x="0" y="-{self.stem_len}" type="priority"/>\n')
            f.write(f'  <node id="left" x="-{self.bar_len}" y="0" type="priority"/>\n')
            f.write(f'  <node id="right" x="{self.bar_len}" y="0" type="priority"/>\n')
            f.write("</nodes>\n")

        net_path = os.path.join(output_dir, "t.net.xml")
        nc_cmd = [
            _get_netconvert(),
            "--node-files", nod_path,
            "--edge-files", edg_path,
            "--output", net_path,
        ]
        if has_ped:
            nc_cmd.extend(["--crossings.guess", "true"])
        subprocess.run(nc_cmd, check=True, capture_output=True, text=True)

        # vTypes: Car (ego=blue), CarOther (red), Motorcycle (orange)
        rou_path = os.path.join(output_dir, "t.rou.xml")
        JM_PROBS = [0, 0.05, 0.1, 0.15, 0.2]
        with open(rou_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            f.write('  <vType id="Car" accel="2.0" decel="5.0" length="4.0" maxSpeed="13.89" sigma="0.0" color="0,0,1"/>\n')
            for p in JM_PROBS:
                pid = str(int(p * 100)).zfill(2)
                f.write(f'  <vType id="CarOther_p{pid}" accel="2.5" decel="5.0" length="4.0" maxSpeed="13.89" sigma="0.15" '
                        f'jmIgnoreJunctionFoeProb="{p}" tau="0.5" color="1,0,0"/>\n')
            if has_moto:
                for p in JM_PROBS:
                    pid = str(int(p * 100)).zfill(2)
                    f.write(f'  <vType id="Motorcycle_p{pid}" accel="4.0" decel="8.0" length="2.2" maxSpeed="16.67" sigma="0.15" '
                            f'jmIgnoreJunctionFoeProb="{p}" tau="0.4" color="1,0.5,0"/>\n')
            f.write('  <route id="ego_route" edges="stem_in right_out"/>\n')
            if has_car:
                f.write('  <route id="other_route" edges="left_in right_out"/>\n')
            if has_moto:
                # Motorcycle from RIGHT side (right_in -> left_out), not behind other car
                f.write('  <route id="moto_route" edges="right_in left_out"/>\n')
            f.write('</routes>\n')

        if has_ped:
            ped_path = os.path.join(output_dir, "t_ped.rou.xml")
            # Pedestrian starts near crossing: departPos puts them ~20m from junction on left_in
            depart_pos = max(0, self.bar_len - 25)
            with open(ped_path, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<routes>\n')
                f.write(f'  <person id="ped0" depart="1" departPos="{depart_pos:.0f}" color="0,0.9,0">\n')
                f.write('    <walk from="left_in" to="right_out"/>\n')
                f.write('  </person>\n')
                f.write('</routes>\n')
            rou_files = f'    <route-files value="{os.path.basename(rou_path)},{os.path.basename(ped_path)}"/>\n'
        else:
            rou_files = f'    <route-files value="{os.path.basename(rou_path)}"/>\n'

        # Ground polygon for larger network (convBoundary ~0-320 x, 0-200 y)
        ground_path = os.path.join(output_dir, "t_ground.poly.xml")
        with open(ground_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<additional>\n')
            f.write('  <poly id="ground" type="grass" color="0.35,0.65,0.25,1" fill="1" layer="-10" '
                    'shape="-50,-50 370,-50 370,250 -50,250"/>\n')
            f.write('</additional>\n')

        add_parts = [os.path.basename(ground_path)]
        if has_pothole:
            poly_path = os.path.join(output_dir, "t_pothole.poly.xml")
            with open(poly_path, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<additional>\n')
                f.write('  <poly id="pothole" type="pothole" color=".3,.3,.3" fill="1" layer="0" shape="-4,-1.5 4,-1.5 4,1.5 -4,1.5"/>\n')
                f.write('</additional>\n')
            add_parts.append(os.path.basename(poly_path))
        add_files = f'    <additional-files value="{",".join(add_parts)}"/>\n'

        gui_path = os.path.join(output_dir, "t_gui.xml")
        with open(gui_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<viewsettings>\n')
            f.write('  <scheme name="real world"/>\n')
            f.write('  <delay value="100"/>\n')
            f.write('  <vehicleSize value="4"/>\n')
            f.write('  <personSize value="2.5"/>\n')
            f.write('  <vehicleNameSize value="1"/>\n')
            f.write('</viewsettings>\n')

        cfg_path = os.path.join(output_dir, "t.sumocfg")
        with open(cfg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration>\n')
            f.write('  <input>\n')
            f.write(f'    <net-file value="{os.path.basename(net_path)}"/>\n')
            f.write(rou_files)
            f.write(add_files)
            f.write("  </input>\n")
            f.write('  <gui_only>\n')
            f.write(f'    <gui-settings-file value="{os.path.basename(gui_path)}"/>\n')
            f.write('  </gui_only>\n')
            f.write("  <time><begin value=\"0\"/><end value=\"10000\"/></time>\n")
            f.write("  <step-length value=\"0.1\"/>\n")
            f.write("</configuration>\n")

        # Save scenario dims for env (stem_len, bar_len)
        dims_path = os.path.join(output_dir, "scenario_dims.yaml")
        with open(dims_path, "w") as f:
            f.write(f"stem_length: {self.stem_len}\n")
            f.write(f"bar_half_length: {self.bar_len}\n")

        return {"net": net_path, "rou": rou_path, "sumocfg": cfg_path, "output_dir": output_dir}

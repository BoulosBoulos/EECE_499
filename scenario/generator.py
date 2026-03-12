"""SUMO T-intersection scenario generation. Eight scenarios:
  1a: ego + other car | 1b: ego + pedestrian | 1c: ego + motorcyclist | 1d: ego + pothole
  2: ego + car + pedestrian | 3: ego + car + pedestrian + motorcyclist
  4: ego + car + pedestrian + motorcyclist + pothole
  Network: doubled size, bidirectional (2 lanes per direction on all arms).
  Routes: all combinations of left/right/stem for diverse maneuvers.
"""

from __future__ import annotations

import os
import subprocess
import shutil

try:
    import yaml
except ImportError:
    yaml = None

SCENARIO_SPEC = {
    "1a": (True, False, False, False),
    "1b": (False, True, False, False),
    "1c": (False, False, True, False),
    "1d": (False, False, False, True),
    "2": (True, True, False, False),
    "3": (True, True, True, False),
    "4": (True, True, True, True),
}
SCENARIO_TYPES = list(SCENARIO_SPEC.keys())

JM_IGNORE_PROBS = [0, 0.05, 0.1, 0.15, 0.2]


def jm_type_suffix(prob: float) -> str:
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


ALL_VEHICLE_ROUTES = {
    "car_left_right":  "left_in right_out",
    "car_right_left":  "right_in left_out",
    "car_left_stem":   "left_in stem_out",
    "car_right_stem":  "right_in stem_out",
    "moto_right_left": "right_in left_out",
    "moto_left_right": "left_in right_out",
    "moto_right_stem": "right_in stem_out",
    "ego_route":       "stem_in right_out",
}


class ScenarioGenerator:
    """Generate T-intersection scenarios with bidirectional network and all route variants."""

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
        if scenario_name not in SCENARIO_SPEC:
            scenario_name = "1a"
        has_car, has_ped, has_moto, has_pothole = SCENARIO_SPEC[scenario_name]
        os.makedirs(output_dir, exist_ok=True)

        sw = ' sidewalkWidth="3"' if has_ped else ""

        edg_path = os.path.join(output_dir, "t.edg.xml")
        with open(edg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
            for eid, fr, to, prio in [
                ("stem_in", "stem", "center", -1), ("stem_out", "center", "stem", -1),
                ("left_in", "left", "center", 1), ("left_out", "center", "left", 1),
                ("right_in", "right", "center", 1), ("right_out", "center", "right", 1),
            ]:
                f.write(f'  <edge from="{fr}" id="{eid}" to="{to}" numLanes="2" speed="13.89" priority="{prio}"{sw}/>\n')
            f.write("</edges>\n")

        nod_path = os.path.join(output_dir, "t.nod.xml")
        with open(nod_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
            f.write(f'  <node id="center" x="0" y="0" type="{self.junction_type}"/>\n')
            f.write(f'  <node id="stem" x="0" y="-{self.stem_len}" type="priority"/>\n')
            f.write(f'  <node id="left" x="-{self.bar_len}" y="0" type="priority"/>\n')
            f.write(f'  <node id="right" x="{self.bar_len}" y="0" type="priority"/>\n')
            f.write("</nodes>\n")

        net_path = os.path.join(output_dir, "t.net.xml")
        nc_cmd = [_get_netconvert(), "--node-files", nod_path, "--edge-files", edg_path, "--output", net_path]
        if has_ped:
            nc_cmd.extend(["--crossings.guess", "true"])
        subprocess.run(nc_cmd, check=True, capture_output=True, text=True)

        rou_path = os.path.join(output_dir, "t.rou.xml")
        with open(rou_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
            f.write('  <vType id="Car" accel="2.0" decel="5.0" length="4.0" maxSpeed="13.89" sigma="0.0" color="0,0,1"/>\n')
            f.write('  <vType id="CarOther" accel="2.5" decel="5.0" length="4.0" maxSpeed="13.89" sigma="0.15" tau="0.5" color="1,0,0"/>\n')
            if has_moto:
                f.write('  <vType id="Motorcycle" accel="4.0" decel="8.0" length="2.2" maxSpeed="16.67" sigma="0.15" tau="0.4" color="1,0.5,0"/>\n')
            for name, edges in ALL_VEHICLE_ROUTES.items():
                f.write(f'  <route id="{name}" edges="{edges}"/>\n')
            f.write('</routes>\n')

        if has_ped:
            ped_path = os.path.join(output_dir, "t_ped.rou.xml")
            depart_pos = max(0, self.bar_len - 25)
            with open(ped_path, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
                f.write(f'  <person id="ped0" depart="1" departPos="{depart_pos:.0f}" color="0,0.9,0">\n')
                f.write('    <walk from="left_in" to="right_out"/>\n')
                f.write('  </person>\n')
                f.write('</routes>\n')
            rou_files = f'    <route-files value="{os.path.basename(rou_path)},{os.path.basename(ped_path)}"/>\n'
        else:
            rou_files = f'    <route-files value="{os.path.basename(rou_path)}"/>\n'

        ground_path = os.path.join(output_dir, "t_ground.poly.xml")
        with open(ground_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
            f.write('  <poly id="ground" type="grass" color="0.35,0.65,0.25,1" fill="1" layer="-10" '
                    'shape="-50,-50 370,-50 370,250 -50,250"/>\n')
            f.write('</additional>\n')

        add_parts = [os.path.basename(ground_path)]
        if has_pothole:
            poly_path = os.path.join(output_dir, "t_pothole.poly.xml")
            with open(poly_path, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
                f.write('  <poly id="pothole" type="pothole" color="0.2,0.15,0.1" fill="1" layer="1" '
                        'shape="-4,-2 4,-2 4,2 -4,2" lineWidth="0.5"/>\n')
                f.write('</additional>\n')
            add_parts.append(os.path.basename(poly_path))
        add_files = f'    <additional-files value="{",".join(add_parts)}"/>\n'

        gui_path = os.path.join(output_dir, "t_gui.xml")
        with open(gui_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<viewsettings>\n')
            f.write('  <scheme name="real world"/>\n')
            f.write('  <delay value="80"/>\n')
            f.write('  <vehicleSize value="5"/>\n')
            f.write('  <personSize value="5"/>\n')
            f.write('  <vehicleNameSize value="1.5"/>\n')
            f.write('</viewsettings>\n')

        cfg_path = os.path.join(output_dir, "t.sumocfg")
        with open(cfg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration>\n  <input>\n')
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

        dims_path = os.path.join(output_dir, "scenario_dims.yaml")
        with open(dims_path, "w") as f:
            f.write(f"stem_length: {self.stem_len}\n")
            f.write(f"bar_half_length: {self.bar_len}\n")

        return {"net": net_path, "rou": rou_path, "sumocfg": cfg_path, "output_dir": output_dir}

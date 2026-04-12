"""Simplified SUMO T-intersection generator for the interaction benchmark.

Produces a single-lane-per-direction network with explicit pedestrian
crossings.  This keeps the benchmark focused on behavioural decision
making rather than lane-level maneuver artifacts.
"""

from __future__ import annotations

import os
import subprocess
import shutil

try:
    import yaml
except ImportError:
    yaml = None

from scenario.conflict_map import ROUTE_CONFLICTS


def _get_netconvert() -> str:
    nc = shutil.which("netconvert")
    if nc:
        return nc
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    nc = os.path.join(sumo_home, "bin", "netconvert")
    if os.path.isfile(nc):
        return nc
    raise RuntimeError("netconvert not found. Install SUMO and set SUMO_HOME.")


SCENARIO_SPEC_V2 = {
    "1a": (True, False, False, False),
    "1b": (False, True, False, False),
    "1c": (False, False, True, False),
    "1d": (False, False, False, True),
    "2":  (True, True, False, False),
    "3":  (True, True, True, False),
    "4":  (True, True, True, True),
}


class InteractionScenarioGenerator:
    """Generate a single-lane T-intersection for the behavioral benchmark."""

    def __init__(
        self,
        stem_len: float = 50.0,
        bar_len: float = 50.0,
        lane_width: float = 3.5,
        sidewalk_width: float = 3.0,
    ):
        self.stem_len = stem_len
        self.bar_len = bar_len
        self.lane_width = lane_width
        self.sw_width = sidewalk_width

    def generate(self, output_dir: str, scenario_name: str = "1a") -> dict:
        spec = SCENARIO_SPEC_V2.get(scenario_name, (True, False, False, False))
        has_car, has_ped, has_moto, has_pothole = spec
        needs_sidewalk = has_ped or scenario_name in ("2", "3", "4")
        os.makedirs(output_dir, exist_ok=True)

        sw = f' sidewalkWidth="{self.sw_width}"' if needs_sidewalk else ""

        edg_path = os.path.join(output_dir, "t.edg.xml")
        with open(edg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
            for eid, fr, to, prio in [
                ("stem_in",  "stem",   "center",  -1),
                ("stem_out", "center", "stem",    -1),
                ("left_in",  "left",   "center",   1),
                ("left_out", "center", "left",     1),
                ("right_in", "right",  "center",   1),
                ("right_out","center", "right",    1),
            ]:
                f.write(
                    f'  <edge from="{fr}" id="{eid}" to="{to}" '
                    f'numLanes="1" speed="13.89" priority="{prio}"{sw}/>\n'
                )
            f.write("</edges>\n")

        nod_path = os.path.join(output_dir, "t.nod.xml")
        with open(nod_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
            f.write(f'  <node id="center" x="0" y="0" type="priority"/>\n')
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
        if needs_sidewalk:
            nc_cmd.extend(["--crossings.guess", "true"])
        subprocess.run(nc_cmd, check=True, capture_output=True, text=True)

        rou_path = os.path.join(output_dir, "t.rou.xml")
        with open(rou_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
            f.write('  <vType id="Car" accel="2.0" decel="5.0" length="4.0" '
                    'maxSpeed="13.89" sigma="0.0" color="0,0,1"/>\n')
            f.write('  <vType id="CarOther" accel="2.5" decel="5.0" length="4.0" '
                    'maxSpeed="13.89" sigma="0.1" tau="0.5" color="1,0,0"/>\n')
            if has_moto or scenario_name in ("3", "4"):
                f.write('  <vType id="Motorcycle" accel="4.0" decel="8.0" '
                        'length="2.2" maxSpeed="16.67" sigma="0.1" tau="0.3" '
                        'color="1,0.5,0"/>\n')
            f.write('  <route id="ego_route" edges="stem_in right_out"/>\n')
            for name, rpc in ROUTE_CONFLICTS.items():
                if rpc.actor_type != "ped":
                    f.write(f'  <route id="{name}" edges="{rpc.actor_edges}"/>\n')
            f.write('</routes>\n')

        ped_path = os.path.join(output_dir, "t_ped.rou.xml")
        if os.path.isfile(ped_path):
            os.remove(ped_path)

        margin = max(self.stem_len, self.bar_len) + 15
        ground_path = os.path.join(output_dir, "t_ground.poly.xml")
        with open(ground_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
            f.write(
                f'  <poly id="ground" type="grass" color="0.35,0.65,0.25,1" '
                f'fill="1" layer="-10" '
                f'shape="-{margin},-{margin} {margin},-{margin} '
                f'{margin},{margin} -{margin},{margin}"/>\n'
            )
            f.write('</additional>\n')

        add_parts = [os.path.basename(ground_path)]
        if has_pothole:
            poly_path = os.path.join(output_dir, "t_pothole.poly.xml")
            with open(poly_path, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
                f.write('  <poly id="pothole" type="pothole" '
                        'color="0.2,0.15,0.1" fill="1" layer="1" '
                        'shape="-4,-2 4,-2 4,2 -4,2" lineWidth="0.5"/>\n')
                f.write('</additional>\n')
            add_parts.append(os.path.basename(poly_path))
        add_files = ",".join(add_parts)

        gui_path = os.path.join(output_dir, "t_gui.xml")
        with open(gui_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<viewsettings>\n')
            f.write('  <scheme name="real world"/>\n')
            f.write('  <delay value="80"/>\n')
            f.write('  <vehicleSize value="5"/>\n')
            f.write('  <personSize value="5"/>\n')
            f.write('</viewsettings>\n')

        cfg_path = os.path.join(output_dir, "t.sumocfg")
        with open(cfg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration>\n  <input>\n')
            f.write(f'    <net-file value="{os.path.basename(net_path)}"/>\n')
            f.write(f'    <route-files value="{os.path.basename(rou_path)}"/>\n')
            f.write(f'    <additional-files value="{add_files}"/>\n')
            f.write("  </input>\n")
            f.write('  <gui_only>\n')
            f.write(f'    <gui-settings-file value="{os.path.basename(gui_path)}"/>\n')
            f.write('  </gui_only>\n')
            f.write('  <time><begin value="0"/><end value="10000"/></time>\n')
            f.write('  <step-length value="0.1"/>\n')
            f.write("</configuration>\n")

        dims_path = os.path.join(output_dir, "scenario_dims.yaml")
        with open(dims_path, "w") as f:
            f.write(f"stem_length: {self.stem_len}\n")
            f.write(f"bar_half_length: {self.bar_len}\n")
            f.write(f"num_lanes: 1\n")
            f.write(f"benchmark_mode: interaction_v2\n")

        return {
            "net": net_path, "rou": rou_path, "sumocfg": cfg_path,
            "output_dir": output_dir,
        }

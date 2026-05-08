"""Step 7 helper — update config_frozen_v1.yaml with the calibrated step count.

Reads results/calibration_analysis/calibrated_total_steps.json :: calibrated_steps,
then updates the four total_steps locations in config_frozen_v1.yaml:
  training.default_total_steps
  tier1.total_steps
  tier2.shared.total_steps
  tier3.total_steps

Approach: line-level regex (not yaml.safe_dump) to preserve comments + whitespace.
"""
from __future__ import annotations
import json, re, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(dry_run: bool = False) -> int:
    cal_path = ROOT / "results" / "calibration_analysis" / "calibrated_total_steps.json"
    yaml_path = ROOT / "config_frozen_v1.yaml"
    if not cal_path.is_file():
        print(f"[yaml-update] missing {cal_path}; cannot proceed")
        return 1
    cal = json.loads(cal_path.read_text())
    new_steps = int(cal["calibrated_steps"])
    print(f"[yaml-update] target calibrated_steps = {new_steps}")

    text = yaml_path.read_text()
    new_text = text

    # Match every `total_steps: <int>` line (preserve indentation + trailing comments).
    pattern = re.compile(r"(^[ \t]*total_steps:[ \t]+)([\d_]+)(.*)$", re.MULTILINE)
    matches = list(pattern.finditer(new_text))
    if not matches:
        print("[yaml-update] no total_steps lines found; aborting")
        return 1
    print(f"[yaml-update] found {len(matches)} total_steps line(s):")
    for m in matches:
        line_no = new_text[:m.start()].count("\n") + 1
        old_value = m.group(2)
        print(f"   line {line_no:4d}: {m.group(0)!r}  (old={old_value})")

    new_text = pattern.sub(lambda m: f"{m.group(1)}{new_steps}{m.group(3)}", new_text)

    # Also update default_total_steps (fallback CLI default), if present.
    default_pattern = re.compile(r"(^[ \t]*default_total_steps:[ \t]+)([\d_]+)(.*)$", re.MULTILINE)
    default_matches = list(default_pattern.finditer(new_text))
    if default_matches:
        print(f"[yaml-update] also updating {len(default_matches)} default_total_steps line(s)")
        new_text = default_pattern.sub(lambda m: f"{m.group(1)}{new_steps}{m.group(3)}", new_text)

    if dry_run:
        print("[yaml-update] DRY RUN — not writing")
        return 0

    yaml_path.write_text(new_text)
    print(f"[yaml-update] wrote {yaml_path}")

    # Regenerate config_lock.json
    sys.path.insert(0, str(ROOT))
    from config_loader import generate_config_lock, compute_config_hash, get_config
    get_config(reload=True)
    lock_path = generate_config_lock()
    print(f"[yaml-update] regenerated lock at {lock_path}, new hash {compute_config_hash()[:16]}…")
    return 0


if __name__ == "__main__":
    dry = "--dry_run" in sys.argv
    sys.exit(main(dry_run=dry))

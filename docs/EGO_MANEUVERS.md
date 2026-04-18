# Ego Maneuver Extension

## Overview

The ego vehicle can execute 6 different maneuvers at the T-intersection, controlled by the `ego_maneuver` parameter. Default `stem_right` preserves backward compatibility.

## T-Intersection Layout

```
              left_in -->          <-- right_in
     [LEFT] ============[CENTER]============ [RIGHT]
              <-- left_out          right_out -->
                            |  ^
                            |  |
                        stem_out  stem_in
                            |  |
                            v  |
                          [STEM]
```

- **Stem** (south): 60 m, minor road, priority = -1
- **Bar** (east-west): 50 m each side, major road, priority = 1
- **Junction center**: (0, 0)

## Available Maneuvers

| Key | Start | Exit | Difficulty | Yield To |
|-----|-------|------|------------|----------|
| `stem_right` | `stem_in` | `right_out` | Tier A (hardest) | All bar traffic |
| `stem_left` | `stem_in` | `left_out` | Tier A (hardest) | All bar traffic + crosses oncoming |
| `right_left` | `right_in` | `left_out` | Tier C (easiest) | Pedestrians only (has priority) |
| `right_stem` | `right_in` | `stem_out` | Tier B | Oncoming left_in traffic |
| `left_right` | `left_in` | `right_out` | Tier C (easiest) | Pedestrians only (has priority) |
| `left_stem` | `left_in` | `stem_out` | Tier B | Oncoming right_in traffic |

## Geometry Features per Maneuver Phase

The ego's journey has 3 phases: **start edge** -> **junction** -> **exit edge**.

| Phase | d_cz | d_stop | d_exit |
|-------|------|--------|--------|
| On start edge | `remaining - 10` | `remaining - 5` | `remaining` |
| In junction | 0 | 0 | ~5-10 m |
| On exit edge | 0 | 0 | `exit_len - lane_pos` |

Where `remaining = edge_len - lane_pos`, and `edge_len` is `stem_len` (60m) for stem edges or `bar_len` (50m) for bar edges.

## Right-of-Way Rules (`_ego_must_yield`)

| Maneuver | Must yield when on... | Rationale |
|----------|----------------------|-----------|
| `stem_right` | stem or junction | Minor road yields to major |
| `stem_left` | stem or junction | Minor road yields to major |
| `right_stem` | right_in or junction | Left turn yields to oncoming |
| `left_stem` | left_in or junction | Left turn yields to oncoming (here: right turn geographically) |
| `right_left` | Never (except peds) | Through traffic on major has priority |
| `left_right` | Never (except peds) | Through traffic on major has priority |

## ROW Context (rho)

| Maneuver type | rho_priority | rho_must_yield |
|--------------|-------------|----------------|
| Stem-origin | 0.0 | 1.0 |
| Bar turn into stem | 0.5 | 0.5 |
| Bar through | 1.0 | 0.0 |

## Conflict Spawning

Agent routes are biased (70/30) toward routes that cross the ego's path:

- **stem_right**: cars from left_in or right_in crossing the junction
- **stem_left**: same, plus right_in traffic the ego must cross
- **right_left / left_right**: stem traffic entering the junction
- **right_stem / left_stem**: oncoming bar traffic

## Occlusion Behavior

- **Stem-origin**: Buildings at NW/NE corners block view of bar traffic. Strong partial observability. alpha_cz starts low (~0.3) and increases as ego approaches.
- **Bar-origin**: Direct view into junction from the side. alpha_cz is high (~0.8-1.0) from the start. Buildings are behind or to the side.

## Example Commands

```bash
# Train with stem-right (default, legacy behavior)
python experiments/pde/train_hjb_aux.py --scenario 1a

# Train with stem-left (hardest maneuver)
python experiments/pde/train_hjb_aux.py --scenario 1a --ego_maneuver stem_left

# Evaluate on bar through-traffic
python experiments/pde/eval.py --checkpoint model.pt --method hjb_aux --scenario 1a --ego_maneuver right_left

# Visualize in SUMO GUI
python experiments/pde/visualize_sumo.py --gui --checkpoint model.pt --method hjb_aux --ego_maneuver left_stem
```

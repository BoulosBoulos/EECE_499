# Simulation Extensions (SPEC A3)

## 1. Pothole Randomization

Each episode randomizes pothole size and position via `_randomize_pothole()` called in `reset()`.

- **Size**: length in [4, 12]m, width in [2, 4]m
- **Position**: placed on the ego's approach path:
  - Stem-origin: x near 0, y in [-40, -5]
  - Right-origin: x in [10, 40], y near 0
  - Left-origin: x in [-40, -10], y near 0
- **Visualization**: dynamic TraCI polygon (replaces static XML polygon each reset)

## 2. Occlusion Ablation (`buildings` flag)

```python
SumoEnv(buildings=True)   # default: NW/NE building polygons active
SumoEnv(buildings=False)  # no occlusion: full visibility
```

When `buildings=False`:
- `_occlusion_polygons` is empty list
- `_compute_los` returns 1.0 for all agents (no static occlusion)
- `alpha_cz = 1.0` (all CZ samples visible)
- `d_occ = 200.0` (no nearby occlusion boundary)
- Building polygons removed from SUMO GUI in `reset()`

**CLI**: `--no_buildings`

**Difference from `state_ablation="no_visibility"`**: `buildings=False` removes the *physical* occlusion — agents are actually visible. `no_visibility` keeps physical occlusion but removes visibility *information* from the state vector.

## 3. Dense Scenario Variants

| Scenario | Agents |
|----------|--------|
| `2_dense` | 2 cars + 2 pedestrians |
| `3_dense` | 2 cars + 2 pedestrians + 1 motorcycle |
| `4_dense` | 2 cars + 2 pedestrians + 1 motorcycle + pothole |

Dense variants use the same SUMO network as their base (e.g., `2_dense` uses `scenarios/sumo_2/`).

**Implementation**: `BehaviorConfig` has `car2` and `pedestrian2` fields. `_spawn_actors` spawns them as `"other2"` and `"ped1"` respectively. Second car uses a different conflict route with +1s depart offset.

## 4. Style Filter (Behavioral Robustness)

```python
SumoEnv(style_filter=None)           # all styles (default)
SumoEnv(style_filter="nominal")      # predictable agents only
SumoEnv(style_filter="adversarial")  # unpredictable agents only
```

| Agent Type | Nominal | Adversarial |
|-----------|---------|-------------|
| Car | nominal, timid | aggressive, distracted, erratic, drunk, rule_violating |
| Pedestrian | normal_walk, slow_elderly | running, stop_midway, hesitant, distracted_slow, jaywalking_fast |
| Motorcycle | nominal, cautious, yield_to_ego | aggressive_fast, late_brake, swerving |

**Experimental protocol**: Train with `--style_filter nominal`, evaluate with `--style_filter adversarial`. Compare degradation across methods.

**CLI**: `--style_filter nominal` or `--style_filter adversarial`

## 5. State Ablation

```python
SumoEnv(state_ablation=None)              # full state (default)
SumoEnv(state_ablation="no_visibility")   # zero out visibility features
```

When `no_visibility`:
- `alpha_cz = 1.0`, `alpha_cross = 1.0`, `d_occ = 200.0`
- `dt_seen = 0.0`, `sigma_percep = 0.05`, `n_occ = 0.0`
- Physical occlusion still active (agent `nu_i` still varies)
- Tests whether visibility features in the 79D PDE state carry useful information

**CLI**: `--state_ablation no_visibility`

## 6. New SumoEnv Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buildings` | bool | True | Enable/disable static occlusion buildings |
| `style_filter` | str/None | None | Filter agent styles: "nominal", "adversarial" |
| `state_ablation` | str/None | None | Ablation mode: "no_visibility" |

## 7. New CLI Flags (all training + eval scripts)

| Flag | Description |
|------|-------------|
| `--no_buildings` | Disable occlusion buildings |
| `--style_filter {nominal,adversarial}` | Filter agent behavioral styles |
| `--state_ablation {no_visibility}` | Zero out visibility features |
| `--scenario {2_dense,3_dense,4_dense}` | Dense scenario variants (added to choices) |

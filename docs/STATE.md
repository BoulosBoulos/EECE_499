# State Vector: Complete Specification

The observation space is a flat vector built from ego state, geometry, visibility, and per-agent features. Total dimension: **134** (base) or **164** (with intent LSTM) or **135/165** (with pothole).

---

## State Structure

```
s_t = [s_ego | s_geom | s_vis | f^1 | f^2 | f^3 | f^4 | f^5]
```

---

## 1. Ego State (s_ego) — 6 dimensions

| Index | Symbol | Description | Source | Unit |
|-------|--------|-------------|--------|------|
| 0 | v | Ego speed | TraCI `getSpeed` | m/s |
| 1 | a | Ego longitudinal acceleration | TraCI `getAcceleration` | m/s² |
| 2 | ψ̇ | Yaw rate | Placeholder (0) | rad/s |
| 3 | ψ | Ego heading | TraCI `getAngle` → radians | rad |
| 4 | Δa | Change in acceleration (jerk proxy) | a − a_prev | m/s² |
| 5 | Δψ̇ | Change in yaw rate | ψ̇ − ψ̇_prev | rad/s |

**Direct from observation:** v, a, ψ  
**Derived:** Δa, Δψ̇ (from previous step)

---

## 2. Route & Intersection Geometry (s_geom) — 12 dimensions

| Index | Symbol | Description | Source | Unit |
|-------|--------|-------------|--------|------|
| 0 | d_stop | Distance to stop line | lane_len − lane_pos − 5 | m |
| 1 | d_cz | Distance to conflict zone | lane_len − lane_pos − 10 | m |
| 2 | d_exit | Distance to exit | lane_len − lane_pos | m |
| 3 | κ | Path curvature | Fixed 0 (no path model) | 1/m |
| 4 | e_y | Lateral path error | Fixed 0 | m |
| 5 | e_ψ | Heading path error | Fixed 0 | rad |
| 6 | w_lane | Lane width | Fixed 3.5 | m |
| 7–9 | g_turn | Turn intention one-hot | Fixed [0,0,1] (right) | — |
| 10–11 | ρ | Right-of-way context | Fixed [0.5,0.5] | — |

**Direct from observation:** d_stop, d_cz, d_exit (from lane position)  
**Fixed/placeholder:** κ, e_y, e_ψ, g_turn, ρ (no path/ROW model yet)

---

## 3. Visibility (s_vis) — 6 dimensions

| Index | Symbol | Description | Source | Unit |
|-------|--------|-------------|--------|------|
| 0 | α_cz | Conflict zone visibility | Fixed 1.0 | — |
| 1 | α_cross | Crossing visibility | Fixed 1.0 | — |
| 2 | d_occ | Distance to occluder | Fixed 1e6 | m |
| 3 | dt_seen | Time since first seen | Fixed 0 | s |
| 4 | σ_percep | Perception uncertainty | Fixed 0.05 | — |
| 5 | n_occ | Number of occluders | Fixed 0 | — |

**All fixed:** No occlusion model implemented.

---

## 4. Per-Agent Features (f^i) — 22 × 5 = 110 dimensions

For each of the top-5 agents (by time-to-conflict and distance):

| Index | Symbol | Description | Source | Unit |
|-------|--------|-------------|--------|------|
| 0 | Δx | Relative x (ego frame) | R(ψ) @ (p_i − p_e) | m |
| 1 | Δy | Relative y (ego frame) | R(ψ) @ (p_i − p_e) | m |
| 2 | Δv_x | Relative velocity x | R(ψ) @ (v_i − v_e) | m/s |
| 3 | Δv_y | Relative velocity y | R(ψ) @ (v_i − v_e) | m/s |
| 4 | Δψ | Relative heading | wrap(ψ_i − ψ_e) | rad |
| 5 | v_i | Agent speed | TraCI | m/s |
| 6 | a_i | Agent acceleration | TraCI | m/s² |
| 7 | d_cz_i | Agent distance to CZ | ‖p_i − p_e‖ × 0.5 | m |
| 8 | d_exit_i | Agent distance to exit | d_cz_i × 0.8 | m |
| 9 | τ_i | Time to conflict | d_cz_i / v_i | s |
| 10 | Δτ_i | τ_i − τ_e | Derived | s |
| 11 | t_cpa | Time to closest approach | CPA formula | s |
| 12 | d_cpa | Distance at CPA | ‖p_cpa‖ | m |
| 13 | TTC_i | Time to collision proxy | (d_cpa − d_safe) / ‖Δv‖ | s |
| 14 | χ_i | Intent/ROW prob | Placeholder 0.5 | — |
| 15 | π_row_i | ROW probability | Placeholder 0.5 | — |
| 16 | ν_i | Visibility | 1.0 | — |
| 17 | σ_i | Uncertainty | 0.1 | — |
| 18–20 | type_onehot | Agent type | [veh,ped,cyc] one-hot | — |
| 21 | mask | Valid agent flag | 1.0 | — |

**Direct from observation:** p_i, ψ_i, v_i, a_i (TraCI)  
**Derived:** Δx, Δy, Δv, Δψ, τ_i, Δτ_i, t_cpa, d_cpa, TTC_i  
**Placeholder:** χ_i, π_row_i

---

## 5. Intent Features (optional) — 30 dimensions

When `use_intent=True`: 5 agents × 6 = 30 extra dimensions.

| Per agent | Description |
|-----------|-------------|
| 3 | Intent probabilities (yield/proceed/turn) |
| 3 | Style probabilities (soft/mid/aggressive) |

**Source:** IntentStylePredictor LSTM over agent history.

---

## 6. Pothole (optional) — 1 dimension

When scenario has pothole (1d, 4):

| Index | Symbol | Description | Source |
|-------|--------|-------------|--------|
| — | d_pothole | Distance to pothole center | ‖p_ego − p_center‖ |

---

## Total Dimensions

| Configuration | Total |
|---------------|-------|
| Base (no intent, no pothole) | 6 + 12 + 6 + 110 = **134** |
| + Intent LSTM | 134 + 30 = **164** |
| + Pothole (scenarios 1d, 4) | 134 + 1 = **135** or 164 + 1 = **165** |

---

## Direct vs Derived Summary

| Category | Direct from observation | Derived | Fixed/placeholder |
|----------|-------------------------|---------|-------------------|
| s_ego | v, a, ψ | Δa, Δψ̇ | ψ̇ |
| s_geom | d_stop, d_cz, d_exit | — | κ, e_y, e_ψ, w_lane, g_turn, ρ |
| s_vis | — | — | All 6 |
| f_agent | p, ψ, v, a (per agent) | Δx, Δy, Δv, Δψ, τ, Δτ, t_cpa, d_cpa, TTC | χ, π_row, ν, σ |

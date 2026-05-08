# SUMO T-Intersection Environment

## 1. Geometry

The environment models a **priority T-intersection**:

- **Stem**: 60 m south of the junction center (ego approach). Edge: `stem_in`.
- **Bar**: 50 m each side (east-west). Edges: `left_in`, `right_in`, `left_out`, `right_out`.
- **Junction center**: origin `(0, 0)`.
- **Lanes**: 2 per direction, lane width 3.5 m.
- **Priority**: bar traffic has priority over stem traffic. Ego (on stem) must yield.

## 2. Ego Task

The ego vehicle starts at the bottom of the stem (~60 m south of the junction), approaches the intersection, and **turns right** onto `right_out`. The episode is successful when the ego clears the intersection (reaches `right_out` with lane position > `bar_len - 10`).

## 3. Agent Types

### Cars (4 maneuvers x 7 styles)
- **Maneuvers**: `straight_left_right`, `straight_right_left`, `turn_left`, `turn_right`
- **Styles**: `nominal`, `aggressive`, `timid`, `distracted`, `erratic`, `drunk`, `rule_violating`

### Pedestrians (2 maneuvers x 7 styles)
- **Maneuvers**: `cross_left_right`, `cross_right_left`
- **Styles**: `normal_walk`, `running`, `slow_elderly`, `stop_midway`, `hesitant`, `distracted_slow`, `jaywalking_fast`

### Motorcycles (3 maneuvers x 6 styles)
- **Maneuvers**: `straight_right_left`, `straight_left_right`, `turn_into_stem`
- **Styles**: `nominal`, `aggressive_fast`, `cautious`, `late_brake`, `swerving`, `yield_to_ego`

## 4. Conflict Scheduling

Spawn timing is calibrated so agents arrive at the conflict zone approximately when the ego does:

- **Ego ETA**: `ego_approach_dist / ego_approach_speed` (~50m / 6 m/s ~ 8.3s)
- **Conflict offset**: `uniform(-2, 3)` seconds -- creates diversity:
  - Negative: agent arrives before ego (ego must yield)
  - Positive: agent arrives after ego (ego can proceed)
- **Agent depart_pos/depart_time**: solved so agent travel time matches the target ETA
- **Pedestrians**: timed to enter the crosswalk when ego is 5-15 m from the CZ

## 5. Static Occlusion Model

Two building polygons at the T-intersection corners block the ego's view:

- **NW building**: corners `[(-3.5, 3.5), (-30, 3.5), (-30, 20), (-3.5, 20)]` -- blocks view of `left_in` traffic
- **NE building**: corners `[(3.5, 3.5), (30, 3.5), (30, 20), (3.5, 20)]` -- blocks view of `right_in` traffic

**Behavioral consequence**: when the ego is far down the stem (y ~ -60), buildings block its view of bar traffic. As the ego creeps forward (y approaches 0), the viewing angle improves and more of the conflict zone becomes visible. This creates the causal link: **CREEP -> more visibility -> better information -> safer GO decision**.

Line-of-sight (`nu_i`) returns:
- `0.05` if a building blocks the view (static occlusion)
- `0.1 - 1.0` based on dynamic vehicle occlusion
- `1.0` if fully visible

## 6. Observation Model

**134D state vector** (without optional features):

| Block | Dim | Description |
|-------|-----|-------------|
| `s_ego` | 6 | `[v, a, psi_dot, psi, jerk, yaw_accel]` |
| `s_geom` | 12 | `[d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane, g_turn(3), rho(2)]` |
| `s_vis` | 6 | `[alpha_cz, alpha_cross, d_occ, dt_seen, sigma_percep, n_occ]` |
| `f_agents` | 5 x 22 = 110 | Per-agent: `[dx, dy, dvx, dvy, dpsi, v, a, d_cz, d_exit, tau, dtau, t_cpa, d_cpa, TTC, chi, pi_row, nu, sigma, type(3), mask]` |

**Optional additions**:
- `+30` if `use_intent=True` (5 agents x 6 intent/style features from LSTM)
- `+1` if scenario has pothole (`d_pothole`)

**Visibility features**:
- `alpha_cz`: fraction of conflict zone visible (geometric sampling against occlusion polygons)
- `d_occ`: distance from ego to nearest static occlusion boundary toward CZ
- `sigma_percep`: `0.05 + 0.15 * (n_occluded / n_agents)`, range [0.05, 0.20]

## 7. Reward Structure

| Component | Weight | Condition |
|-----------|--------|-----------|
| Progress | `w_prog = 1.0` | Per meter traveled |
| Time penalty | `w_time = -0.1` | Per timestep |
| Risk | `w_risk = -3.0` | When TTC < `ttc_thr` (3.0s) |
| Collision | `w_coll = -20.0` | SUMO collision or proximity < `d_coll` (2.0m) |
| Pothole | `w_pothole = -5.0` | Ego drives over pothole |
| Abort comfort | `w_abort_comfort = -0.5` | ABORT action used |
| Success bonus | `w_success = 10.0` | Terminal: ego clears intersection |
| Action switch | `w_switch = -0.05` | Action differs from previous step |
| ROW violation | `w_rule = -2.0` | Ego enters CZ while priority agent is close |

## 8. Action Space

Discrete, 5 actions:

| Index | Name | TraCI Execution |
|-------|------|-----------------|
| 0 | STOP | Controlled braking: `v - 5.0 * dt` |
| 1 | CREEP | Regulate toward 1.0 m/s |
| 2 | YIELD | Gentle deceleration: `v - 0.5 * dt` |
| 3 | GO | Accelerate at 2.0 m/s^2, max 13.89 m/s |
| 4 | ABORT | Emergency braking: `v - 8.0 * dt` |

## 9. Termination Conditions

| Condition | `terminated` | `success` | `truncated` |
|-----------|-------------|-----------|-------------|
| Collision (SUMO event or proximity) | True | False | False |
| Ego on `right_out` past `bar_len - 10` | True | True | False |
| Ego disappeared without collision | True | True | False |
| Step count >= `max_steps` (500) | False | False | True |

## 10. Scenarios

| Scenario | Car | Pedestrian | Motorcycle | Pothole |
|----------|-----|------------|------------|---------|
| 1a | Yes | No | No | No |
| 1b | No | Yes | No | No |
| 1c | No | No | Yes | No |
| 1d | Yes | Yes | Yes | No |
| 2 | Yes | Yes | No | No |
| 3 | Yes | No | Yes | No |
| 4 | Yes | Yes | Yes | Yes |

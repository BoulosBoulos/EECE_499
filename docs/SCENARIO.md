# Scenarios: SUMO and Synthetic

## SUMO T-Intersection Scenarios (main pipeline)

Eight scenarios with **collisions enabled** and optional pothole:

| Scenario | EGO + | Description |
|----------|-------|-------------|
| **1a** | Other car | Ego + 1 crossing car |
| **1b** | Pedestrian | Ego + 1 pedestrian (sidewalks, crossings) |
| **1c** | Motorcyclist | Ego + 1 motorcyclist |
| **1d** | Pothole | Ego + pothole only |
| **2** | Car + pedestrian | Ego + car + pedestrian |
| **3** | Car + ped + moto | Ego + car + pedestrian + motorcyclist |
| **4** | Car + ped + moto + pothole | Full composite |

### Separate scenarios across the pipeline

```bash
# Train
python experiments/run_train.py --scenario 1a --total_steps 50000
python experiments/run_train.py --scenario 4 --use_intent  # with intent LSTM

# Eval
python experiments/run_eval.py --checkpoint results/model_pinn_1a.pt --scenario 1a
python experiments/run_eval.py --checkpoint results/model_pinn_4.pt --scenario 4

# Visualize
python experiments/run_visualize_sumo.py --scenario 4 --policy go --gui

# HPO per scenario
python experiments/run_hpo.py --per_scenario --n_trials 30

# Ablation study (all scenarios × PINN vs no-PINN)
make ablation
```

### Per-episode aggressiveness sampling

Other vehicles' `jmIgnoreJunctionFoeProb` is sampled per episode from [0, 0.05, 0.1, 0.15, 0.2] to reduce bias. Pass `jm_ignore_fixed=0.1` to SumoEnv to use a fixed value (e.g. for HPO or ablation).

### Scenario generation

Scenarios are generated on first use into `scenarios/sumo_{1a,1b,...,4}/`. **After upgrading**: delete `scenarios/` to regenerate with new vTypes. Or manually:

```python
from scenario.generator import ScenarioGenerator
gen = ScenarioGenerator()
gen.generate("scenarios/sumo_4", scenario_name="4")
```

---

## Synthetic T-Intersection Scenario (legacy)

### What is the data? Where does it come from?

**There is no pre-recorded dataset.** The training data is **experience collected online** from the environment. Each step:

1. The policy observes the current state (ego + agents + geometry).
2. The policy outputs a discrete action (STOP, CREEP, YIELD, GO, ABORT).
3. The environment transitions: ego and agents move, reward is computed.
4. This (state, action, reward, next_state) tuple is stored in a rollout buffer for PPO updates.

So the "data" is **generated on-the-fly** by interacting with the synthetic T-intersection environment. It is **not** loaded from a file.

---

## How is the synthetic scenario created? Based on what?

The scenario is defined in `env/t_intersection_env.py` and modeled after a **simplified T-intersection**:

- **Ego vehicle:** Starts at (0, 0), heading along the x-axis. Must proceed along x toward a "goal" at x ≈ 30 m.
- **Conflict zone (CZ):** Roughly x ∈ [15, 25] — where ego’s path crosses cross-traffic.
- **Cross-traffic agents:** 1–3 vehicles spawning on the left (y = 15 m) or right (y = -15 m), moving perpendicular to ego (along ±y direction depending on side).

**Procedural generation at reset:**
- Number of agents: random in {1, 2, 3}.
- Each agent: random side (left/right), random x in [5, 25], random speed in [0.5, 3.0] m/s.
- Agents are modeled as **vehicles**, not pedestrians or cyclists (no VRUs in the synthetic env).

The design follows a minimal T-intersection: ego on the stem, cross-traffic on the top bar. SUMO will later provide a more detailed scenario; this synthetic setup is a stand-in for it.

---

## Scenario layout

```
        cross-traffic (left)  y=15
        ←───────────────────────
        
        | conflict zone |
        x≈15   ...   x≈25

        ←───────────────────────
        cross-traffic (right)  y=-15

ego →   (0,0) --------→ x=30 (goal)
```

---

## When does an episode end?

An episode ends when **either**:

1. **Max steps:** 200 steps (default).
2. **Goal reached:** Ego’s x-position exceeds 30 m.

```python
done = self._step_count >= self.max_steps or p_new[0] > 30
```

---

## How are cross-traffic agents behaving?

Cross-traffic agents:

- Move at **constant speed** in a **straight line** along their heading (ψ = π/2 for left, ψ = -π/2 for right).
- Do **not** react to ego.
- Are modeled as **vehicles** (`type: "veh"`), not pedestrians or cyclists.

So there are **no VRUs** (vulnerable road users like pedestrians/cyclists) in the synthetic env; only crossing vehicles. VRUs can be added later in the SUMO scenario.

---

## How is ego behaving?

Ego’s behavior is determined by the **discrete action** chosen by the RL policy each step:

| Action | Index | Effect (acceleration) |
|--------|-------|------------------------|
| STOP   | 0     | Hard brake (-5 m/s²) |
| CREEP  | 1     | Gentle accel if v &lt; 1 else gentle decel |
| YIELD  | 2     | Slight decel (-0.5 m/s²) |
| GO     | 3     | Accelerate (+1 m/s²) |
| ABORT  | 4     | Hard brake (-5 m/s²) |

Speed is clipped to [0, 10] m/s. Position is integrated with Euler: `p_new = p + v * dt * [cos(ψ), sin(ψ)]`.

---

## How to see discrete actions the RL outputs

Use the **visualize** script with a trained policy:

```bash
python3 experiments/run_visualize.py --policy checkpoint --checkpoint results/model_pinn_base.pt --episodes 3
```

Outputs:

- **`results/trajectory_ep{N}.csv`** — columns `action` (0–4) and `action_name` (STOP, CREEP, YIELD, GO, ABORT) per step.
- **`results/trajectory_ep{N}_actions.png`** — action index over time (added by the script).

To inspect raw actions in the CSV:

```bash
# Show first 20 steps with action names
cut -d',' -f1,7,8 results/trajectory_ep0.csv | head -21
```

---

## Summary

| Question | Answer |
|----------|--------|
| Where is the training data? | Online experience from the env; no pre-recorded file. |
| How is synthetic data created? | Procedurally at reset: ego + 1–3 cross-traffic vehicles. |
| Based on what? | Simplified T-intersection model (ego on stem, cross-traffic on bar). |
| When does an episode end? | Max 200 steps or ego x &gt; 30 m. |
| How do agents behave? | Constant velocity, straight line; no reaction to ego. |
| VRUs? | None in synthetic env; only vehicles. |
| How does ego behave? | Discrete actions: STOP, CREEP, YIELD, GO, ABORT. |
| How to see discrete actions? | `run_visualize.py --policy checkpoint`, then open trajectory CSV or action plot. |

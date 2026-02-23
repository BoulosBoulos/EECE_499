# End-to-End Pipeline

Pipeline for the SUMO T-intersection behavioral decision model (DM).

## Pipeline Stages

```
Perception → State → Intent/Style → Policy → Shield → Action → Reward
```

### 1. Perception
- Raw observations from SUMO via TraCI (positions, speeds, lanes, signals)
- Occlusion handling: mark agents as visible/occluded based on geometry and occluders

### 2. State
- Build structured state from perception
- See `docs/STATE_SCHEMA.md` for feature groups, top-K agents, masks, normalization

### 3. Intent/Style (optional)
- High-level intent encoding (e.g., turn direction, aggressiveness)
- Can be learned or rule-based

### 4. Policy
- Pluggable RL algorithm (PPO, SAC, etc.)
- Maps state → discrete behavior: STOP, CREEP, YIELD, GO, ABORT

### 5. Shield
- Safety layer: override policy output if collision imminent
- Ensures feasible behaviors given partial observability

### 6. Action
- Execute selected behavior (low-level control in SUMO)

### 7. Reward
- Sparse/dense reward from `configs/reward/`
- Physics-informed critic: auxiliary residual losses (no HJB)

## Data Flow

- **Configs**: `configs/scenario/`, `configs/state/`, `configs/algo/`, `configs/residuals/`, `configs/reward/`
- **Modules**: `scenario/generator.py`, `state/builder.py`, `env/sumo_env.py`, `rl/interface.py`

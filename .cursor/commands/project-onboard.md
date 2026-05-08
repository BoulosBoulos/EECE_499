# Project Onboard

Get up to speed with the SUMO T-intersection behavioral DM project.

## Quick Start
1. Read `docs/RUNNING.md` for setup, train, and eval commands.
2. Read `docs/PIPELINE.md` for the end-to-end pipeline.
3. Read `docs/STATE_SCHEMA.md` for the state representation.

## Config Layout
- `configs/scenario/` — SUMO scenario generation
- `configs/state/` — state builder and feature groups
- `configs/algo/` — RL algorithm (pluggable)
- `configs/residuals/` — physics-informed auxiliary losses
- `configs/reward/` — reward shaping

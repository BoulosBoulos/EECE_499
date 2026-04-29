.PHONY: setup train-hjb-aux train-soft-hjb-aux train-eikonal-aux train-cbf-aux train-drppo \
        eval-pde ablation-pde ablation-tier1 ablation-tier2 ablation-tier3 \
        plot-pde visualize-pde-gui regen-scenarios clean

PYTHON ?= python3
SCENARIO ?= 1a
MANEUVER ?= stem_right
STEPS ?= 50000
SEED ?= 42
PARALLEL ?= 32
METHOD ?= hjb_aux

SCENARIOS    ?= 1a 4_dense
MANEUVERS    ?= stem_right stem_right
SEEDS        ?= 42 123 456
MAX_PARALLEL ?= 5

setup:
	pip install -r requirements.txt --break-system-packages
	@echo "Set SUMO_HOME and PYTHONPATH before running experiments."

regen-scenarios:
	$(PYTHON) -c "from scenario.generator import ScenarioGenerator; \
		g = ScenarioGenerator(); \
		[g.generate(f'scenarios/sumo_{s}', s) for s in ['1a','1b','1c','1d','2','3','4']]"

# ── Single training runs ────────────────────────────────────────────────

train-hjb-aux:
	$(PYTHON) experiments/pde/train_hjb_aux.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --total_steps $(STEPS) --seed $(SEED)

train-soft-hjb-aux:
	$(PYTHON) experiments/pde/train_soft_hjb_aux.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --total_steps $(STEPS) --seed $(SEED)

train-eikonal-aux:
	$(PYTHON) experiments/pde/train_eikonal_aux.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --total_steps $(STEPS) --seed $(SEED)

train-cbf-aux:
	$(PYTHON) experiments/pde/train_cbf_aux.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --total_steps $(STEPS) --seed $(SEED)

train-drppo:
	$(PYTHON) experiments/pde/train_drppo_baseline.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --total_steps $(STEPS) --seed $(SEED)

# ── Evaluation ──────────────────────────────────────────────────────────

eval-pde:
	$(PYTHON) experiments/pde/eval.py --checkpoint $(CHECKPOINT) --method $(METHOD) --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --episodes 50 --save_failures

# ── Ablation tiers ──────────────────────────────────────────────────────

ablation-tier1:
	$(PYTHON) experiments/pde/run_full_ablation.py --tier 1 --max_parallel $(PARALLEL) --total_steps $(STEPS)

ablation-tier2:
	$(PYTHON) experiments/pde/run_full_ablation.py --tier 2 --max_parallel $(PARALLEL) --total_steps $(STEPS)

ablation-tier3:
	$(PYTHON) experiments/pde/run_full_ablation.py --tier 3 --max_parallel $(PARALLEL) --total_steps $(STEPS)

ablation-all:
	$(PYTHON) experiments/pde/run_full_ablation.py --tier all --max_parallel $(PARALLEL) --total_steps $(STEPS)

# ── Serial ablation (single-process) ───────────────────────────────────

ablation-serial:
	$(PYTHON) experiments/pde/run_ablation.py --total_steps $(STEPS) --ego_maneuver $(MANEUVER)

# ── Visualization ───────────────────────────────────────────────────────

visualize-pde-gui:
	$(PYTHON) experiments/pde/visualize_sumo.py --gui --checkpoint $(CHECKPOINT) --method $(METHOD) --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --episodes 3

# ── Plotting ────────────────────────────────────────────────────────────

plot-pde:
	$(PYTHON) experiments/pde/plot_pde.py --pde_dir results/ablation --out_dir results/ablation

# ── Calibration ────────────────────────────────────────────────────

calibrate:
	$(PYTHON) experiments/pde/run_calibration.py \
	    --scenarios $(SCENARIOS) \
	    --ego_maneuvers $(MANEUVERS) \
	    --seeds $(SEEDS) \
	    --steps $(STEPS) \
	    --out_dir results/calibration \
	    --max_parallel $(MAX_PARALLEL)

calibrate-analyze:
	$(PYTHON) experiments/pde/run_calibration.py --scenario $(SCENARIO) --ego_maneuver $(MANEUVER) --analyze_only

# ── Cleanup ─────────────────────────────────────────────────────────────

clean:
	rm -rf results/ablation/tier*/
	rm -rf results/pde/*.csv results/pde/*.pt results/pde/*.json

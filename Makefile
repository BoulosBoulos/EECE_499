.PHONY: setup train train-1a train-1b train-1c train-1d train-2 train-3 train-4 eval eval-multiseed eval-1a eval-1b eval-1c eval-1d eval-2 eval-3 eval-4 hpo ablation ablation-sensitivity ablation-full jobs-manifest jobs-run-one ablation-16gpu ablation-aggregate jobs-manifest-batch1 jobs-manifest-batch2 jobs-manifest-batch1-no-intent jobs-manifest-batch2-no-intent jobs-manifest-batch1-intent jobs-manifest-batch2-intent ablation-16gpu-batch1 ablation-16gpu-batch2 ablation-aggregate-batch1 ablation-aggregate-batch2 ablation-merge plot-ablation visualize-ablation visualize visualize-gui plot lint train-intent regen-scenarios dashboard train-hjb-aux train-soft-hjb-aux train-hjb-aux-all train-soft-hjb-aux-all train-pde-all eval-pde eval-pde-all pde-train-eval-all ablation-pde plot-pde visualize-pde-gui plot-interaction regen-interaction-scenarios interaction-rule-baseline interaction-train interaction-train-all interaction-train-pinn interaction-eval interaction-eval-pinn interaction-viz-gui interaction-viz-all interaction-plot interaction-full interaction-curriculum interaction-manifest interaction-train-hjb-aux interaction-train-soft-hjb-aux interaction-train-hjb-aux-all interaction-train-soft-hjb-aux-all interaction-train-pde-all interaction-test

PYTHONPATH := $(CURDIR)
export PYTHONPATH

setup:
	rm -rf .venv && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

regen-scenarios:
	rm -rf scenarios && python3 -c "from scenario.generator import ScenarioGenerator; g=ScenarioGenerator(); [g.generate(f'scenarios/sumo_{s}', s) for s in ['1a','1b','1c','1d','2','3','4']]"

# ─── Training ──────────────────────────────────────────────────────────────

train:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000

train-1a train-1b train-1c train-1d train-2 train-3 train-4:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000 --scenario $(patsubst train-%,%,$@)

train-actor:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000 --pinn_placement actor

train-both:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000 --pinn_placement both

train-safety:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000 --use_safety_filter

train-intent:
	python3 experiments/train_intent.py --n_episodes 200 --n_epochs 50 --scenario 3

# ─── Evaluation ────────────────────────────────────────────────────────────

eval:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_1a.pt --episodes 100 --deterministic

eval-multiseed:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_1a.pt --episodes 50 --seeds 42 123 456 789 --deterministic

eval-stochastic:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_1a.pt --episodes 50 --seeds 42 123 456 --stochastic

eval-1a eval-1b eval-1c eval-1d eval-2 eval-3 eval-4:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_$(patsubst eval-%,%,$@).pt --scenario $(patsubst eval-%,%,$@) --episodes 100 --deterministic --out results/eval_$(patsubst eval-%,%,$@).csv

# ─── Ablation Studies ──────────────────────────────────────────────────────

ablation:
	python3 experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation --seeds 42 123 456 789 999

ablation-sensitivity:
	python3 experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --variants nopinn pinn_critic pinn_actor pinn_both

ablation-safety:
	python3 experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation --seeds 42 123 456 789 999 --variants safety_filter pinn_critic_sf nopinn pinn_critic

ablation-full:
	python3 experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --eval_stochastic

# ─── 16-GPU parallel ablation ───────────────────────────────────────────────
# 1) Generate job manifest; 2) Launch 16 workers; 3) Aggregate CSVs for dashboard
jobs-manifest:
	python3 experiments/generate_jobs.py --out_dir results/ablation --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0

jobs-run-one:
	CUDA_VISIBLE_DEVICES=0 python3 experiments/run_single_job.py --manifest results/ablation/job_manifest.json --job_id 0 --out_dir results/ablation

ablation-16gpu:
	./scripts/launch_parallel_16gpu.sh --manifest results/ablation/job_manifest.json --num_gpus 16

ablation-aggregate:
	python3 experiments/aggregate_results.py --out_dir results/ablation --manifest results/ablation/job_manifest.json

# ─── 16-GPU split into 2 runs by scenario (each run fits in ~48h) ───────────
# Batch 1: scenarios 1a, 1b, 1c, 1d.  Batch 2: scenarios 2, 3, 4.
# Default: every combo runs WITH and WITHOUT LSTM (--intent_ablation). Prerequisite: make train-intent.
jobs-manifest-batch1:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --intent_ablation

jobs-manifest-batch2:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --intent_ablation

# Optional: only no-LSTM or only LSTM (subset of full ablation)
jobs-manifest-batch1-no-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0

jobs-manifest-batch2-no-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0

jobs-manifest-batch1-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --use_intent

jobs-manifest-batch2-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --use_intent

ablation-16gpu-batch1:
	./scripts/launch_parallel_16gpu.sh --manifest results/ablation_batch1/job_manifest.json --num_gpus 16 --eval_stochastic

ablation-16gpu-batch2:
	./scripts/launch_parallel_16gpu.sh --manifest results/ablation_batch2/job_manifest.json --num_gpus 16 --eval_stochastic

ablation-aggregate-batch1:
	python3 experiments/aggregate_results.py --out_dir results/ablation_batch1 --manifest results/ablation_batch1/job_manifest.json

ablation-aggregate-batch2:
	python3 experiments/aggregate_results.py --out_dir results/ablation_batch2 --manifest results/ablation_batch2/job_manifest.json

ablation-merge:
	python3 experiments/merge_ablation_batches.py --batch1 results/ablation_batch1 --batch2 results/ablation_batch2 --out_dir results/ablation

# ─── Visualization ─────────────────────────────────────────────────────────

visualize:
	python3 experiments/run_visualize_sumo.py --episodes 3 --out_dir results

visualize-gui:
	python3 experiments/run_visualize_sumo.py --episodes 3 --gui --out_dir results

hpo:
	python3 experiments/run_hpo.py --n_trials 50 --total_steps 10000 --out_dir results/hpo

plot-ablation:
	python3 experiments/plot_ablation.py --csv results/ablation/ablation_results.csv --out_dir results/ablation

visualize-ablation:
	python3 experiments/run_visualize_ablation.py --ablation_dir results/ablation --out_dir results/ablation_viz --episodes 2

plot:
	python3 experiments/plot_comparison.py --dir results

dashboard:
	streamlit run experiments/dashboard.py --server.port 8501

# ─── PDE-Family Methods ────────────────────────────────────────────────────
# All-scenario loops: override PDE_STEPS, PDE_OUT_DIR, PDE_EVAL_EPISODES, PDE_EVAL_SEEDS as needed.
# Example: make train-pde-all PDE_STEPS=100000
# Optional: PDE_TRAIN_EXTRA="--use_intent --seed 42"  PDE_EVAL_EXTRA="--use_intent" (must match)

PDE_SCENARIOS := 1a 1b 1c 1d 2 3 4
PDE_STEPS ?= 50000
PDE_OUT_DIR ?= results/pde
PDE_EVAL_EPISODES ?= 50
PDE_EVAL_SEEDS ?= 42 123 456
PDE_TRAIN_EXTRA ?=
PDE_EVAL_EXTRA ?=

train-hjb-aux:
	python3 experiments/pde/train_hjb_aux.py --scenario 1a --total_steps 50000

train-soft-hjb-aux:
	python3 experiments/pde/train_soft_hjb_aux.py --scenario 1a --total_steps 50000

train-hjb-aux-all:
	@for s in $(PDE_SCENARIOS); do \
		echo "=== [PDE] HJB aux train scenario $$s ($(PDE_STEPS) steps) ==="; \
		python3 experiments/pde/train_hjb_aux.py --scenario $$s --total_steps $(PDE_STEPS) --out_dir $(PDE_OUT_DIR) $(PDE_TRAIN_EXTRA) || exit 1; \
	done

train-soft-hjb-aux-all:
	@for s in $(PDE_SCENARIOS); do \
		echo "=== [PDE] Soft-HJB aux train scenario $$s ($(PDE_STEPS) steps) ==="; \
		python3 experiments/pde/train_soft_hjb_aux.py --scenario $$s --total_steps $(PDE_STEPS) --out_dir $(PDE_OUT_DIR) $(PDE_TRAIN_EXTRA) || exit 1; \
	done

train-pde-all: train-hjb-aux-all train-soft-hjb-aux-all

eval-pde:
	python3 experiments/pde/eval.py --checkpoint results/pde/model_hjb_aux_1a.pt --method hjb_aux --scenario 1a --episodes 50 --seeds 42 123

eval-pde-all:
	@for s in $(PDE_SCENARIOS); do \
		echo "=== [PDE] eval HJB aux scenario $$s ==="; \
		python3 experiments/pde/eval.py --checkpoint $(PDE_OUT_DIR)/model_hjb_aux_$$s.pt --method hjb_aux --scenario $$s --episodes $(PDE_EVAL_EPISODES) --seeds $(PDE_EVAL_SEEDS) --out_dir $(PDE_OUT_DIR) $(PDE_EVAL_EXTRA) || exit 1; \
		echo "=== [PDE] eval Soft-HJB aux scenario $$s ==="; \
		python3 experiments/pde/eval.py --checkpoint $(PDE_OUT_DIR)/model_soft_hjb_aux_$$s.pt --method soft_hjb_aux --scenario $$s --episodes $(PDE_EVAL_EPISODES) --seeds $(PDE_EVAL_SEEDS) --out_dir $(PDE_OUT_DIR) $(PDE_EVAL_EXTRA) || exit 1; \
	done

# Train both PDE variants on every scenario, then evaluate every checkpoint (14 trains + 14 eval runs).
pde-train-eval-all: train-pde-all eval-pde-all

ablation-pde:
	python3 experiments/pde/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/pde_ablation --seeds 42 123 456 789 999

plot-pde:
	python3 experiments/pde/plot_pde.py --pde_dir results/pde_ablation --legacy_dir results/ablation --out_dir results/pde_ablation

visualize-pde-gui:
	python3 experiments/pde/visualize_sumo.py --gui --episodes 1 --scenario 1a --method hjb_aux --checkpoint results/pde/model_hjb_aux_1a.pt

plot-interaction:
	python3 experiments/pde/plot_interaction.py --csv results/pde/pde_trajectory_hjb_aux_ep0.csv --out_dir results/pde

# ─── Interaction Benchmark (v2) ────────────────────────────────────────────
# Template-driven, conflict-centric behavioural decision benchmark.
# See docs/BEHAVIORAL_DECISION_BENCHMARK_REDESIGN.md

INT_SCENARIOS := 1a 1b 1c 1d 2 3 4
INT_OUT := results/interaction
INT_STEPS ?= 50000
INT_EVAL_EPS ?= 50
INT_EVAL_SEEDS ?= 42 123 456

regen-interaction-scenarios:
	rm -rf scenarios/interaction_* && python3 -c "from scenario.generator_v2 import InteractionScenarioGenerator; g=InteractionScenarioGenerator(); [g.generate(f'scenarios/interaction_{s}', s) for s in ['1a','1b','1c','1d','2','3','4']]"

# Rule-based baseline
interaction-rule-baseline:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Rule baseline scenario $$s ==="; \
		python3 experiments/interaction/rule_baseline.py --scenario $$s --episodes $(INT_EVAL_EPS) --seed 42 --out_dir $(INT_OUT) || exit 1; \
	done

# Training
interaction-train:
	python3 experiments/interaction/run_train.py --scenario 1a --total_steps $(INT_STEPS) --out_dir $(INT_OUT)

interaction-train-all:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction train nopinn scenario $$s ==="; \
		python3 experiments/interaction/run_train.py --scenario $$s --total_steps $(INT_STEPS) --out_dir $(INT_OUT) --pinn_placement none || exit 1; \
	done

interaction-train-pinn:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction train pinn_critic scenario $$s ==="; \
		python3 experiments/interaction/run_train.py --scenario $$s --total_steps $(INT_STEPS) --out_dir $(INT_OUT) --pinn_placement critic || exit 1; \
	done

# Evaluation
interaction-eval:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction eval nopinn scenario $$s ==="; \
		python3 experiments/interaction/run_eval.py --checkpoint $(INT_OUT)/model_interaction_nopinn_$$s.pt --scenario $$s --variant nopinn --episodes $(INT_EVAL_EPS) --seeds $(INT_EVAL_SEEDS) --out_dir $(INT_OUT) --deterministic || exit 1; \
	done

interaction-eval-pinn:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction eval pinn_critic scenario $$s ==="; \
		python3 experiments/interaction/run_eval.py --checkpoint $(INT_OUT)/model_interaction_critic_$$s.pt --scenario $$s --variant pinn_critic --episodes $(INT_EVAL_EPS) --seeds $(INT_EVAL_SEEDS) --out_dir $(INT_OUT) --deterministic || exit 1; \
	done

# Visualization
interaction-viz-gui:
	python3 experiments/interaction/visualize_sumo.py --gui --scenario 1a --policy rule --episodes 3

interaction-viz-all:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction visualize scenario $$s ==="; \
		python3 experiments/interaction/visualize_sumo.py --scenario $$s --policy rule --episodes 1 --out_dir $(INT_OUT) || exit 1; \
	done

# Plots
interaction-plot:
	python3 experiments/interaction/plot_metrics.py --results_csv $(INT_OUT)/interaction_eval_results.csv --train_dir $(INT_OUT) --out_dir $(INT_OUT)

# Curriculum training
interaction-curriculum:
	python3 experiments/interaction/curriculum_train.py --out_dir $(INT_OUT)/curriculum --seed 42

# Frozen eval manifest
interaction-manifest:
	python3 experiments/interaction/eval_manifest.py --scenarios $(INT_SCENARIOS) --episodes 20 --seeds 42 123 456 789 999 --out $(INT_OUT)/eval_manifest.json

# PDE on interaction benchmark
interaction-train-hjb-aux:
	python3 experiments/interaction/train_hjb_aux.py --scenario 1a --total_steps $(INT_STEPS) --out_dir $(INT_OUT)

interaction-train-soft-hjb-aux:
	python3 experiments/interaction/train_soft_hjb_aux.py --scenario 1a --total_steps $(INT_STEPS) --out_dir $(INT_OUT)

interaction-train-hjb-aux-all:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction HJB-aux scenario $$s ==="; \
		python3 experiments/interaction/train_hjb_aux.py --scenario $$s --total_steps $(INT_STEPS) --out_dir $(INT_OUT) || exit 1; \
	done

interaction-train-soft-hjb-aux-all:
	@for s in $(INT_SCENARIOS); do \
		echo "=== Interaction Soft-HJB-aux scenario $$s ==="; \
		python3 experiments/interaction/train_soft_hjb_aux.py --scenario $$s --total_steps $(INT_STEPS) --out_dir $(INT_OUT) || exit 1; \
	done

interaction-train-pde-all: interaction-train-hjb-aux-all interaction-train-soft-hjb-aux-all

# Tests
interaction-test:
	python3 -m pytest tests/ -v --tb=short 2>/dev/null || python3 -m unittest discover -s tests -v

# Full pipeline: generate scenarios, run rule baseline, train, eval, plot
interaction-full: regen-interaction-scenarios interaction-rule-baseline interaction-train-all interaction-eval interaction-plot

lint:
	ruff check . 2>/dev/null || true

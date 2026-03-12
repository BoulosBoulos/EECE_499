.PHONY: setup train train-1a train-1b train-1c train-1d train-2 train-3 train-4 eval eval-multiseed eval-1a eval-1b eval-1c eval-1d eval-2 eval-3 eval-4 hpo ablation ablation-sensitivity ablation-full jobs-manifest jobs-run-one ablation-16gpu ablation-aggregate jobs-manifest-batch1 jobs-manifest-batch2 jobs-manifest-batch1-intent jobs-manifest-batch2-intent jobs-manifest-batch1-intent-ablation jobs-manifest-batch2-intent-ablation ablation-16gpu-batch1 ablation-16gpu-batch2 ablation-aggregate-batch1 ablation-aggregate-batch2 ablation-merge plot-ablation visualize-ablation visualize visualize-gui plot lint train-intent regen-scenarios dashboard

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
# After both: make ablation-aggregate-batch1, make ablation-aggregate-batch2, then make ablation-merge.
# Prerequisite for intent: make train-intent (writes results/intent_model.pt).
jobs-manifest-batch1:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0

jobs-manifest-batch2:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0

jobs-manifest-batch1-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --use_intent

jobs-manifest-batch2-intent:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --use_intent

jobs-manifest-batch1-intent-ablation:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch1 --scenarios 1a 1b 1c 1d --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --intent_ablation

jobs-manifest-batch2-intent-ablation:
	python3 experiments/generate_jobs.py --out_dir results/ablation_batch2 --scenarios 2 3 4 --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --intent_ablation

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

lint:
	ruff check . 2>/dev/null || true

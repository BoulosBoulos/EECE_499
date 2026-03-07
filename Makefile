.PHONY: setup train train-1a train-1b train-1c train-1d train-2 train-3 train-4 eval eval-1a eval-1b eval-1c eval-1d eval-2 eval-3 eval-4 hpo ablation visualize visualize-gui plot lint

PYTHONPATH := $(CURDIR)
export PYTHONPATH

setup:
	rm -rf .venv && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

train:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000

train-1a train-1b train-1c train-1d train-2 train-3 train-4:
	python3 experiments/run_train.py --config configs/algo/default.yaml --total_steps 50000 --scenario $(patsubst train-%,%,$@)

eval:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_1a.pt --episodes 100

eval-1a eval-1b eval-1c eval-1d eval-2 eval-3 eval-4:
	python3 experiments/run_eval.py --checkpoint results/model_pinn_$(patsubst eval-%,%,$@).pt --scenario $(patsubst eval-%,%,$@) --episodes 100 --out results/eval_$(patsubst eval-%,%,$@).csv

visualize:
	python3 experiments/run_visualize_sumo.py --episodes 3 --out_dir results

visualize-gui:
	python3 experiments/run_visualize_sumo.py --episodes 3 --gui --out_dir results

hpo:
	python3 experiments/run_hpo.py --n_trials 50 --total_steps 10000 --out_dir results/hpo

ablation:
	python3 experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation

plot:
	python3 experiments/plot_comparison.py --dir results

lint:
	ruff check . 2>/dev/null || true

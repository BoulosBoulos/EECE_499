.PHONY: setup train eval train-compare lint

PYTHONPATH := $(CURDIR)
export PYTHONPATH

setup:
	rm -rf .venv && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

train:
	python3 experiments/run_train.py --config configs/algo/default.yaml

train-compare:
	python3 experiments/run_train.py --config configs/algo/default.yaml --compare --total_steps 50000

eval:
	python3 experiments/run_eval.py --checkpoint results/model_pinn.pt --episodes 100

lint:
	ruff check . 2>/dev/null || true

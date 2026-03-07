# Running HPO on Google Colab

## Quick start (notebook)

1. Upload the project to Colab (zip + unzip or git clone).
2. Open `notebooks/colab_hpo.ipynb` in Colab.
3. Run all cells.

## Manual setup

1. **Upload or clone the project** into Colab (e.g. upload as zip and unzip, or clone from git).

2. **Install SUMO and dependencies** in a cell:
   ```python
   # Cell 1: Install SUMO
   !add-apt-repository ppa:sumo/stable -y 2>/dev/null || true
   !apt-get update -y
   !apt-get install -y sumo sumo-tools sumo-doc
   %env SUMO_HOME=/usr/share/sumo
   import sys
   sys.path.append("/usr/share/sumo/tools")
   ```

3. **Install Python deps**:
   ```python
   # Cell 2: Python deps
   !pip install -q traci pyyaml matplotlib optuna gymnasium numpy torch
   ```

4. **Set working directory** (if you uploaded to `/content/EECE 499`):
   ```python
   import os
   os.chdir("/content/EECE 499")  # or your project path
   %env PYTHONPATH=/content/EECE 499
   ```

5. **Run HPO**:
   ```python
   # Cell 3: HPO
   !python experiments/run_hpo.py --n_trials 20 --total_steps 3000 --out_dir results/hpo
   ```

## Notes

- Colab sessions are ephemeral; save `results/hpo/hpo_results.json` to Drive if needed.
- Use GPU runtime for faster training (`Runtime > Change runtime type > GPU`).
- SUMO GUI (`sumo-gui`) will not work in Colab; use headless `sumo` (default in the env).

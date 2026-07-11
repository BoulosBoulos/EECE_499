#!/bin/bash
# Job 1 (blocking diagnostic): replay episodes of the nominal-trained 1b/stem_right
# checkpoint under verbose logging, to decide whether SR=0 on 1b/2/4 stem_right is an
# eval-config artifact or a genuine difficulty.
#
# Runs three conditions on the SAME drppo checkpoint:
#   (1) 1b/stem_right, style_filter=nominal   <- the failing condition
#   (2) 1b/stem_right, style_filter=None      <- T1's setting (scored ~22%)
#   (3) 1a/stem_right, style_filter=nominal   <- working control (~99%)
# and repeats (1) with hjb_aux to confirm it is method-independent.
#
# Output: logs/job1_diagnostic_%j.out  (verbose per-episode style + termination)

#SBATCH --job-name=job1_diag
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job1_diagnostic_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/EECE_499/logs/job1_diagnostic_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00

set -uo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/EECE_499
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
export SUMO_HOME=$(.venv/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
export PYTHONUNBUFFERED=1

T3=$REPO/results/tier_3_full/tier3_behav
DRPPO_1B="$T3/1b_stem_right_drppo_nominal_s42/model_drppo_1b_stem_right.pt"
HJB_1B="$T3/1b_stem_right_hjb_aux_nominal_s42/model_hjb_aux_1b_stem_right.pt"
DRPPO_1A="$T3/1a_stem_right_drppo_nominal_s123/model_drppo_1a_stem_right.pt"

echo "############ JOB 1 DIAGNOSTIC  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(hostname) ############"

echo; echo "########## (1) 1b/stem_right  drppo  style_filter=nominal  (FAILING CONDITION) ##########"
python3 experiments/pde/job1_diagnostic.py \
    --checkpoint "$DRPPO_1B" --method drppo --scenario 1b --maneuver stem_right \
    --style_filter nominal --n_episodes 10

echo; echo "########## (2) 1b/stem_right  drppo  style_filter=None  (T1 SETTING, ~22% SR) ##########"
python3 experiments/pde/job1_diagnostic.py \
    --checkpoint "$DRPPO_1B" --method drppo --scenario 1b --maneuver stem_right \
    --style_filter none --n_episodes 10

echo; echo "########## (3) 1a/stem_right  drppo  style_filter=nominal  (WORKING CONTROL, ~99% SR) ##########"
python3 experiments/pde/job1_diagnostic.py \
    --checkpoint "$DRPPO_1A" --method drppo --scenario 1a --maneuver stem_right \
    --style_filter nominal --n_episodes 10

echo; echo "########## (4) 1b/stem_right  hjb_aux  style_filter=nominal  (METHOD-INDEPENDENCE CHECK) ##########"
python3 experiments/pde/job1_diagnostic.py \
    --checkpoint "$HJB_1B" --method hjb_aux --scenario 1b --maneuver stem_right \
    --style_filter nominal --n_episodes 10

echo; echo "############ JOB 1 DIAGNOSTIC COMPLETE  $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"

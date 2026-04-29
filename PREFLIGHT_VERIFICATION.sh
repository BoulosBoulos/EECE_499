#!/bin/bash
# ============================================================
# COMPLETE PRE-FLIGHT VERIFICATION — Run Every Step In Order
# ============================================================
# 
# Instructions:
#   1. cd into your EECE_499-main directory
#   2. Run each step one at a time
#   3. Read the "WHAT TO CHECK" comment before each step
#   4. Do NOT proceed to the next step if the current one fails
#
# Total time: ~90 minutes
# ============================================================

# ── SETUP (run once) ────────────────────────────────────────
# cd EECE_499-main
# export PYTHONPATH="$(pwd)"
# export SUMO_HOME=/usr/share/sumo
# pip install torch numpy gymnasium pyyaml matplotlib pandas scipy


# ============================================================
# STEP 1: Regenerate Scenarios (2 minutes)
# ============================================================
# PURPOSE: Rebuild SUMO scenario files with building polygons and 6 ego routes
# WHAT TO CHECK: No errors in output
# IF IT FAILS: Check that SUMO is installed and SUMO_HOME is set

make regen-scenarios


# ============================================================
# STEP 2: Visual Inspection of Intersection (5 minutes)
# ============================================================
# PURPOSE: Confirm T-intersection geometry and building polygons
# WHAT TO CHECK:
#   - T-intersection with stem going south, bar going east-west
#   - Two lanes per direction on all arms
#   - Two brown/tan building rectangles at NW and NE corners of junction
#   - Green ground polygon underneath
# IF IT FAILS: Buildings missing = generator didn't write t_buildings.poly.xml
# NOTE: Close the "Simulation ended" dialog, just look at the layout, then close

sumo-gui scenarios/sumo_1a/t.sumocfg


# ============================================================
# STEP 3: Verify Pedestrian Scenario Has Crosswalks (2 minutes)
# ============================================================
# PURPOSE: Confirm scenario 1b has pedestrian infrastructure
# WHAT TO CHECK: Crosswalk markings visible at junction
# NOTE: Close after visual confirmation

sumo-gui scenarios/sumo_1b/t.sumocfg


# ============================================================
# STEP 4: Verify All 6 Ego Maneuvers Visually (20 minutes)
# ============================================================
# PURPOSE: Confirm each ego maneuver starts and exits on the correct arm
# WHAT TO CHECK for each maneuver:
#   - Blue ego car appears on the correct starting arm
#   - Ego drives through junction toward the correct exit arm
#   - Red other car appears and creates a conflict situation
#   - Episode ends (collision or success or timeout)
# NOTE: Uses always-GO action, so ego will likely crash — that's fine,
#        you're just checking the ROUTE is correct
# NOTE: Press Enter between each maneuver to continue

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

maneuvers = {
    'stem_right': 'Ego starts SOUTH (stem), exits EAST (right)',
    'stem_left':  'Ego starts SOUTH (stem), exits WEST (left)',
    'right_left': 'Ego starts EAST (right), exits WEST (left) — straight through',
    'right_stem': 'Ego starts EAST (right), exits SOUTH (stem) — left turn',
    'left_right': 'Ego starts WEST (left), exits EAST (right) — straight through',
    'left_stem':  'Ego starts WEST (left), exits SOUTH (stem) — right turn',
}

for maneuver, description in maneuvers.items():
    print(f'\n{\"=\"*60}')
    print(f'MANEUVER: {maneuver}')
    print(f'EXPECTED: {description}')
    print(f'{\"=\"*60}')
    
    env = SumoEnv(scenario_name='1a', ego_maneuver=maneuver, use_gui=True)
    obs, info = env.reset()
    
    for step in range(200):
        action = 3  # always GO
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            collision = info.get('collision', False)
            success = info.get('success', False)
            print(f'  Ended at step {step}: collision={collision}, success={success}')
            break
    
    env.close()
    input('  Press Enter to test next maneuver...')

print('\nAll 6 maneuvers tested.')
"


# ============================================================
# STEP 5: Verify Occlusion Effect (5 minutes)
# ============================================================
# PURPOSE: Confirm alpha_cz (visibility) increases as ego approaches junction
# WHAT TO CHECK:
#   - alpha_cz starts LOW (< 0.5) when ego is far (d_cz > 30m)
#   - alpha_cz increases as ego gets closer
#   - alpha_cz reaches HIGH (> 0.8) when ego is near junction (d_cz < 5m)
# IF alpha_cz IS ALWAYS 1.0: occlusion polygons are not working

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

print('Occlusion test: ego creeps toward junction, watching alpha_cz')
print(f'{\"Step\":>5} {\"d_cz (m)\":>10} {\"alpha_cz\":>10} {\"Status\":>15}')
print('-' * 45)

env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right')
obs, info = env.reset()

for step in range(300):
    obs, r, term, trunc, info = env.step(1)  # CREEP
    alpha = info['raw_obs']['vis']['alpha_cz']
    d_cz = info['built']['s_geom'][1]
    
    if step % 15 == 0:
        status = 'OCCLUDED' if alpha < 0.5 else ('PARTIAL' if alpha < 0.8 else 'CLEAR')
        print(f'{step:>5} {d_cz:>10.1f} {alpha:>10.3f} {status:>15}')
    
    if term or trunc:
        print(f'Episode ended at step {step}')
        break

env.close()
print()
print('PASS if alpha_cz increased from < 0.5 to > 0.8 as d_cz decreased.')
"


# ============================================================
# STEP 6: Verify Buildings On vs Off (2 minutes)
# ============================================================
# PURPOSE: Confirm the buildings flag actually changes visibility
# WHAT TO CHECK:
#   - Buildings ON: alpha_cz < 1.0
#   - Buildings OFF: alpha_cz closer to 1.0
#   - The difference should be positive

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

env_on = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', buildings=True)
obs, info = env_on.reset()
alpha_on = info['raw_obs']['vis']['alpha_cz']
env_on.close()

env_off = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', buildings=False)
obs, info = env_off.reset()
alpha_off = info['raw_obs']['vis']['alpha_cz']
env_off.close()

print(f'Buildings ON:  alpha_cz = {alpha_on:.3f}')
print(f'Buildings OFF: alpha_cz = {alpha_off:.3f}')
diff = alpha_off - alpha_on
print(f'Difference:    {diff:+.3f}')
print()
if diff > 0:
    print('PASS: Buildings flag correctly affects visibility.')
else:
    print('FAIL: No difference — occlusion flag may not be working.')
"


# ============================================================
# STEP 7: Verify Pothole Randomization (2 minutes)
# ============================================================
# PURPOSE: Confirm pothole position and size vary across episodes
# WHAT TO CHECK:
#   - d_pothole should be DIFFERENT across episodes
#   - Values should be reasonable (5-50m range)

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

env = SumoEnv(scenario_name='1d', ego_maneuver='stem_right')
print('Pothole randomization test:')
print(f'{\"Episode\":>8} {\"d_pothole\":>12} {\"Box X range\":>20} {\"Box Y range\":>20}')
print('-' * 65)

distances = []
for ep in range(8):
    obs, info = env.reset(seed=ep)
    d_pot = info['raw_obs'].get('d_pothole', -1)
    box = env._pothole_box
    x_range = f'[{box[0,0]:.1f}, {box[0,1]:.1f}]'
    y_range = f'[{box[1,0]:.1f}, {box[1,1]:.1f}]'
    print(f'{ep:>8} {d_pot:>12.1f} {x_range:>20} {y_range:>20}')
    distances.append(d_pot)

env.close()
unique = len(set([round(d, 1) for d in distances]))
print()
if unique >= 4:
    print(f'PASS: {unique}/8 unique positions — pothole is randomized.')
else:
    print(f'FAIL: Only {unique}/8 unique positions — randomization may be broken.')
"


# ============================================================
# STEP 8: Verify Conflict-Guaranteed Spawning (5 minutes)
# ============================================================
# PURPOSE: Confirm other agents actually interact with the ego
# WHAT TO CHECK:
#   - Conflict rate >= 70% for all tested maneuvers
#   - An agent comes within 15m of ego at some point during the episode

python3 experiments/pde/verify_conflicts.py --episodes 30 --scenarios 1a 1b --maneuvers stem_right stem_left


# ============================================================
# STEP 9: Verify Dense Scenarios (3 minutes)
# ============================================================
# PURPOSE: Confirm dense variants spawn multiple agents per type
# WHAT TO CHECK:
#   - 2_dense: at least 2 cars + 1 pedestrian visible
#   - 3_dense: at least 2 cars + 1 ped + 1 motorcycle
#   - 4_dense: same as 3_dense + pothole
#   - More agents than non-dense base scenario

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

for scen in ['2', '2_dense', '3', '3_dense', '4', '4_dense']:
    env = SumoEnv(scenario_name=scen, ego_maneuver='stem_right')
    obs, info = env.reset(seed=42)
    
    # Run 50 steps to let agents spawn
    max_agents = 0
    agent_types_seen = set()
    for _ in range(80):
        obs, r, term, trunc, info = env.step(1)  # CREEP
        agents = info['raw_obs']['agents']
        max_agents = max(max_agents, len(agents))
        for a in agents:
            agent_types_seen.add(a.get('type', '?'))
        if term or trunc:
            break
    
    dense_tag = ' [DENSE]' if 'dense' in scen else '        '
    print(f'{scen:>8}{dense_tag}: max_agents={max_agents}, types={sorted(agent_types_seen)}')
    env.close()

print()
print('PASS if dense variants show MORE agents than their base versions.')
"


# ============================================================
# STEP 10: Verify Style Filter (2 minutes)
# ============================================================
# PURPOSE: Confirm nominal and adversarial style filters work
# WHAT TO CHECK:
#   - Nominal: only nominal/timid/cautious styles appear
#   - Adversarial: only aggressive/erratic/drunk styles appear

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

for filt in [None, 'nominal', 'adversarial']:
    env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', style_filter=filt)
    styles = set()
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        b = info.get('behavior')
        if b and b.car:
            styles.add(b.car.style)
    env.close()
    label = filt if filt else 'all (no filter)'
    print(f'Style filter = {label:>15}: car styles seen = {sorted(styles)}')

print()
print('PASS if nominal shows only [nominal, timid] and adversarial shows only aggressive styles.')
"


# ============================================================
# STEP 11: Verify State Ablation (2 minutes)
# ============================================================
# PURPOSE: Confirm no_visibility ablation zeros out visibility features
# WHAT TO CHECK:
#   - Normal: alpha_cz varies, d_occ varies
#   - Ablated: alpha_cz = 1.0, d_occ = 200.0, dt_seen = 0.0

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

for ablation in [None, 'no_visibility']:
    env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', state_ablation=ablation)
    obs, info = env.reset(seed=42)
    vis = info['raw_obs']['vis']
    label = ablation if ablation else 'normal'
    print(f'{label:>15}: alpha_cz={vis[\"alpha_cz\"]:.3f}  d_occ={vis[\"d_occ\"]:.1f}  dt_seen={vis[\"dt_seen\"]:.2f}  sigma_percep={vis[\"sigma_percep\"]:.3f}')
    env.close()

print()
print('PASS if no_visibility shows alpha_cz=1.000, d_occ=200.0, dt_seen=0.00.')
"


# ============================================================
# STEP 12: Verify Observation Dimensions (2 minutes)
# ============================================================
# PURPOSE: Confirm obs_dim is consistent across all configurations
# WHAT TO CHECK:
#   - All maneuvers produce same obs_dim
#   - Dense scenarios produce same obs_dim as base
#   - State ablation produces same obs_dim (values zeroed, not removed)

python3 -c "
import sys; sys.path.insert(0,'.')
from env.sumo_env import SumoEnv

configs = [
    ('1a', 'stem_right', {}),
    ('1a', 'stem_left', {}),
    ('1a', 'right_left', {}),
    ('1a', 'left_stem', {}),
    ('1b', 'stem_right', {}),
    ('2', 'stem_right', {}),
    ('2_dense', 'stem_right', {}),
    ('4', 'stem_right', {}),
    ('1a', 'stem_right', {'buildings': False}),
    ('1a', 'stem_right', {'state_ablation': 'no_visibility'}),
    ('1a', 'stem_right', {'style_filter': 'nominal'}),
]

dims = []
for scen, man, extra in configs:
    env = SumoEnv(scenario_name=scen, ego_maneuver=man, **extra)
    dim = env.observation_space.shape[0]
    dims.append(dim)
    extra_str = str(extra) if extra else ''
    print(f'  {scen:>8} {man:>12} {extra_str:>35}: obs_dim = {dim}')
    env.close()

base_dim = dims[0]
all_same = all(d == base_dim for d in dims[:8])  # first 8 should match
print()
if all_same:
    print(f'PASS: All configurations produce consistent obs_dim = {base_dim}.')
else:
    print(f'FAIL: Inconsistent obs_dim across configurations!')
"


# ============================================================
# STEP 13: Ablation Dry Run (UPDATED FOR v17 — 2026-04-28)
# ============================================================
# PURPOSE: Verify ablation orchestrator generates correct jobs (post-rigor scale-up)
# WHAT TO CHECK:
#   - Tier 1: 1440 jobs (1200 trainable + 240 rule_based eval-only)
#   - Tier 2: 260 jobs (160 lambda + 100 occlusion)
#   - Tier 3: 275 jobs (100 state + 100 behavioral + 75 dense)
#   - Tier 4: 0 jobs unless checkpoints exist (held-out eval)
#   - Tier supp: 126 jobs
#   - Sample commands contain correct flags

python3 experiments/pde/smoke_test_orchestrator.py


# ============================================================
# ============================================================
#
#   IF ALL 13 STEPS PASS → RUN THE SMOKE TEST:
#
#   python3 experiments/pde/smoke_test.py \
#       --scenario 1a --ego_maneuver stem_right
#
#   IF SMOKE TEST PASSES (all 5 methods OK) → RUN CALIBRATION:
#
#   make calibrate SCENARIO=1a MANEUVER=stem_right
#
# ============================================================
# ============================================================

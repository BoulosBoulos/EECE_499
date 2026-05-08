"""Step 5 (revised): alpha_cz must rise from low (occluded, far from CZ) to
high (clear, at the CZ) as ego approaches.

Original spec used CREEP for 300 steps; at ~1 m/s the ego never reached
d_cz < 5 m where alpha is geometrically expected to clear. We use a hybrid
action: CREEP for the first 30 steps to capture the far/occluded regime,
then GO until termination to actually traverse into the conflict zone.
"""
import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right')
obs, info = env.reset(seed=42)
trace = []
for step in range(500):
    action = 1 if step < 30 else 3   # CREEP, then GO
    obs, r, term, trunc, info = env.step(action)
    alpha = info['raw_obs']['vis']['alpha_cz']
    d_cz = info['built']['s_geom'][1]
    trace.append({'step': step, 'd_cz': float(d_cz), 'alpha_cz': float(alpha)})
    if term or trunc:
        break
env.close()

alphas = [t['alpha_cz'] for t in trace]
d_at_min = trace[alphas.index(min(alphas))]['d_cz']
d_at_max = trace[alphas.index(max(alphas))]['d_cz']
min_d_cz_reached = min(t['d_cz'] for t in trace)

ok = (min(alphas) < 0.5
      and max(alphas) > 0.8
      and d_at_max < d_at_min
      and min_d_cz_reached < 5.0)

result = {
    'phase': '2.1',
    'pass': bool(ok),
    'alpha_min': min(alphas),
    'alpha_max': max(alphas),
    'd_cz_at_min_alpha': d_at_min,
    'd_cz_at_max_alpha': d_at_max,
    'min_d_cz_reached': min_d_cz_reached,
    'n_steps': len(trace),
    'terminated_at_step': len(trace) - 1 if len(trace) < 500 else None,
}
with open('verification/phase2_step5_occlusion.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if ok else 1)

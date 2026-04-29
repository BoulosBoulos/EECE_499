import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

result = {}
for ablation in [None, 'no_visibility']:
    env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', state_ablation=ablation)
    _, info = env.reset(seed=42)
    vis = info['raw_obs']['vis']
    label = ablation if ablation else 'normal'
    result[label] = {k: float(vis[k]) for k in ['alpha_cz', 'd_occ', 'dt_seen', 'sigma_percep']}
    env.close()

abl = result['no_visibility']
ok = (abs(abl['alpha_cz'] - 1.0) < 1e-6
      and abs(abl['d_occ'] - 200.0) < 1e-3
      and abs(abl['dt_seen'] - 0.0) < 1e-6)
out = {'phase': '2.7', 'pass': bool(ok), 'values': result}
with open('verification/phase2_step11_state.json', 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if ok else 1)

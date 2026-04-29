import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

configs = [
    ('1a', 'stem_right', {}),
    ('1a', 'stem_left', {}),
    ('1a', 'right_left', {}),
    ('1a', 'left_stem', {}),
    ('1b', 'stem_right', {}),
    ('2',  'stem_right', {}),
    ('2_dense', 'stem_right', {}),
    ('4',  'stem_right', {}),
    ('1a', 'stem_right', {'buildings': False}),
    ('1a', 'stem_right', {'state_ablation': 'no_visibility'}),
    ('1a', 'stem_right', {'style_filter': 'nominal'}),
]
dims = []
for scen, man, extra in configs:
    env = SumoEnv(scenario_name=scen, ego_maneuver=man, **extra)
    dim = env.observation_space.shape[0]
    dims.append({'scen': scen, 'man': man, 'extra': extra, 'obs_dim': int(dim)})
    env.close()
unique = set(d['obs_dim'] for d in dims)
ok = len(unique) == 1
out = {'phase': '2.8', 'pass': bool(ok), 'unique_dims': sorted(unique), 'configs': dims}
with open('verification/phase2_step12_obsdim.json', 'w') as f:
    json.dump(out, f, indent=2)
print('unique obs_dims =', unique)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

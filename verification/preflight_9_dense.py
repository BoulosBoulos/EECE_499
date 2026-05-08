import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

results = {}
for scen in ['2', '2_dense', '3', '3_dense', '4', '4_dense']:
    env = SumoEnv(scenario_name=scen, ego_maneuver='stem_right')
    env.reset(seed=42)
    max_n = 0
    for _ in range(80):
        _, _, term, trunc, info = env.step(1)
        max_n = max(max_n, len(info['raw_obs']['agents']))
        if term or trunc: break
    env.close()
    results[scen] = max_n
    print(f'  {scen:>10}: max_agents = {max_n}')

ok = (results['2_dense'] > results['2']
      and results['3_dense'] > results['3']
      and results['4_dense'] > results['4'])
out = {'phase': '2.5', 'pass': bool(ok), 'max_agents_per_scenario': results}
with open('verification/phase2_step9_dense.json', 'w') as f:
    json.dump(out, f, indent=2)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

env = SumoEnv(scenario_name='1d', ego_maneuver='stem_right')
positions = []
for ep in range(8):
    env.reset(seed=ep)
    box = env._pothole_box.copy()
    positions.append(box.tolist())
env.close()

mids = sorted({(round((b[0][0]+b[0][1])/2, 1),
                round((b[1][0]+b[1][1])/2, 1)) for b in positions})
n_unique = len(mids)
ok = n_unique >= 4
result = {'phase': '2.3', 'pass': ok, 'n_unique_positions': n_unique,
          'positions': positions}
with open('verification/phase2_step7_pothole.json', 'w') as f:
    json.dump(result, f, indent=2)
print('  unique positions:', n_unique, '/', 8)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

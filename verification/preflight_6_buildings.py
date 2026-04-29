import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

def avg_alpha(buildings):
    env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', buildings=buildings)
    env.reset(seed=42)
    vals = []
    for _ in range(30):
        _, _, term, trunc, info = env.step(1)
        vals.append(info['raw_obs']['vis']['alpha_cz'])
        if term or trunc: break
    env.close()
    return sum(vals)/len(vals)

a_on = avg_alpha(True)
a_off = avg_alpha(False)
diff = a_off - a_on
ok = diff > 0.1
result = {'phase': '2.2', 'pass': ok,
          'mean_alpha_buildings_on': a_on, 'mean_alpha_buildings_off': a_off,
          'diff': diff}
with open('verification/phase2_step6_buildings.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if ok else 1)

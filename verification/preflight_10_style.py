import sys, json
sys.path.insert(0, '.')
from env.sumo_env import SumoEnv

result = {}
for filt in [None, 'nominal', 'adversarial']:
    env = SumoEnv(scenario_name='1a', ego_maneuver='stem_right', style_filter=filt)
    styles = set()
    for ep in range(15):
        _, info = env.reset(seed=ep)
        b = info.get('behavior')
        if b and getattr(b, 'car', None):
            styles.add(b.car.style)
    env.close()
    label = filt if filt else 'all'
    result[label] = sorted(styles)
    print(f'  {label:>14}: {sorted(styles)}')

nom = set(result['nominal']); adv = set(result['adversarial'])
ok = (len(nom & {'aggressive', 'erratic', 'drunk'}) == 0
      and len(adv & {'aggressive', 'erratic', 'drunk'}) >= 1
      and len(result['all']) >= 3)
out = {'phase': '2.6', 'pass': bool(ok), 'styles_seen': result}
with open('verification/phase2_step10_style.json', 'w') as f:
    json.dump(out, f, indent=2)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

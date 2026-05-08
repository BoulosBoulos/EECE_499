import os, json, glob
out = {'phases': {}}
for path in sorted(glob.glob('verification/phase*.json')):
    name = os.path.basename(path).replace('.json', '')
    with open(path) as f: out['phases'][name] = json.load(f)

all_pass = all(v.get('pass', True) is True
               for v in out['phases'].values()
               if isinstance(v, dict))
out['ALL_PASS'] = all_pass

with open('verification/REPORT.json', 'w') as f:
    json.dump(out, f, indent=2)

lines = ['# Verification Report', '']
for name, body in out['phases'].items():
    p = body.get('pass', None)
    badge = 'PASS' if p is True else ('FAIL' if p is False else '—')
    lines.append(f'- [{badge}] **{name}**')
    for k, v in body.items():
        if k == 'pass': continue
        lines.append(f'    - {k}: {v}')
lines += ['', f'## Overall: {"PASS" if all_pass else "FAIL"}']
with open('verification/REPORT.md', 'w') as f:
    f.write('\n'.join(lines))
print('Wrote verification/REPORT.md  &  verification/REPORT.json')
print('ALL_PASS =', all_pass)

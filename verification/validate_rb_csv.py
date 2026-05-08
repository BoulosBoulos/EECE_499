import csv, json, sys

csv_path = 'verification/smoke_rb/eval_rule_based_1a_stem_right.csv'
required_cols = {
    'seed', 'eval_mode', 'mean_return', 'collision_rate',
    'success_rate', 'mean_ttc'
}

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

cols_present = required_cols.issubset(rows[0].keys()) if rows else False
modes = {r['eval_mode'] for r in rows}
both_modes = modes == {'deterministic', 'stochastic'}

# Sanity on values: success_rate in [0,1], collision_rate in [0,1], mean_ttc finite & > 0
def _f(x):
    try: return float(x)
    except Exception: return None

values_sane = all(
    _f(r['success_rate']) is not None and 0.0 <= float(r['success_rate']) <= 1.0
    and _f(r['collision_rate']) is not None and 0.0 <= float(r['collision_rate']) <= 1.0
    and _f(r['mean_ttc']) is not None and float(r['mean_ttc']) > 0.0
    for r in rows)

n_rows = len(rows)
ok = (n_rows >= 1) and cols_present and both_modes and values_sane
result = {
    'phase': '4.2',
    'pass': bool(ok),
    'n_rows': n_rows,
    'modes_present': sorted(modes),
    'required_cols_present': cols_present,
    'values_sane': values_sane,
}
with open('verification/phase4_rb_validation.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if ok else 1)

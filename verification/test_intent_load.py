import sys, json, os
sys.path.insert(0, '.')
import torch
from models.intent_style import IntentStylePredictor

per_member = []
load_ok = True
for ens_idx in range(3):
    ckpt_path = f'results/intent_model_v9_member{ens_idx}.pt'
    exists = os.path.isfile(ckpt_path)
    has_model_key = False
    member_ok = False
    if exists:
        data = torch.load(ckpt_path, map_location='cpu')
        has_model_key = isinstance(data, dict) and 'model' in data
        if has_model_key:
            pred = IntentStylePredictor(
                input_dim=12, hidden_dim=384, num_layers=3,
                bidirectional=True, dropout=0.2,
            ).eval()
            try:
                pred.load_state_dict(data['model'])
                member_ok = True
            except Exception as e:
                print(f'FAIL load_state_dict member {ens_idx}:', e)
    per_member.append({
        'member': ens_idx,
        'checkpoint': ckpt_path,
        'exists': bool(exists),
        'has_model_key': bool(has_model_key),
        'load_ok': bool(member_ok),
    })
    load_ok = load_ok and member_ok

result = {'phase': '5', 'pass': bool(load_ok), 'per_member': per_member}
with open('verification/phase5_intent_validation.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if load_ok else 1)

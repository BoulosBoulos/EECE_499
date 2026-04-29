import sys, json, os
sys.path.insert(0, '.')
import torch
from models.intent_style import IntentStylePredictor

ckpt_path = 'results/intent_model.pt'
ok = os.path.isfile(ckpt_path)
data = torch.load(ckpt_path, map_location='cpu') if ok else None
has_model_key = ok and isinstance(data, dict) and 'model' in data

load_ok = False
if has_model_key:
    pred = IntentStylePredictor(input_dim=9, hidden_dim=64).eval()
    try:
        pred.load_state_dict(data['model'])
        load_ok = True
    except Exception as e:
        print('FAIL load_state_dict:', e)

result = {'phase': '5', 'pass': bool(load_ok),
          'checkpoint_exists': ok, 'has_model_key': has_model_key, 'load_ok': load_ok}
with open('verification/phase5_intent_validation.json','w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if load_ok else 1)

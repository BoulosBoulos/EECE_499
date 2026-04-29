import sys; sys.path.insert(0, '.')
import torch, json
from models.pde.dynamics import _smooth_clamp_nonneg

xs = torch.tensor([-0.5, -0.05, 0.0, 0.05, 0.5], requires_grad=True)
ys = _smooth_clamp_nonneg(xs)
gs = torch.autograd.grad(ys.sum(), xs)[0].detach().tolist()
print('  inputs :', xs.detach().tolist())
print('  outputs:', ys.detach().tolist())
print('  grads  :', gs)

ok = all(g > 0 for g in gs[1:])  # skip the very-negative point which can be ~0
print('PASS' if ok else 'FAIL')
with open('verification/phase1_smooth_clamp.json','w') as f:
    json.dump({'phase':'1.3','inputs': xs.detach().tolist(), 'grads': gs, 'pass': ok}, f, indent=2)
sys.exit(0 if ok else 1)

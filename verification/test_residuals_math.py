"""Empirical sanity for all 4 PDE residuals on synthetic xi.

Verifies:
  - Forward pass produces shape (B,) finite tensor
  - Backward pass produces finite, non-zero gradients on U_phi parameters
  - Autograd graph for grad_U flows correctly (residual is differentiable in U_phi)
"""
import sys, os, json
sys.path.insert(0, '.')
import torch
import numpy as np

from models.pde.state_builder import XI_DIM
from models.pde.dynamics import BehavioralDynamics
from models.pde.residuals import (
    hjb_residual, soft_hjb_residual, eikonal_residual, cbf_residual,
)
from models.pde.hjb_aux_critic import HJBAuxCritic

torch.manual_seed(42)
np.random.seed(42)

B = 16
xi = torch.randn(B, XI_DIM, dtype=torch.float32)
xi[:, 8] = torch.rand(B)  # alpha_cz in [0,1]
xi[:, 7] = torch.rand(B) * 50  # d_cz reasonable

dyn = BehavioralDynamics(dt=0.1)
U_phi = HJBAuxCritic(in_dim=XI_DIM, hidden_dim=64)

results = {'phase': '1.2', 'residuals': {}}
all_ok = True

for name, fn, kwargs in [
    ('hjb',       hjb_residual,       {'gamma': 0.99}),
    ('soft_hjb',  soft_hjb_residual,  {'gamma': 0.99, 'tau': 0.1}),
    ('eikonal',   eikonal_residual,   {'v_min': 0.5, 'ttc_thr': 3.0}),
    ('cbf',       cbf_residual,       {'alpha_cbf': 1.0, 'cbf_safe_offset': 10.0}),
]:
    try:
        for p in U_phi.parameters(): p.grad = None
        rho = fn(U_phi, xi.clone(), dyn, **kwargs)
        assert rho.shape == (B,), f'shape mismatch: {rho.shape}'
        assert torch.isfinite(rho).all(), 'non-finite forward'
        loss = (rho ** 2).mean()
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in U_phi.parameters() if p.grad is not None)
        assert np.isfinite(gnorm) and gnorm > 0, f'bad gnorm = {gnorm}'
        rho_min = rho.min().item(); rho_max = rho.max().item(); rho_mean = rho.mean().item()
        print(f'  OK   {name:10s}  rho in [{rho_min:+.3f}, {rho_max:+.3f}]  mean {rho_mean:+.3f}  ||g||={gnorm:.3e}')
        results['residuals'][name] = {'pass': True, 'rho_min': rho_min, 'rho_max': rho_max,
                                       'rho_mean': rho_mean, 'gnorm': gnorm}
    except Exception as e:
        print(f'  FAIL {name:10s}  {type(e).__name__}: {e}')
        results['residuals'][name] = {'pass': False, 'error': f'{type(e).__name__}: {e}'}
        all_ok = False

results['pass'] = all_ok
with open('verification/phase1_residuals.json', 'w') as f:
    json.dump(results, f, indent=2)
sys.exit(0 if all_ok else 1)

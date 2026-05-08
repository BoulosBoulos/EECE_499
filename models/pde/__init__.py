"""PDE-based auxiliary critic methods:
- Hard-HJB residual critic
- Soft-HJB entropy-regularized critic
- Eikonal-constrained critic
- CBF-PDE safety-regularized critic
- Fusion (Soft-HJB optimality + CBF safety) critic — Phase 1C
"""

from models.pde.hjb_aux_agent import HJBAuxAgent
from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
from models.pde.eikonal_aux_agent import EikonalAuxAgent
from models.pde.cbf_aux_agent import CBFAuxAgent
from models.pde.fusion_aux_agent import FusionAuxAgent

__all__ = [
    "HJBAuxAgent",
    "SoftHJBAuxAgent",
    "EikonalAuxAgent",
    "CBFAuxAgent",
    "FusionAuxAgent",
]

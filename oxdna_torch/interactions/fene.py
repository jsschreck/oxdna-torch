"""
FENE backbone potential for bonded nucleotides.

E_backbone = -(FENE_EPS/2) * ln(1 - ((r_back - r0)^2 / FENE_DELTA^2))

where r_back is the distance between backbone sites of bonded neighbors.
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..utils import safe_norm


def fene_energy(
    positions: Tensor,
    back_offsets: Tensor,
    bonded_pairs: Tensor,
    box: Optional[Tensor] = None,
    params: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Compute FENE backbone energy for all bonded pairs.

    Args:
        positions: (N, 3) center-of-mass positions
        back_offsets: (N, 3) backbone site offsets from COM
        bonded_pairs: (B, 2) int tensor of bonded pair indices [p, q]
            where q = p.n3 (3' neighbor of p)
        box: (3,) periodic box dimensions, or None
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        Scalar total FENE energy
    """
    p_idx = bonded_pairs[:, 0]  # (B,)
    q_idx = bonded_pairs[:, 1]  # (B,)

    # Displacement between centers of mass: r = q.pos - p.pos
    r_com = positions[q_idx] - positions[p_idx]  # (B, 3)

    # Apply minimum image convention
    if box is not None:
        r_com = r_com - box * torch.round(r_com / box)

    # Backbone-to-backbone vector
    r_back = r_com + back_offsets[q_idx] - back_offsets[p_idx]  # (B, 3)

    # Distance between backbone sites
    r_back_mod = safe_norm(r_back, dim=-1)  # (B,)

    if params is not None:
        fene_eps = params['fene_eps']
        fene_r0 = params['fene_r0']
        fene_delta = params['fene_delta']
        delta2 = fene_delta * fene_delta
    else:
        fene_eps = C.FENE_EPS
        fene_r0 = C.FENE_R0_OXDNA
        delta2 = C.FENE_DELTA2

    # Distance from equilibrium
    r_back_r0 = r_back_mod - fene_r0  # (B,)

    # FENE energy: -(eps/2) * ln(1 - (r-r0)^2 / delta^2)
    # Clamp to prevent log of negative numbers (would indicate broken bond)
    x = (r_back_r0 ** 2) / delta2
    x_clamped = torch.clamp(x, max=1.0 - 1e-6)

    energy_per_pair = -(fene_eps / 2.0) * torch.log(1.0 - x_clamped)  # (B,)

    return energy_per_pair.sum()

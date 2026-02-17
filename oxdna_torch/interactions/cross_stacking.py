"""
Cross stacking interaction between non-bonded nucleotides.

E_crst = f2(r_cstack) * f4(theta1) * f4(theta2) * f4(theta3) * f4_sym(theta4)
         * f4_sym(theta7) * f4_sym(theta8)

where f4_sym(theta) = f4(theta) + f4(-theta) for theta4, theta7, theta8
(allowing interaction at both +/- angles).

The angles use the same definitions as hydrogen bonding.
The radial function uses f2 (harmonic) instead of f1 (Morse).
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..smooth import f2, f4_of_cos
from ..utils import safe_norm, dot
from ..pairs import min_image_displacement


def cross_stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    base_offsets: Tensor,
    nonbonded_pairs: Tensor,
    box: Optional[Tensor] = None,
    params: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Compute cross stacking energy for all non-bonded pairs.

    Args:
        positions: (N, 3) COM positions
        quaternions: (N, 4) unit quaternions
        base_offsets: (N, 3) base site offsets from COM
        nonbonded_pairs: (P, 2) non-bonded pair indices
        box: (3,) periodic box dimensions, or None
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        Scalar total cross stacking energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    from ..quaternion import quat_to_rotmat

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # COM displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)

    # Base-to-base displacement
    r_cstack = r_com + base_offsets[q_idx] - base_offsets[p_idx]
    r_cstack_mod = safe_norm(r_cstack, dim=-1)

    # Distance filter
    in_range = (r_cstack_mod > C.CRST_RCLOW) & (r_cstack_mod < C.CRST_RCHIGH)

    if not in_range.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    r_cstack_dir = r_cstack / r_cstack_mod.unsqueeze(-1)

    # Get orientation vectors
    R = quat_to_rotmat(quaternions)
    a1_all = R[:, :, 0]
    a3_all = R[:, :, 2]

    pa1 = a1_all[p_idx]
    pa3 = a3_all[p_idx]
    qa1 = a1_all[q_idx]  # b1
    qa3 = a3_all[q_idx]  # b3

    # Compute cosines
    cost1 = -dot(pa1, qa1, dim=-1)
    cost2 = -dot(qa1, r_cstack_dir, dim=-1)
    cost3 = dot(pa1, r_cstack_dir, dim=-1)
    cost4 = dot(pa3, qa3, dim=-1)
    cost7 = -dot(qa3, r_cstack_dir, dim=-1)
    cost8 = dot(pa3, r_cstack_dir, dim=-1)

    # f2 radial (cross stacking type = 0)
    val_f2 = f2(r_cstack_mod, C.CRST_F2, params=params)

    # Angular modulations
    val_f4t1 = f4_of_cos(cost1, C.CRST_F4_THETA1, params=params)
    val_f4t2 = f4_of_cos(cost2, C.CRST_F4_THETA2, params=params)
    val_f4t3 = f4_of_cos(cost3, C.CRST_F4_THETA3, params=params)

    # Symmetric f4 for theta4, theta7, theta8: f4(cos) + f4(-cos)
    val_f4t4 = f4_of_cos(cost4, C.CRST_F4_THETA4, params=params) + f4_of_cos(-cost4, C.CRST_F4_THETA4, params=params)
    val_f4t7 = f4_of_cos(cost7, C.CRST_F4_THETA7, params=params) + f4_of_cos(-cost7, C.CRST_F4_THETA7, params=params)
    val_f4t8 = f4_of_cos(cost8, C.CRST_F4_THETA8, params=params) + f4_of_cos(-cost8, C.CRST_F4_THETA8, params=params)

    # Total cross stacking energy per pair
    energy_per_pair = val_f2 * val_f4t1 * val_f4t2 * val_f4t3 * val_f4t4 * val_f4t7 * val_f4t8

    # Zero out pairs outside distance range
    energy_per_pair = torch.where(in_range, energy_per_pair, torch.zeros_like(energy_per_pair))

    return energy_per_pair.sum()

"""
Hydrogen bonding interaction between non-bonded Watson-Crick base pairs.

E_hbond = f1(r_hydro) * f4(theta1) * f4(theta2) * f4(theta3) * f4(theta4) * f4(theta7) * f4(theta8)

Only computed for Watson-Crick pairs: A-T (btype sum = 3) and C-G (btype sum = 3).

The angles are:
  theta1 = acos(-a1 . b1)          relative base orientation
  theta2 = acos(-b1 . r_hydro_dir) angle from q base to bond
  theta3 = acos(a1 . r_hydro_dir)  angle from p base to bond
  theta4 = acos(a3 . b3)           base twist angle
  theta7 = acos(-b3 . r_hydro_dir) angle from q base normal to bond
  theta8 = acos(a3 . r_hydro_dir)  angle from p base normal to bond
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..smooth import f1, f4_of_cos
from ..utils import safe_norm, dot, safe_acos
from ..pairs import min_image_displacement


def hydrogen_bonding_energy(
    positions: Tensor,
    quaternions: Tensor,
    base_offsets: Tensor,
    nonbonded_pairs: Tensor,
    base_types: Tensor,
    hbond_eps_matrix: Tensor,
    box: Optional[Tensor] = None,
    params: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Compute hydrogen bonding energy for all non-bonded pairs.

    Args:
        positions: (N, 3) COM positions
        quaternions: (N, 4) unit quaternions
        base_offsets: (N, 3) base site offsets from COM
        nonbonded_pairs: (P, 2) non-bonded pair indices
        base_types: (N,) int tensor of base types (A=0, C=1, G=2, T=3)
        hbond_eps_matrix: (4, 4) HB epsilon lookup [p_type, q_type]
        box: (3,) periodic box dimensions, or None
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        Scalar total hydrogen bonding energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    from ..quaternion import quat_to_rotmat

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # Check Watson-Crick eligibility: btype_p + btype_q == 3
    p_types = base_types[p_idx]
    q_types = base_types[q_idx]
    is_wc = (p_types + q_types) == 3  # (P,) bool

    if not is_wc.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    # Filter to only WC pairs for efficiency
    wc_mask = is_wc
    p_idx_wc = p_idx[wc_mask]
    q_idx_wc = q_idx[wc_mask]
    p_types_wc = p_types[wc_mask]
    q_types_wc = q_types[wc_mask]

    # COM displacement
    r_com = positions[q_idx_wc] - positions[p_idx_wc]
    r_com = min_image_displacement(r_com, box)

    # Base-to-base displacement
    r_hydro = r_com + base_offsets[q_idx_wc] - base_offsets[p_idx_wc]
    r_hydro_mod = safe_norm(r_hydro, dim=-1)

    # Distance filter
    in_range = (r_hydro_mod > C.HYDR_RCLOW) & (r_hydro_mod < C.HYDR_RCHIGH)

    if not in_range.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    r_hydro_dir = r_hydro / r_hydro_mod.unsqueeze(-1)

    # Get orientation vectors
    R = quat_to_rotmat(quaternions)
    a1_all = R[:, :, 0]
    a3_all = R[:, :, 2]

    pa1 = a1_all[p_idx_wc]
    pa3 = a3_all[p_idx_wc]
    qa1 = a1_all[q_idx_wc]  # b1
    qa3 = a3_all[q_idx_wc]  # b3

    # Compute cosines of angles
    cost1 = -dot(pa1, qa1, dim=-1)           # -a1 . b1
    cost2 = -dot(qa1, r_hydro_dir, dim=-1)   # -b1 . r_dir
    cost3 = dot(pa1, r_hydro_dir, dim=-1)    # a1 . r_dir
    cost4 = dot(pa3, qa3, dim=-1)            # a3 . b3
    cost7 = -dot(qa3, r_hydro_dir, dim=-1)   # -b3 . r_dir
    cost8 = dot(pa3, r_hydro_dir, dim=-1)    # a3 . r_dir

    # Look up per-pair epsilon
    eps = hbond_eps_matrix[p_types_wc, q_types_wc]  # (W,)

    # Compute f1 radial (hydrogen bonding type = 0)
    val_f1 = f1(r_hydro_mod, C.HYDR_F1, eps, params=params)

    # Compute angular modulations
    val_f4t1 = f4_of_cos(cost1, C.HYDR_F4_THETA1, params=params)
    val_f4t2 = f4_of_cos(cost2, C.HYDR_F4_THETA2, params=params)
    val_f4t3 = f4_of_cos(cost3, C.HYDR_F4_THETA3, params=params)
    val_f4t4 = f4_of_cos(cost4, C.HYDR_F4_THETA4, params=params)
    val_f4t7 = f4_of_cos(cost7, C.HYDR_F4_THETA7, params=params)
    val_f4t8 = f4_of_cos(cost8, C.HYDR_F4_THETA8, params=params)

    # Total HB energy per pair
    energy_per_pair = val_f1 * val_f4t1 * val_f4t2 * val_f4t3 * val_f4t4 * val_f4t7 * val_f4t8

    # Zero out pairs outside distance range
    energy_per_pair = torch.where(in_range, energy_per_pair, torch.zeros_like(energy_per_pair))

    return energy_per_pair.sum()

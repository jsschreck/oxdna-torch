"""
Sequential stacking interaction between bonded nucleotides.

E_stack = f1(r_stack) * f4(theta4) * f4(theta5) * f4(theta6) * f5(phi1) * f5(phi2)

Computed only between bonded neighbors (p and p.n3).

The angles are:
  theta4 = acos(a3 . b3)           relative twist of bases
  theta5 = acos(a3 . r_stack_dir)  tilt of p base normal
  theta6 = acos(-b3 . r_stack_dir) tilt of q base normal
  phi1 = acos(a2 . r_backref_dir)  azimuthal angle on p
  phi2 = acos(b2 . r_backref_dir)  azimuthal angle on q

The f1 radial function uses the STCK parameter set (type=1)
with sequence-dependent epsilon.
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..smooth import f1, f4_of_cos, f5
from ..utils import safe_norm, dot
from ..pairs import min_image_displacement


def stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    stack_offsets: Tensor,
    bonded_pairs: Tensor,
    stacking_eps: Tensor,
    box: Optional[Tensor] = None,
    params: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Compute sequential stacking energy for all bonded pairs.

    Args:
        positions: (N, 3) COM positions
        quaternions: (N, 4) unit quaternions
        stack_offsets: (N, 3) stacking site offsets from COM
        bonded_pairs: (B, 2) bonded pair indices [p, q]
        stacking_eps: (B,) sequence-dependent stacking epsilon for each pair
        box: (3,) periodic box dimensions, or None
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        Scalar total stacking energy
    """
    from ..quaternion import quat_to_rotmat

    p_idx = bonded_pairs[:, 0]
    q_idx = bonded_pairs[:, 1]

    # Get rotation matrices for orientation vectors
    R = quat_to_rotmat(quaternions)  # (N, 3, 3)
    a1 = R[:, :, 0]  # (N, 3) - principal axis
    a2 = R[:, :, 1]  # (N, 3)
    a3 = R[:, :, 2]  # (N, 3) - base normal

    # Select pair orientation vectors
    pa1 = a1[p_idx]  # (B, 3)
    pa2 = a2[p_idx]
    pa3 = a3[p_idx]
    qa1 = a1[q_idx]  # actually b1
    qa2 = a2[q_idx]  # actually b2
    qa3 = a3[q_idx]  # actually b3

    # COM displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)

    # Reference backbone vector (without grooving, used for phi angles)
    # rbackref = r_com + b1 * POS_BACK - a1 * POS_BACK
    r_backref = r_com + qa1 * C.POS_BACK - pa1 * C.POS_BACK  # (B, 3)
    r_backref_mod = safe_norm(r_backref, dim=-1)  # (B,)

    # Stack-to-stack displacement
    r_stack = r_com + stack_offsets[q_idx] - stack_offsets[p_idx]  # (B, 3)
    r_stack_mod = safe_norm(r_stack, dim=-1)  # (B,)
    r_stack_dir = r_stack / r_stack_mod.unsqueeze(-1)  # (B, 3)

    # Compute cosines of angles
    cost4 = dot(pa3, qa3, dim=-1)           # a3 . b3
    cost5 = dot(pa3, r_stack_dir, dim=-1)   # a3 . r_stack_dir
    cost6 = -dot(qa3, r_stack_dir, dim=-1)  # -b3 . r_stack_dir
    cosphi1 = dot(pa2, r_backref, dim=-1) / r_backref_mod  # a2 . r_backref / |r_backref|
    cosphi2 = dot(qa2, r_backref, dim=-1) / r_backref_mod  # b2 . r_backref / |r_backref|

    # Compute f1 radial (Morse-like with stacking parameters)
    # f1 needs eps and shift per pair
    if params is not None:
        # Use tensor params for gradient flow through A, RC, R0
        stck_A = params['f1_A'][1]    # STCK = f1_type 1
        stck_RC = params['f1_RC'][1]
        stck_R0 = params['f1_R0'][1]
        shift_factor = (1.0 - torch.exp(-(stck_RC - stck_R0) * stck_A)) ** 2
    else:
        import math
        shift_factor = (1.0 - math.exp(-(C.STCK_RC - C.STCK_R0) * C.STCK_A)) ** 2
    shift = stacking_eps * shift_factor

    # Manually compute f1 for stacking (type=1) with per-pair epsilon
    val_f1 = _f1_stacking(r_stack_mod, stacking_eps, shift, params=params)

    # Compute angular modulations
    # C++ code: f4t4 = _custom_f4(cost4, STCK_F4_THETA4)
    #           f4t5 = _custom_f4(-cost5, STCK_F4_THETA5)  <-- note the negation
    #           f4t6 = _custom_f4(cost6, STCK_F4_THETA6)
    val_f4t4 = f4_of_cos(cost4, C.STCK_F4_THETA4, params=params)
    val_f4t5 = f4_of_cos(-cost5, C.STCK_F4_THETA5, params=params)
    val_f4t6 = f4_of_cos(cost6, C.STCK_F4_THETA6, params=params)

    # Compute azimuthal modulations
    val_f5phi1 = f5(cosphi1, C.STCK_F5_PHI1, params=params)
    val_f5phi2 = f5(cosphi2, C.STCK_F5_PHI2, params=params)

    # Total stacking energy per pair
    energy_per_pair = val_f1 * val_f4t4 * val_f4t5 * val_f4t6 * val_f5phi1 * val_f5phi2

    return energy_per_pair.sum()


def _f1_stacking(r: Tensor, eps: Tensor, shift: Tensor,
                 params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """f1 for stacking with per-pair epsilon.

    This is the same as smooth.f1 but takes vectorized eps and shift.

    Args:
        r: (B,) distances
        eps: (B,) per-pair epsilon values
        shift: (B,) per-pair shift values
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (B,) energy values
    """
    if params is not None:
        A = params['f1_A'][1]       # STCK = f1_type 1
        R0 = params['f1_R0'][1]
        BLOW = params['f1_BLOW'][1]
        BHIGH = params['f1_BHIGH'][1]
        RLOW = params['f1_RLOW'][1]
        RHIGH = params['f1_RHIGH'][1]
        RCLOW = params['f1_RCLOW'][1]
        RCHIGH = params['f1_RCHIGH'][1]
    else:
        A = C.STCK_A
        R0 = C.STCK_R0
        BLOW = C.STCK_BLOW
        BHIGH = C.STCK_BHIGH
        RLOW = C.STCK_RLOW
        RHIGH = C.STCK_RHIGH
        RCLOW = C.STCK_RCLOW
        RCHIGH = C.STCK_RCHIGH

    # Morse potential
    tmp = 1.0 - torch.exp(-(r - R0) * A)
    morse = eps * tmp * tmp - shift

    # Smooth onset
    onset = eps * BLOW * (r - RCLOW) ** 2

    # Smooth cutoff
    cutoff = eps * BHIGH * (r - RCHIGH) ** 2

    zero = torch.zeros_like(r)

    val = torch.where(r < RCHIGH,
                      torch.where(r > RHIGH, cutoff,
                                  torch.where(r > RLOW, morse,
                                              torch.where(r > RCLOW, onset, zero))),
                      zero)
    return val

"""
Debye–Hückel electrostatic interaction for oxRNA2.

Uses the same Yukawa + quadratic-cutoff functional form as the DNA2
implementation but with RNA2-specific constants:
  Q        = 0.0858          (vs DNA2: 0.0543)
  lambda_0 = 0.3667258       (vs DNA2: 0.3616455)

The interaction site is the RNA backbone (BACK = a1*(-0.4) + a3*(0.2)),
matching RNAInteraction2::_debye_huckel which uses RNANucleotide::BACK.

Source: RNAInteraction2.cpp _debye_huckel() and init()
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
from ..pairs import min_image_displacement
from ..utils import safe_norm
from ..quaternion import quat_to_rotmat


def rna2_debye_huckel_params(
    temperature: float,
    salt_concentration: float = 1.0,
) -> dict:
    """Compute RNA2 Debye–Hückel derived parameters.

    Args:
        temperature:        temperature in oxDNA reduced units (T_K / 3000)
        salt_concentration: molar salt concentration (default 1.0 M, as in
                            the oxRNA2 paper and typical RNA simulations)

    Returns:
        dict with keys: lambda_, minus_kappa, rhigh, rc, b, prefactor
    """
    q = RC.RNA2_DH_PREFACTOR
    lf = RC.RNA2_DH_LAMBDAFACTOR
    t_ref = RC.RNA2_DH_T_REF

    lambda_ = lf * math.sqrt(temperature / t_ref) / math.sqrt(salt_concentration)
    minus_kappa = -1.0 / lambda_
    rhigh = 3.0 * lambda_

    x = rhigh
    l = lambda_
    rc = x * (q * x + 3.0 * q * l) / (q * (x + l))
    b = -(math.exp(-x / l) * q * q * (x + l) * (x + l)) / (-4.0 * x * x * x * l * l * q)

    return {
        'lambda_':    lambda_,
        'minus_kappa': minus_kappa,
        'rhigh':      rhigh,
        'rc':         rc,
        'b':          b,
        'prefactor':  q,
    }


def rna2_debye_huckel_energy(
    positions: Tensor,
    quaternions: Tensor,
    nonbonded_pairs: Tensor,
    terminus_mask: Tensor,
    dh_params: dict,
    box: Optional[Tensor] = None,
    half_charged_ends: bool = True,
) -> Tensor:
    """Compute RNA2 Debye–Hückel electrostatic energy for all non-bonded pairs.

    The backbone site offset in the RNA body frame is:
        back = a1 * RNA_POS_BACK_a1 + a3 * RNA_POS_BACK_a3
    matching RNANucleotide::BACK used in RNAInteraction2.cpp.

    Args:
        positions:       (N, 3) COM positions
        quaternions:     (N, 4) unit quaternions [w, x, y, z]
        nonbonded_pairs: (P, 2) non-bonded pair indices
        terminus_mask:   (N,) bool tensor, True if nucleotide is a strand terminus
        dh_params:       dict from rna2_debye_huckel_params()
        box:             (3,) periodic box or None
        half_charged_ends: if True, terminus nucleotides contribute half charge

    Returns:
        Scalar total Debye–Hückel energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    rhigh = dh_params['rhigh']
    rc    = dh_params['rc']
    b     = dh_params['b']
    minus_kappa = dh_params['minus_kappa']
    q     = dh_params['prefactor']

    # Compute backbone site offsets from quaternions
    # back = a1 * RNA_POS_BACK_a1 + a3 * RNA_POS_BACK_a3
    R = quat_to_rotmat(quaternions)          # (N, 3, 3)
    a1 = R[:, :, 0]                          # (N, 3)
    a3 = R[:, :, 2]                          # (N, 3)
    back_offsets = (RC.RNA_POS_BACK_a1 * a1
                    + RC.RNA_POS_BACK_a3 * a3)   # (N, 3)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    r_com  = positions[q_idx] - positions[p_idx]
    r_com  = min_image_displacement(r_com, box)
    r_back = r_com + back_offsets[q_idx] - back_offsets[p_idx]
    r_mod  = safe_norm(r_back, dim=-1)

    in_range = r_mod < rc
    if not in_range.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    # Half-charged ends
    if half_charged_ends:
        pterm = terminus_mask[p_idx].float()
        qterm = terminus_mask[q_idx].float()
        cut_factor = (1.0 - 0.5 * pterm) * (1.0 - 0.5 * qterm)
    else:
        cut_factor = torch.ones(nonbonded_pairs.shape[0],
                                dtype=positions.dtype, device=positions.device)

    r = r_mod.clamp(min=1e-9)
    yukawa   = torch.exp(minus_kappa * r) * (q / r)
    quadratic = b * (r - rc) ** 2

    energy_per_pair = torch.where(r < rhigh, yukawa, quadratic)
    energy_per_pair = energy_per_pair * cut_factor
    energy_per_pair = torch.where(in_range, energy_per_pair,
                                  torch.zeros_like(energy_per_pair))

    return energy_per_pair.sum()

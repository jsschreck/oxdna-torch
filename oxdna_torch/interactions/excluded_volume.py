"""
Excluded volume interactions (repulsive Lennard-Jones).

Bonded excluded volume: 3 site pairs per bonded pair
  - base-base (EXCL_S2, R2, B2, RC2)
  - p_base-q_back (EXCL_S3, R3, B3, RC3)
  - p_back-q_base (EXCL_S4, R4, B4, RC4)

Non-bonded excluded volume: 4 site pairs per non-bonded pair
  - back-back (EXCL_S1, R1, B1, RC1)
  - base-base (EXCL_S2, R2, B2, RC2)
  - p_base-q_back (EXCL_S3, R3, B3, RC3) -- note: swapped from bonded
  - p_back-q_base (EXCL_S4, R4, B4, RC4)
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..smooth import repulsive_lj
from ..pairs import min_image_displacement


def bonded_excluded_volume_energy(
    positions: Tensor,
    back_offsets: Tensor,
    base_offsets: Tensor,
    bonded_pairs: Tensor,
    box: Optional[Tensor] = None,
    excl_eps: Optional[Tensor] = None,
) -> Tensor:
    """Compute bonded excluded volume energy for all bonded pairs.

    Three site-pair interactions per bonded pair:
      1. base(p) - base(q)
      2. base(p) - back(q)
      3. back(p) - base(q)

    Args:
        positions: (N, 3) COM positions
        back_offsets: (N, 3) backbone offsets from COM
        base_offsets: (N, 3) base offsets from COM
        bonded_pairs: (B, 2) bonded pair indices [p, q]
        box: (3,) periodic box dimensions, or None
        excl_eps: optional learnable excluded volume epsilon

    Returns:
        Scalar total bonded excluded volume energy
    """
    p_idx = bonded_pairs[:, 0]
    q_idx = bonded_pairs[:, 1]

    # COM displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)

    # Site-to-site displacement vectors
    # 1. base(p) -> base(q)
    r_bb = r_com + base_offsets[q_idx] - base_offsets[p_idx]
    r_bb_sq = (r_bb * r_bb).sum(dim=-1)

    # 2. base(p) -> back(q)
    r_bk = r_com + back_offsets[q_idx] - base_offsets[p_idx]
    r_bk_sq = (r_bk * r_bk).sum(dim=-1)

    # 3. back(p) -> base(q)
    r_kb = r_com + base_offsets[q_idx] - back_offsets[p_idx]
    r_kb_sq = (r_kb * r_kb).sum(dim=-1)

    energy = (
        repulsive_lj(r_bb_sq, C.EXCL_S2, C.EXCL_R2, C.EXCL_B2, C.EXCL_RC2, excl_eps=excl_eps).sum()
        + repulsive_lj(r_bk_sq, C.EXCL_S3, C.EXCL_R3, C.EXCL_B3, C.EXCL_RC3, excl_eps=excl_eps).sum()
        + repulsive_lj(r_kb_sq, C.EXCL_S4, C.EXCL_R4, C.EXCL_B4, C.EXCL_RC4, excl_eps=excl_eps).sum()
    )

    return energy


def nonbonded_excluded_volume_energy(
    positions: Tensor,
    back_offsets: Tensor,
    base_offsets: Tensor,
    nonbonded_pairs: Tensor,
    box: Optional[Tensor] = None,
    excl_eps: Optional[Tensor] = None,
) -> Tensor:
    """Compute non-bonded excluded volume energy for all non-bonded pairs.

    Four site-pair interactions per non-bonded pair:
      1. base(p) - base(q)
      2. back(p) - base(q)
      3. base(p) - back(q)
      4. back(p) - back(q)

    Args:
        positions: (N, 3) COM positions
        back_offsets: (N, 3) backbone offsets from COM
        base_offsets: (N, 3) base offsets from COM
        nonbonded_pairs: (P, 2) non-bonded pair indices
        box: (3,) periodic box dimensions, or None
        excl_eps: optional learnable excluded volume epsilon

    Returns:
        Scalar total non-bonded excluded volume energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # COM displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)

    # 1. base-base
    r_bb = r_com + base_offsets[q_idx] - base_offsets[p_idx]
    r_bb_sq = (r_bb * r_bb).sum(dim=-1)

    # 2. back(p) - base(q)
    r_kb = r_com + base_offsets[q_idx] - back_offsets[p_idx]
    r_kb_sq = (r_kb * r_kb).sum(dim=-1)

    # 3. base(p) - back(q)
    r_bk = r_com + back_offsets[q_idx] - base_offsets[p_idx]
    r_bk_sq = (r_bk * r_bk).sum(dim=-1)

    # 4. back-back
    r_kk = r_com + back_offsets[q_idx] - back_offsets[p_idx]
    r_kk_sq = (r_kk * r_kk).sum(dim=-1)

    energy = (
        repulsive_lj(r_bb_sq, C.EXCL_S2, C.EXCL_R2, C.EXCL_B2, C.EXCL_RC2, excl_eps=excl_eps).sum()
        + repulsive_lj(r_kb_sq, C.EXCL_S4, C.EXCL_R4, C.EXCL_B4, C.EXCL_RC4, excl_eps=excl_eps).sum()
        + repulsive_lj(r_bk_sq, C.EXCL_S3, C.EXCL_R3, C.EXCL_B3, C.EXCL_RC3, excl_eps=excl_eps).sum()
        + repulsive_lj(r_kk_sq, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1, excl_eps=excl_eps).sum()
    )

    return energy

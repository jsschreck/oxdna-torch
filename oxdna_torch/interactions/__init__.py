"""
oxDNA interaction energy terms.

Each module computes one of the 7 pairwise energy terms:
  Bonded:
    - fene: FENE backbone potential
    - excluded_volume: bonded excluded volume (base-base, base-back)
    - stacking: sequential stacking between bonded neighbors

  Non-bonded:
    - excluded_volume: non-bonded excluded volume (all site pairs)
    - hbond: hydrogen bonding (Watson-Crick pairs)
    - cross_stacking: cross-stacking between bases on different strands
    - coaxial_stacking: coaxial stacking between aligned bases

All energy functions take positions, quaternions, and pair indices
and return scalar energy values. Forces are obtained via autograd.
"""

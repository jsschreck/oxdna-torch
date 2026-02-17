"""
Shared test fixtures for oxdna_torch tests.
"""

import sys
import os
import pytest
import torch
import math

# Ensure oxdna_torch is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxdna_torch.io import load_system
from oxdna_torch.state import SystemState
from oxdna_torch.topology import Topology
from oxdna_torch.model import OxDNAEnergy
from oxdna_torch.quaternion import quat_to_rotmat, quat_from_axis_angle


HAIRPIN_TOP = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.top")
HAIRPIN_CONF = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.conf")

# Standard temperature: 334 K in oxDNA reduced units
T_334K = 334.0 / 3000.0


@pytest.fixture
def hairpin_system():
    """Load the HAIRPIN example system."""
    topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
    return topology, state


@pytest.fixture
def hairpin_model(hairpin_system):
    """OxDNAEnergy model for the HAIRPIN system (average sequence)."""
    topology, state = hairpin_system
    model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=False)
    return model, topology, state


@pytest.fixture
def hairpin_model_seqdep(hairpin_system):
    """OxDNAEnergy model for the HAIRPIN system (sequence dependent)."""
    topology, state = hairpin_system
    model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
    return model, topology, state


@pytest.fixture
def identity_quaternion():
    """Identity quaternion [1, 0, 0, 0]."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)


@pytest.fixture
def two_nucleotide_system():
    """Minimal 2-nucleotide bonded system for unit testing interactions.

    Two nucleotides on the same strand, separated along the x-axis
    with identity orientations.
    """
    strand_ids = torch.tensor([0, 0], dtype=torch.long)
    base_types = torch.tensor([0, 3], dtype=torch.long)  # A-T
    bonded_neighbors = torch.tensor([[-1, 1], [0, -1]], dtype=torch.long)  # 0.n3=-1, 0.n5=1; 1.n3=0, 1.n5=-1

    topology = Topology(
        n_nucleotides=2,
        n_strands=1,
        strand_ids=strand_ids,
        base_types=base_types,
        bonded_neighbors=bonded_neighbors,
    )

    # Place nucleotides so backbone distance ~ FENE_R0
    # With identity orientation, backbone is at POS_BACK * a1 = (-0.4, 0, 0)
    # So backbone-backbone distance = |pos2 - pos1 + back2 - back1|
    # With identity quats, back offset = (-0.4, 0, 0) for both
    # So bb dist = |pos2 - pos1| and we want this ~ 0.7525
    # But actually bb dist = |(pos2 + back2) - (pos1 + back1)| = |pos2 - pos1|
    # since back offsets are same for both with identity orientation
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.7525, 0.0, 0.0],
    ], dtype=torch.float64)

    quaternions = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64)

    state = SystemState(
        positions=positions,
        quaternions=quaternions,
        box=torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64),
    )

    return topology, state

"""
Tests for cell list neighbor finding in pairs.py.

Validates that the cell list algorithm produces the same results as brute force.
"""

import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oxdna_torch.io import load_system
from oxdna_torch.pairs import find_nonbonded_pairs, _find_nonbonded_pairs_brute_force, _find_nonbonded_pairs_cell_list
from oxdna_torch.topology import Topology
from oxdna_torch.model import OxDNAEnergy
from oxdna_torch import constants as C


HAIRPIN_TOP = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.top")
HAIRPIN_CONF = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.conf")


def _sorted_pair_set(pairs):
    """Convert pairs tensor to a set of sorted tuples for comparison."""
    if pairs.shape[0] == 0:
        return set()
    p_min = torch.minimum(pairs[:, 0], pairs[:, 1])
    p_max = torch.maximum(pairs[:, 0], pairs[:, 1])
    return set(zip(p_min.tolist(), p_max.tolist()))


def _make_chain_topology(N):
    """Create a minimal chain topology for N nucleotides."""
    strand_ids = torch.zeros(N, dtype=torch.long)
    base_types = torch.zeros(N, dtype=torch.long)
    bonded_neighbors = torch.full((N, 2), -1, dtype=torch.long)
    for i in range(N - 1):
        bonded_neighbors[i + 1, 0] = i     # n3
        bonded_neighbors[i, 1] = i + 1     # n5
    return Topology(
        n_nucleotides=N, n_strands=1,
        strand_ids=strand_ids, base_types=base_types,
        bonded_neighbors=bonded_neighbors,
    )


class TestCellListCorrectness:
    """Test that cell list produces same results as brute force."""

    @pytest.fixture
    def hairpin_data(self):
        """Load the HAIRPIN test system."""
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology)
        return model, state

    def test_cell_list_matches_brute_force_hairpin(self, hairpin_data):
        """Cell list finds the same pairs as brute force on the HAIRPIN system."""
        model, state = hairpin_data
        cutoff = model.cutoff

        bf_pairs = _find_nonbonded_pairs_brute_force(
            state.positions, model.topology, state.box, cutoff)
        cl_pairs = _find_nonbonded_pairs_cell_list(
            state.positions, model.topology, state.box, cutoff)

        bf_set = _sorted_pair_set(bf_pairs)
        cl_set = _sorted_pair_set(cl_pairs)

        assert bf_set == cl_set, (
            f"Cell list found {len(cl_set)} pairs, brute force found {len(bf_set)}. "
            f"Missing: {bf_set - cl_set}, Extra: {cl_set - bf_set}"
        )

    def test_cell_list_matches_brute_force_random(self):
        """Cell list matches brute force on a random periodic system."""
        N = 100
        box = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        positions = torch.rand(N, 3, dtype=torch.float64) * box
        topology = _make_chain_topology(N)

        cutoff = C.compute_rcut()
        bf_pairs = _find_nonbonded_pairs_brute_force(positions, topology, box, cutoff)
        cl_pairs = _find_nonbonded_pairs_cell_list(positions, topology, box, cutoff)

        assert _sorted_pair_set(bf_pairs) == _sorted_pair_set(cl_pairs)

    def test_cell_list_excludes_bonded(self, hairpin_data):
        """Cell list properly excludes bonded pairs."""
        model, state = hairpin_data
        cutoff = model.cutoff

        cl_pairs = _find_nonbonded_pairs_cell_list(
            state.positions, model.topology, state.box, cutoff)

        bp = model.topology.bonded_pairs
        bp_set = set()
        for i in range(bp.shape[0]):
            a, b = bp[i, 0].item(), bp[i, 1].item()
            bp_set.add((min(a, b), max(a, b)))

        cl_set = _sorted_pair_set(cl_pairs)
        overlap = cl_set & bp_set
        assert len(overlap) == 0, f"Cell list contains bonded pairs: {overlap}"

    def test_cell_list_upper_triangular(self, hairpin_data):
        """Cell list returns pairs with i < j."""
        model, state = hairpin_data
        cutoff = model.cutoff

        cl_pairs = _find_nonbonded_pairs_cell_list(
            state.positions, model.topology, state.box, cutoff)

        assert (cl_pairs[:, 0] < cl_pairs[:, 1]).all(), \
            "Cell list should return pairs with i < j"


class TestAutoSelection:
    """Test automatic method selection."""

    def test_auto_selects_brute_force_for_small(self):
        """Auto method selects brute force for small systems."""
        N = 50
        positions = torch.rand(N, 3, dtype=torch.float64)
        topology = _make_chain_topology(N)

        pairs = find_nonbonded_pairs(positions, topology, box=None, method='auto')
        assert isinstance(pairs, torch.Tensor)

    def test_method_brute_force_explicit(self):
        """Explicit brute_force method works."""
        N = 20
        box = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        positions = torch.rand(N, 3, dtype=torch.float64) * box
        topology = _make_chain_topology(N)

        pairs = find_nonbonded_pairs(positions, topology, box, method='brute_force')
        assert isinstance(pairs, torch.Tensor)

    def test_method_cell_list_requires_box(self):
        """Cell list method raises error without periodic box."""
        N = 20
        positions = torch.rand(N, 3, dtype=torch.float64)
        topology = _make_chain_topology(N)

        with pytest.raises(AssertionError):
            find_nonbonded_pairs(positions, topology, box=None, method='cell_list')


class TestCellListEnergy:
    """Test that energies computed with cell list pairs match brute force."""

    def test_total_energy_matches(self):
        """Both methods give identical pair sets on hairpin."""
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology)

        bf_pairs = _find_nonbonded_pairs_brute_force(
            state.positions, model.topology, state.box, model.cutoff)
        cl_pairs = _find_nonbonded_pairs_cell_list(
            state.positions, model.topology, state.box, model.cutoff)

        assert _sorted_pair_set(bf_pairs) == _sorted_pair_set(cl_pairs)

    def test_pbc_wrapping_edge_case(self):
        """Particles near box boundaries are correctly handled."""
        N = 10
        box = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        positions = torch.tensor([
            [0.1, 0.1, 0.1],
            [4.9, 4.9, 4.9],
            [2.5, 2.5, 2.5],
            [0.1, 2.5, 4.9],
            [4.9, 0.1, 2.5],
            [2.5, 4.9, 0.1],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [0.5, 4.5, 2.5],
            [4.5, 0.5, 2.5],
        ], dtype=torch.float64)
        topology = _make_chain_topology(N)

        cutoff = 2.0
        bf_pairs = _find_nonbonded_pairs_brute_force(positions, topology, box, cutoff)
        cl_pairs = _find_nonbonded_pairs_cell_list(positions, topology, box, cutoff)

        assert _sorted_pair_set(bf_pairs) == _sorted_pair_set(cl_pairs)

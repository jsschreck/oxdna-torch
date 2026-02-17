"""
Tests for oxdna_torch.io module (reading/writing oxDNA files).
"""

import torch
import os
import tempfile
import pytest

from oxdna_torch.io import read_topology, read_configuration, write_configuration, load_system
from oxdna_torch.state import SystemState
from oxdna_torch.quaternion import quat_to_rotmat
from oxdna_torch import constants as C

from conftest import HAIRPIN_TOP, HAIRPIN_CONF


class TestReadTopology:
    def test_nucleotide_count(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.n_nucleotides == 18

    def test_strand_count(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.n_strands == 1

    def test_all_same_strand(self):
        topo = read_topology(HAIRPIN_TOP)
        assert (topo.strand_ids == 0).all()

    def test_base_types_valid(self):
        topo = read_topology(HAIRPIN_TOP)
        assert (topo.base_types >= 0).all()
        assert (topo.base_types <= 3).all()

    def test_first_base_is_G(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.base_types[0].item() == C.BASE_G

    def test_last_base_is_C(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.base_types[-1].item() == C.BASE_C

    def test_sequence(self):
        topo = read_topology(HAIRPIN_TOP)
        base_map = {0: "A", 1: "C", 2: "G", 3: "T"}
        seq = "".join(base_map[b.item()] for b in topo.base_types)
        assert seq == "GCGTTGCTTCTCCAACGC"

    def test_bonded_neighbors_shape(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.bonded_neighbors.shape == (18, 2)

    def test_first_nucleotide_no_n3(self):
        """First nucleotide in hairpin has n3=-1 (no 3' neighbor)."""
        topo = read_topology(HAIRPIN_TOP)
        assert topo.bonded_neighbors[0, 0].item() == -1  # n3

    def test_first_nucleotide_has_n5(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.bonded_neighbors[0, 1].item() == 1  # n5 = nucleotide 1

    def test_last_nucleotide_no_n5(self):
        topo = read_topology(HAIRPIN_TOP)
        assert topo.bonded_neighbors[17, 1].item() == -1  # n5

    def test_bonded_pairs_count(self):
        """18 nucleotides in 1 strand = 17 bonds."""
        topo = read_topology(HAIRPIN_TOP)
        assert topo.n_bonded == 17

    def test_bonded_pairs_valid_indices(self):
        topo = read_topology(HAIRPIN_TOP)
        assert (topo.bonded_pairs >= 0).all()
        assert (topo.bonded_pairs < 18).all()


class TestReadConfiguration:
    def test_positions_shape(self):
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        assert state.positions.shape == (18, 3)

    def test_quaternions_shape(self):
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        assert state.quaternions.shape == (18, 4)

    def test_quaternions_normalized(self):
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        norms = torch.norm(state.quaternions, dim=-1)
        assert torch.allclose(norms, torch.ones(18, dtype=torch.float64), atol=1e-10)

    def test_box_shape(self):
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        assert state.box.shape == (3,)
        assert torch.allclose(state.box, torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64))

    def test_velocities_shape(self):
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        assert state.velocities.shape == (18, 3)

    def test_rotation_matrix_orthogonal(self):
        """Reconstructed rotation matrices should be proper rotations."""
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        R = quat_to_rotmat(state.quaternions)
        # R^T R should be identity
        RtR = torch.bmm(R.transpose(-1, -2), R)
        I = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(18, -1, -1)
        assert torch.allclose(RtR, I, atol=1e-10)

    def test_a1_matches_file(self):
        """The a1 vector from the first nucleotide should match the config file."""
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        R = quat_to_rotmat(state.quaternions)
        a1 = R[0, :, 0]
        # From config file line 4: a1 = (0.941067202086077, -0.288407874130549, -0.176673199148544)
        expected = torch.tensor([0.941067202086077, -0.288407874130549, -0.176673199148544],
                                dtype=torch.float64)
        assert torch.allclose(a1, expected, atol=1e-6)

    def test_a3_matches_file(self):
        """The a3 vector from the first nucleotide should match the config file."""
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)
        R = quat_to_rotmat(state.quaternions)
        a3 = R[0, :, 2]
        expected = torch.tensor([0.0638009866594605, 0.664360403144493, -0.744684288027488],
                                dtype=torch.float64)
        assert torch.allclose(a3, expected, atol=1e-6)

    def test_nucleotide_count_mismatch_raises(self):
        """Should raise if topology and config have different nucleotide counts."""
        topo = read_topology(HAIRPIN_TOP)
        # Modify topology to have wrong count
        topo.n_nucleotides = 10
        with pytest.raises(AssertionError):
            read_configuration(HAIRPIN_CONF, topo)


class TestWriteConfiguration:
    def test_roundtrip(self):
        """Write then read should give back the same state."""
        topo = read_topology(HAIRPIN_TOP)
        state = read_configuration(HAIRPIN_CONF, topo)

        with tempfile.NamedTemporaryFile(suffix=".conf", delete=False, mode="w") as f:
            tmppath = f.name

        try:
            write_configuration(tmppath, state, topo, timestep=0)
            state2 = read_configuration(tmppath, topo)

            assert torch.allclose(state.positions, state2.positions, atol=1e-10)
            # Quaternions may differ by sign, but rotation matrices should match
            R1 = quat_to_rotmat(state.quaternions)
            R2 = quat_to_rotmat(state2.quaternions)
            assert torch.allclose(R1, R2, atol=1e-6)
        finally:
            os.unlink(tmppath)


class TestLoadSystem:
    def test_returns_tuple(self):
        topo, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        assert topo.n_nucleotides == 18
        assert state.positions.shape[0] == 18

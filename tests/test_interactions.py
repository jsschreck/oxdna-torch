"""
Tests for individual interaction energy terms.
"""

import torch
import math
import pytest

from oxdna_torch import constants as C
from oxdna_torch.state import SystemState
from oxdna_torch.topology import Topology
from oxdna_torch.pairs import compute_site_offsets
from oxdna_torch.interactions.fene import fene_energy
from oxdna_torch.interactions.excluded_volume import (
    bonded_excluded_volume_energy,
    nonbonded_excluded_volume_energy,
)
from oxdna_torch.interactions.stacking import stacking_energy
from oxdna_torch.interactions.hbond import hydrogen_bonding_energy
from oxdna_torch.interactions.cross_stacking import cross_stacking_energy
from oxdna_torch.interactions.coaxial_stacking import coaxial_stacking_energy


class TestFENE:
    def test_zero_at_equilibrium(self):
        """FENE energy should be near its minimum at r = r0."""
        # Two nucleotides at exactly FENE_R0 backbone distance
        # With identity orientation, backbone offset = (-0.4, 0, 0)
        # backbone dist = |pos_q + back_q - pos_p - back_p| = |pos_q - pos_p|
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [C.FENE_R0_OXDNA, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = fene_energy(positions, back_off, bonded_pairs)
        # At equilibrium (r - r0 = 0), FENE = -(eps/2)*ln(1) = 0
        assert abs(energy.item()) < 1e-10, f"FENE at equilibrium should be ~0, got {energy.item()}"

    def test_positive_away_from_equilibrium(self):
        """FENE energy should be positive away from equilibrium."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [C.FENE_R0_OXDNA + 0.1, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = fene_energy(positions, back_off, bonded_pairs)
        assert energy.item() > 0, f"FENE away from eq should be positive, got {energy.item()}"

    def test_increases_with_stretch(self):
        """FENE energy should increase as we stretch the bond."""
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energies = []
        for delta in [0.05, 0.10, 0.15, 0.20]:
            positions = torch.tensor([
                [0.0, 0.0, 0.0],
                [C.FENE_R0_OXDNA + delta, 0.0, 0.0],
            ], dtype=torch.float64)
            e = fene_energy(positions, back_off, bonded_pairs)
            energies.append(e.item())

        for i in range(len(energies) - 1):
            assert energies[i + 1] > energies[i], \
                f"FENE should increase with stretch: {energies}"

    def test_diverges_near_delta(self):
        """FENE should become very large near FENE_DELTA."""
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [C.FENE_R0_OXDNA + C.FENE_DELTA - 0.001, 0.0, 0.0],
        ], dtype=torch.float64)
        energy = fene_energy(positions, back_off, bonded_pairs)
        assert energy.item() > 4.0, f"FENE near delta should be very large, got {energy.item()}"

    def test_differentiable(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [C.FENE_R0_OXDNA + 0.05, 0.0, 0.0],
        ], dtype=torch.float64).requires_grad_(True)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = fene_energy(positions, back_off, bonded_pairs)
        energy.backward()
        assert not torch.isnan(positions.grad).any()

    def test_force_direction(self):
        """Force should pull nucleotides together when stretched."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [C.FENE_R0_OXDNA + 0.1, 0.0, 0.0],
        ], dtype=torch.float64).requires_grad_(True)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, _ = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = fene_energy(positions, back_off, bonded_pairs)
        energy.backward()
        forces = -positions.grad
        # Force on q (index 1) should point toward p (negative x)
        assert forces[1, 0].item() < 0, "Force should pull stretched bond back"

    def test_multiple_pairs(self, hairpin_system):
        """Should handle all bonded pairs in a real system."""
        topology, state = hairpin_system
        back_off, _, _ = compute_site_offsets(state.quaternions)
        energy = fene_energy(state.positions, back_off, topology.bonded_pairs, state.box)
        assert energy.item() > 0  # FENE is always >= 0
        assert torch.isfinite(torch.tensor(energy.item()))


class TestBondedExcludedVolume:
    def test_zero_when_well_separated(self):
        """Excluded volume should be zero when sites are far apart."""
        # Place nucleotides far enough that all site-site distances > rc
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # very far apart
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, base_off = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = bonded_excluded_volume_energy(positions, back_off, base_off, bonded_pairs)
        assert energy.item() == 0.0

    def test_positive_when_overlapping(self):
        """Excluded volume should be positive when sites overlap."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],  # bases will overlap
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, base_off = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = bonded_excluded_volume_energy(positions, back_off, base_off, bonded_pairs)
        assert energy.item() > 0

    def test_differentiable(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=torch.float64).requires_grad_(True)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, base_off = compute_site_offsets(quaternions)
        bonded_pairs = torch.tensor([[1, 0]], dtype=torch.long)

        energy = bonded_excluded_volume_energy(positions, back_off, base_off, bonded_pairs)
        energy.backward()
        assert not torch.isnan(positions.grad).any()


class TestNonbondedExcludedVolume:
    def test_zero_for_empty_pairs(self):
        """Should return 0 when no pairs are provided."""
        positions = torch.randn(5, 3, dtype=torch.float64)
        quaternions = torch.tensor([[1, 0, 0, 0]] * 5, dtype=torch.float64)
        back_off, _, base_off = compute_site_offsets(quaternions)
        empty_pairs = torch.zeros(0, 2, dtype=torch.long)
        energy = nonbonded_excluded_volume_energy(positions, back_off, base_off, empty_pairs)
        assert energy.item() == 0.0

    def test_includes_back_back(self):
        """Non-bonded excl vol should include back-back interaction (extra vs bonded)."""
        # Place two nucleotides where back-back distance < EXCL_RC1
        # but other site pairs are far enough
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        back_off, _, base_off = compute_site_offsets(quaternions)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)

        energy = nonbonded_excluded_volume_energy(positions, back_off, base_off, pairs)
        # Should be non-negative (repulsive only)
        assert energy.item() >= 0


class TestStacking:
    def test_negative_for_good_geometry(self, hairpin_system):
        """Stacking energy should be negative for a reasonable configuration."""
        topology, state = hairpin_system
        _, stack_off, _ = compute_site_offsets(state.quaternions)
        T = 334.0 / 3000.0
        stacking_eps = topology.compute_stacking_eps(T, seq_dependent=False)

        energy = stacking_energy(
            state.positions, state.quaternions, stack_off,
            topology.bonded_pairs, stacking_eps, state.box,
        )
        assert energy.item() < 0, f"Stacking should be attractive, got {energy.item()}"

    def test_differentiable(self, hairpin_system):
        topology, state = hairpin_system
        pos = state.positions.clone().detach().requires_grad_(True)
        quat = state.quaternions.clone().detach().requires_grad_(True)
        _, stack_off, _ = compute_site_offsets(quat)
        T = 334.0 / 3000.0
        stacking_eps = topology.compute_stacking_eps(T, seq_dependent=False)

        energy = stacking_energy(
            pos, quat, stack_off,
            topology.bonded_pairs, stacking_eps, state.box,
        )
        energy.backward()
        assert not torch.isnan(pos.grad).any()
        assert not torch.isnan(quat.grad).any()

    def test_seq_dependent_differs_from_average(self, hairpin_system):
        """Sequence-dependent stacking should differ from average."""
        topology, state = hairpin_system
        _, stack_off, _ = compute_site_offsets(state.quaternions)
        T = 334.0 / 3000.0

        eps_avg = topology.compute_stacking_eps(T, seq_dependent=False)
        eps_seq = topology.compute_stacking_eps(T, seq_dependent=True)

        e_avg = stacking_energy(
            state.positions, state.quaternions, stack_off,
            topology.bonded_pairs, eps_avg, state.box,
        )
        e_seq = stacking_energy(
            state.positions, state.quaternions, stack_off,
            topology.bonded_pairs, eps_seq, state.box,
        )
        assert abs(e_avg.item() - e_seq.item()) > 0.01, \
            "Seq-dep and average stacking should differ meaningfully"


class TestHydrogenBonding:
    def test_zero_for_non_wc_pairs(self):
        """HB should be zero for non-Watson-Crick pairs."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        _, _, base_off = compute_site_offsets(quaternions)
        # A-A pair (not WC: 0+0 != 3)
        base_types = torch.tensor([0, 0], dtype=torch.long)
        hbond_eps = torch.zeros(4, 4, dtype=torch.float64)
        hbond_eps[0, 3] = hbond_eps[3, 0] = C.HYDR_EPS_OXDNA
        hbond_eps[1, 2] = hbond_eps[2, 1] = C.HYDR_EPS_OXDNA

        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        energy = hydrogen_bonding_energy(
            positions, quaternions, base_off, pairs, base_types, hbond_eps,
        )
        assert energy.item() == 0.0

    def test_zero_for_empty_pairs(self):
        positions = torch.randn(3, 3, dtype=torch.float64)
        quaternions = torch.tensor([[1, 0, 0, 0]] * 3, dtype=torch.float64)
        _, _, base_off = compute_site_offsets(quaternions)
        base_types = torch.tensor([0, 1, 2], dtype=torch.long)
        hbond_eps = torch.zeros(4, 4, dtype=torch.float64)
        empty = torch.zeros(0, 2, dtype=torch.long)
        energy = hydrogen_bonding_energy(
            positions, quaternions, base_off, empty, base_types, hbond_eps,
        )
        assert energy.item() == 0.0

    def test_zero_outside_range(self):
        """HB should be zero when bases are too far apart."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        _, _, base_off = compute_site_offsets(quaternions)
        base_types = torch.tensor([0, 3], dtype=torch.long)  # A-T
        hbond_eps = torch.zeros(4, 4, dtype=torch.float64)
        hbond_eps[0, 3] = hbond_eps[3, 0] = C.HYDR_EPS_OXDNA
        pairs = torch.tensor([[0, 1]], dtype=torch.long)

        energy = hydrogen_bonding_energy(
            positions, quaternions, base_off, pairs, base_types, hbond_eps,
        )
        assert energy.item() == 0.0


class TestCrossStacking:
    def test_zero_for_empty_pairs(self):
        positions = torch.randn(3, 3, dtype=torch.float64)
        quaternions = torch.tensor([[1, 0, 0, 0]] * 3, dtype=torch.float64)
        _, _, base_off = compute_site_offsets(quaternions)
        empty = torch.zeros(0, 2, dtype=torch.long)
        energy = cross_stacking_energy(positions, quaternions, base_off, empty)
        assert energy.item() == 0.0

    def test_zero_outside_range(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        _, _, base_off = compute_site_offsets(quaternions)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        energy = cross_stacking_energy(positions, quaternions, base_off, pairs)
        assert energy.item() == 0.0


class TestCoaxialStacking:
    def test_zero_for_empty_pairs(self):
        positions = torch.randn(3, 3, dtype=torch.float64)
        quaternions = torch.tensor([[1, 0, 0, 0]] * 3, dtype=torch.float64)
        _, stack_off, _ = compute_site_offsets(quaternions)
        empty = torch.zeros(0, 2, dtype=torch.long)
        energy = coaxial_stacking_energy(positions, quaternions, stack_off, empty)
        assert energy.item() == 0.0

    def test_zero_outside_range(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=torch.float64)
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        _, stack_off, _ = compute_site_offsets(quaternions)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        energy = coaxial_stacking_energy(positions, quaternions, stack_off, pairs)
        assert energy.item() == 0.0

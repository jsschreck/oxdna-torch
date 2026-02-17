"""
Tests for the full OxDNAEnergy model.
"""

import torch
import math
import pytest

from oxdna_torch import constants as C
from oxdna_torch.model import OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.topology import Topology

from conftest import T_334K


class TestEnergyReference:
    """Compare computed energies against the reference oxDNA binary output."""

    def test_hairpin_total_energy_avg_seq(self, hairpin_model):
        """Total energy should match the reference from the config file header.

        The HAIRPIN initial.conf header says:
            E = -0.365026473198196 -0.365026473198196 0
        This is Epot per nucleotide (average-sequence model at 334K).
        """
        model, topology, state = hairpin_model
        energy = model(SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        ))
        per_nucl = energy.item() / topology.n_nucleotides
        ref_per_nucl = -0.365026473198196
        assert abs(per_nucl - ref_per_nucl) < 1e-4, \
            f"Per-nucleotide energy: {per_nucl:.8f} vs reference {ref_per_nucl:.8f}"

    def test_hairpin_energy_components_sum(self, hairpin_model):
        """Sum of individual components should equal total energy."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        total = model(s)
        components = model.energy_components(s)
        comp_sum = sum(components.values())
        assert abs(total.item() - comp_sum.item()) < 1e-10

    def test_fene_positive(self, hairpin_model):
        """FENE energy should be positive for a real config."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        components = model.energy_components(s)
        assert components['fene'].item() > 0

    def test_stacking_negative(self, hairpin_model):
        """Stacking energy should be negative (attractive) for a real config."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        components = model.energy_components(s)
        assert components['stacking'].item() < 0

    def test_excl_vol_non_negative(self, hairpin_model):
        """Excluded volume should be >= 0 (purely repulsive)."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        components = model.energy_components(s)
        assert components['bonded_excl_vol'].item() >= 0
        assert components['nonbonded_excl_vol'].item() >= 0


class TestFiniteDifferences:
    """Verify autograd forces against numerical finite differences."""

    def test_position_gradient_accuracy(self, hairpin_model):
        """Autograd force should match finite-difference force to high precision."""
        model, topology, state = hairpin_model
        eps = 1e-6

        # Autograd
        pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=state.quaternions.detach(), box=state.box)
        energy = model(s)
        energy.backward()
        ag_grad = pos.grad.clone()

        # Check several nucleotides and dimensions
        for nucl_idx in [0, 5, 10, 17]:
            for dim in [0, 1, 2]:
                p_plus = state.positions.clone().detach()
                p_plus[nucl_idx, dim] += eps
                e_plus = model(SystemState(
                    positions=p_plus, quaternions=state.quaternions.detach(), box=state.box
                )).item()

                p_minus = state.positions.clone().detach()
                p_minus[nucl_idx, dim] -= eps
                e_minus = model(SystemState(
                    positions=p_minus, quaternions=state.quaternions.detach(), box=state.box
                )).item()

                fd_grad = (e_plus - e_minus) / (2 * eps)
                ag_val = ag_grad[nucl_idx, dim].item()

                rel_err = abs(fd_grad - ag_val) / (abs(ag_val) + 1e-10)
                assert rel_err < 1e-5, \
                    f"Gradient mismatch at nucl={nucl_idx} dim={dim}: " \
                    f"FD={fd_grad:.8f} AG={ag_val:.8f} rel_err={rel_err:.2e}"

    def test_quaternion_gradient_accuracy(self, hairpin_model):
        """Quaternion gradients should match finite differences."""
        model, topology, state = hairpin_model
        eps = 1e-6

        # Autograd
        quat = state.quaternions.clone().detach().requires_grad_(True)
        s = SystemState(positions=state.positions.detach(), quaternions=quat, box=state.box)
        energy = model(s)
        energy.backward()
        ag_grad = quat.grad.clone()

        # Check a few quaternion components
        for nucl_idx in [0, 8]:
            for dim in [0, 1, 2, 3]:
                q_plus = state.quaternions.clone().detach()
                q_plus[nucl_idx, dim] += eps
                # Renormalize
                q_plus[nucl_idx] = q_plus[nucl_idx] / q_plus[nucl_idx].norm()
                e_plus = model(SystemState(
                    positions=state.positions.detach(), quaternions=q_plus, box=state.box
                )).item()

                q_minus = state.quaternions.clone().detach()
                q_minus[nucl_idx, dim] -= eps
                q_minus[nucl_idx] = q_minus[nucl_idx] / q_minus[nucl_idx].norm()
                e_minus = model(SystemState(
                    positions=state.positions.detach(), quaternions=q_minus, box=state.box
                )).item()

                fd_grad = (e_plus - e_minus) / (2 * eps)
                ag_val = ag_grad[nucl_idx, dim].item()

                # Use slightly looser tolerance because of renormalization
                abs_err = abs(fd_grad - ag_val)
                assert abs_err < 1e-3, \
                    f"Quat gradient mismatch at nucl={nucl_idx} dim={dim}: " \
                    f"FD={fd_grad:.6f} AG={ag_val:.6f} abs_err={abs_err:.2e}"


class TestGradientProperties:
    """Test gradient flow and stability properties."""

    def test_no_nan_gradients(self, hairpin_model):
        """No NaN gradients for a normal configuration."""
        model, topology, state = hairpin_model
        pos = state.positions.clone().detach().requires_grad_(True)
        quat = state.quaternions.clone().detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=quat, box=state.box)
        energy = model(s)
        energy.backward()
        assert not torch.isnan(pos.grad).any(), "NaN in position gradients"
        assert not torch.isnan(quat.grad).any(), "NaN in quaternion gradients"

    def test_no_inf_gradients(self, hairpin_model):
        """No Inf gradients for a normal configuration."""
        model, topology, state = hairpin_model
        pos = state.positions.clone().detach().requires_grad_(True)
        quat = state.quaternions.clone().detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=quat, box=state.box)
        energy = model(s)
        energy.backward()
        assert not torch.isinf(pos.grad).any(), "Inf in position gradients"
        assert not torch.isinf(quat.grad).any(), "Inf in quaternion gradients"

    def test_gradient_magnitude_reasonable(self, hairpin_model):
        """Gradient magnitudes should be in a reasonable range."""
        model, topology, state = hairpin_model
        pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=state.quaternions.detach(), box=state.box)
        energy = model(s)
        energy.backward()
        max_grad = pos.grad.abs().max().item()
        assert max_grad < 1e6, f"Gradient too large: {max_grad}"
        assert max_grad > 1e-6, f"Gradient suspiciously small: {max_grad}"

    def test_energy_finite(self, hairpin_model):
        """Energy should be finite for a physical configuration."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        energy = model(s)
        assert torch.isfinite(energy), f"Energy not finite: {energy.item()}"


class TestTemperature:
    def test_temperature_changes_stacking(self, hairpin_system):
        """Different temperatures should give different stacking energies."""
        topology, state = hairpin_system
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )

        model_cold = OxDNAEnergy(topology, temperature=300 / 3000, seq_dependent=False)
        model_hot = OxDNAEnergy(topology, temperature=370 / 3000, seq_dependent=False)

        e_cold = model_cold.energy_components(s)['stacking'].item()
        e_hot = model_hot.energy_components(s)['stacking'].item()

        assert e_cold != e_hot, "Temperature should affect stacking energy"

    def test_update_temperature(self, hairpin_model):
        """update_temperature should change the energy."""
        model, topology, state = hairpin_model
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )

        e1 = model(s).item()
        model.update_temperature(370 / 3000)
        e2 = model(s).item()
        assert e1 != e2, "Updating temperature should change energy"


class TestSequenceDependence:
    def test_seq_dep_differs(self, hairpin_system):
        """Sequence-dependent and average models should give different energies."""
        topology, state = hairpin_system
        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )

        model_avg = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=False)
        model_seq = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)

        e_avg = model_avg(s).item()
        e_seq = model_seq(s).item()
        assert abs(e_avg - e_seq) > 0.01, \
            f"Seq-dep ({e_seq}) and average ({e_avg}) should differ"


class TestComputeForces:
    def test_compute_forces_shape(self, hairpin_model):
        model, topology, state = hairpin_model
        forces = model.compute_forces(state)
        assert forces.shape == (topology.n_nucleotides, 3)

    def test_compute_forces_no_nan(self, hairpin_model):
        model, topology, state = hairpin_model
        forces = model.compute_forces(state)
        assert not torch.isnan(forces).any()

    def test_compute_forces_matches_autograd(self, hairpin_model):
        """compute_forces should give same result as manual autograd."""
        model, topology, state = hairpin_model

        # Via compute_forces
        forces1 = model.compute_forces(state)

        # Via manual autograd
        pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=state.quaternions.detach(), box=state.box)
        energy = model(s)
        energy.backward()
        forces2 = -pos.grad

        assert torch.allclose(forces1, forces2, atol=1e-10)

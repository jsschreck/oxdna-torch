"""
Tests for the integrator module (dynamics and backprop through time).
"""

import torch
import pytest

from oxdna_torch.model import OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.integrator import LangevinIntegrator

from conftest import T_334K


class TestLangevinIntegrator:
    def test_step_returns_state(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s_new = integrator.step(s, stochastic=False)

        assert s_new.positions.shape == s.positions.shape
        assert s_new.quaternions.shape == s.quaternions.shape

    def test_step_changes_positions(self, hairpin_model):
        """A step should change positions (non-zero forces)."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s_new = integrator.step(s, stochastic=False)

        # Positions should have changed
        diff = (s_new.positions - s.positions).abs().max().item()
        assert diff > 1e-10, "Positions didn't change after a step"

    def test_quaternions_remain_normalized(self, hairpin_model):
        """Quaternions should remain unit-length after integration."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for _ in range(5):
            s = integrator.step(s, stochastic=True)

        norms = torch.norm(s.quaternions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10), \
            f"Quaternion norms: {norms}"

    def test_deterministic_without_noise(self, hairpin_model):
        """Two runs without noise should give identical results."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s1 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s1 = integrator.step(s1, stochastic=False)

        s2 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s2 = integrator.step(s2, stochastic=False)

        assert torch.allclose(s1.positions, s2.positions, atol=1e-12)
        assert torch.allclose(s1.quaternions, s2.quaternions, atol=1e-12)

    def test_stochastic_gives_different_results(self, hairpin_model):
        """Two runs with noise should give different results."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        torch.manual_seed(42)
        s1 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s1 = integrator.step(s1, stochastic=True)

        torch.manual_seed(123)
        s2 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s2 = integrator.step(s2, stochastic=True)

        diff = (s1.positions - s2.positions).abs().max().item()
        assert diff > 1e-6, "Stochastic steps should differ with different seeds"

    def test_energy_finite_after_steps(self, hairpin_model):
        """Energy should remain finite after a few small steps."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.0005, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for _ in range(5):
            s = integrator.step(s, stochastic=False)

        energy = model(SystemState(
            positions=s.positions.detach(),
            quaternions=s.quaternions.detach(),
            box=s.box,
        ))
        assert torch.isfinite(energy), f"Energy not finite after 5 steps: {energy.item()}"


class TestBackpropThroughTime:
    """The key feature: gradients flowing through integration steps."""

    def test_grad_flows_to_initial_positions(self, hairpin_model):
        """Gradients from final energy should reach initial positions."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        init_pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(
            positions=init_pos,
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )

        # 3 deterministic steps
        for _ in range(3):
            s = integrator.step(s, stochastic=False)

        final_energy = model(s)
        final_energy.backward()

        assert init_pos.grad is not None, "No gradient on initial positions"
        assert not torch.isnan(init_pos.grad).any(), "NaN in backprop-through-time gradient"
        assert init_pos.grad.abs().max().item() > 1e-6, "Gradient suspiciously small"

    def test_grad_no_nan_multiple_steps(self, hairpin_model):
        """Gradients should be clean after multiple steps."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.0005, gamma=1.0, temperature=T_334K)

        init_pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(
            positions=init_pos,
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )

        for _ in range(5):
            s = integrator.step(s, stochastic=False)

        final_energy = model(s)
        final_energy.backward()

        assert not torch.isnan(init_pos.grad).any()
        assert not torch.isinf(init_pos.grad).any()

    def test_different_initial_positions_give_different_grads(self, hairpin_model):
        """Perturbing initial positions should change the gradient."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        # Run 1
        pos1 = state.positions.clone().detach().requires_grad_(True)
        s1 = SystemState(positions=pos1, quaternions=state.quaternions.detach(), box=state.box)
        for _ in range(2):
            s1 = integrator.step(s1, stochastic=False)
        e1 = model(s1)
        e1.backward()
        grad1 = pos1.grad.clone()

        # Run 2: slightly perturbed initial positions
        pos2_data = state.positions.clone().detach()
        pos2_data[0, 0] += 0.01
        pos2 = pos2_data.requires_grad_(True)
        s2 = SystemState(positions=pos2, quaternions=state.quaternions.detach(), box=state.box)
        for _ in range(2):
            s2 = integrator.step(s2, stochastic=False)
        e2 = model(s2)
        e2.backward()
        grad2 = pos2.grad.clone()

        diff = (grad1 - grad2).abs().max().item()
        assert diff > 1e-6, "Gradients should differ for different initial conditions"

    def test_can_optimize_positions(self, hairpin_model):
        """Demonstrate that we can minimize energy by gradient descent on positions."""
        model, topology, state = hairpin_model

        # Start with a slightly perturbed config
        pos = state.positions.clone().detach()
        pos += torch.randn_like(pos) * 0.01
        pos = pos.requires_grad_(True)

        optimizer = torch.optim.SGD([pos], lr=0.001)

        initial_energy = model(SystemState(
            positions=pos, quaternions=state.quaternions.detach(), box=state.box
        )).item()

        for _ in range(10):
            optimizer.zero_grad()
            energy = model(SystemState(
                positions=pos, quaternions=state.quaternions.detach(), box=state.box
            ))
            energy.backward()
            optimizer.step()

        final_energy = model(SystemState(
            positions=pos, quaternions=state.quaternions.detach(), box=state.box
        )).item()

        assert final_energy < initial_energy, \
            f"Gradient descent should lower energy: {initial_energy:.4f} -> {final_energy:.4f}"


class TestRollout:
    def test_rollout_returns_trajectory(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        trajectory = integrator.rollout(s, n_steps=5, stochastic=False, save_every=1)

        # Should have initial + 5 states
        assert len(trajectory) == 6

    def test_rollout_save_every(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        trajectory = integrator.rollout(s, n_steps=10, stochastic=False, save_every=5)

        # initial + steps 5 and 10
        assert len(trajectory) == 3

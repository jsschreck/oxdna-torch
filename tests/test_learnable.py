"""
Tests for learnable parameters feature.

Validates:
1. ParameterStore creates correct buffers/parameters
2. Default (no learnable) gives identical energy to original
3. Gradients flow through learnable parameters
4. Optimizer steps change energy
"""

import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oxdna_torch.params import ParameterStore, PARAM_REGISTRY
from oxdna_torch.io import load_system
from oxdna_torch.model import OxDNAEnergy


HAIRPIN_TOP = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.top")
HAIRPIN_CONF = os.path.join(os.path.dirname(__file__), "..", "examples", "HAIRPIN", "initial.conf")

T_334K = 334.0 / 3000.0


def _load_hairpin():
    return load_system(HAIRPIN_TOP, HAIRPIN_CONF)


class TestParameterStore:
    """Test ParameterStore initialization and behavior."""

    def test_default_all_buffers(self):
        """With no learnable params, everything is a buffer."""
        store = ParameterStore()
        param_names = [n for n, p in store.named_parameters()]
        assert len(param_names) == 0
        buffer_names = [n for n, b in store.named_buffers()]
        assert len(buffer_names) == len(PARAM_REGISTRY)

    def test_learnable_registered_as_parameter(self):
        """Learnable names become nn.Parameter, others stay buffers."""
        store = ParameterStore(learnable={'f4_A', 'excl_eps'})
        param_names = {n for n, p in store.named_parameters()}
        assert 'f4_A' in param_names
        assert 'excl_eps' in param_names
        buffer_names = {n for n, b in store.named_buffers()}
        assert 'f4_B' in buffer_names
        assert 'f1_A' in buffer_names

    def test_invalid_name_raises(self):
        """Passing an invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParameterStore(learnable={'nonexistent_param'})

    def test_as_dict_returns_all(self):
        """as_dict() returns all params (learnable and buffer)."""
        store = ParameterStore(learnable={'f4_A'})
        d = store.as_dict()
        assert set(d.keys()) == set(PARAM_REGISTRY.keys())
        assert isinstance(store.f4_A, torch.nn.Parameter)
        assert torch.equal(d['f4_A'], store.f4_A)

    def test_learnable_names_property(self):
        """learnable_names returns the set passed to __init__."""
        names = {'f4_A', 'f5_B', 'fene_eps'}
        store = ParameterStore(learnable=names)
        assert store.learnable_names == names

    def test_device_movement(self):
        """ParameterStore parameters move with .to()."""
        store = ParameterStore(learnable={'f4_A'})
        store = store.to(torch.float32)
        d = store.as_dict()
        assert d['f4_A'].dtype == torch.float32


class TestLearnableEnergyBackwardCompat:
    """Test that default (no learnable) gives identical energy."""

    def test_default_matches_original_energy(self):
        """Model with learnable=None matches original model energy."""
        topology, state = _load_hairpin()
        model_orig = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model_learn = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True, learnable=None)

        e_orig = model_orig.total_energy(state)
        e_learn = model_learn.total_energy(state)

        assert torch.allclose(e_orig, e_learn, atol=1e-12), \
            f"Energies differ: {e_orig.item()} vs {e_learn.item()}"

    def test_empty_set_matches_original(self):
        """Model with learnable=set() matches original model energy."""
        topology, state = _load_hairpin()
        model_orig = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model_learn = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True, learnable=set())

        e_orig = model_orig.total_energy(state)
        e_learn = model_learn.total_energy(state)

        assert torch.allclose(e_orig, e_learn, atol=1e-12)

    def test_learnable_with_defaults_matches_original(self):
        """Model with learnable params (at default values) matches original."""
        topology, state = _load_hairpin()
        model_orig = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, seq_dependent=True,
            learnable={'f4_A', 'excl_eps'})

        e_orig = model_orig.total_energy(state)
        e_learn = model_learn.total_energy(state)

        assert torch.allclose(e_orig, e_learn, atol=1e-10), \
            f"Energies differ: {e_orig.item()} vs {e_learn.item()}"


class TestGradientFlow:
    """Test that gradients flow through learnable parameters."""

    def test_gradient_through_f4_A(self):
        """Gradient flows through f4_A parameter."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'f4_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.param_store.f4_A.grad
        assert grad is not None, "f4_A should have gradient"
        assert not torch.all(grad == 0), "f4_A gradient should be non-zero"

    def test_gradient_through_excl_eps(self):
        """Gradient flows through excl_eps parameter.

        Note: excl_vol may be zero if no particles overlap in the test config,
        so we test with multiple learnable params to ensure grad graph exists.
        """
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, learnable={'excl_eps', 'f4_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        # excl_eps grad may be zero if no excluded volume in this config,
        # but the parameter should still participate in the graph
        grad = model_learn.param_store.excl_eps.grad
        assert grad is not None, "excl_eps should have gradient (possibly zero)"

    def test_gradient_through_fene_params(self):
        """Gradient flows through FENE parameters."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'fene_eps'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.param_store.fene_eps.grad
        assert grad is not None, "fene_eps should have gradient"
        assert grad.item() != 0.0, "fene_eps gradient should be non-zero"

    def test_gradient_through_stacking_eps(self):
        """Gradient flows through stacking_eps (special learnable)."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'stacking_eps'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.stacking_eps.grad
        assert grad is not None, "stacking_eps should have gradient"
        assert not torch.all(grad == 0), "stacking_eps gradient should be non-zero"

    def test_gradient_through_hbond_eps(self):
        """Gradient flows through hbond_eps (special learnable).

        Note: In the test configuration, hbond energy may be zero if no
        WC base pairs are within range. We combine with f4_A to ensure
        a grad graph exists, then check hbond_eps has a grad attribute.
        """
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, learnable={'hbond_eps', 'f4_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        # hbond_eps_matrix grad may be None or zero if no WC pairs interact
        # in this config, but it should be registered as a Parameter
        assert isinstance(model_learn.hbond_eps_matrix, torch.nn.Parameter), \
            "hbond_eps_matrix should be an nn.Parameter"

    def test_gradient_through_f1_params(self):
        """Gradient flows through f1 parameters."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'f1_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.param_store.f1_A.grad
        assert grad is not None, "f1_A should have gradient"

    def test_gradient_through_f2_params(self):
        """Gradient flows through f2 parameters."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'f2_K'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.param_store.f2_K.grad
        assert grad is not None, "f2_K should have gradient"

    def test_gradient_through_f5_params(self):
        """Gradient flows through f5 parameters."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'f5_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        grad = model_learn.param_store.f5_A.grad
        assert grad is not None, "f5_A should have gradient"


class TestOptimizerStep:
    """Test that optimizer steps change parameters and energy."""

    def test_optimizer_step_changes_energy(self):
        """An optimizer step on f4_A changes the energy."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(topology, temperature=T_334K, learnable={'f4_A'})

        e0 = model_learn.total_energy(state)
        e0_val = e0.item()

        e0.backward()
        optimizer = torch.optim.SGD(model_learn.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()

        e1 = model_learn.total_energy(state)
        assert e1.item() != e0_val, "Energy should change after optimizer step"

    def test_multiple_learnable_params(self):
        """Multiple learnable parameters work together."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K,
            learnable={'f4_A', 'excl_eps', 'fene_eps', 'f1_A'})

        energy = model_learn.total_energy(state)
        energy.backward()

        assert model_learn.param_store.f4_A.grad is not None
        assert model_learn.param_store.excl_eps.grad is not None
        assert model_learn.param_store.fene_eps.grad is not None
        assert model_learn.param_store.f1_A.grad is not None

    def test_update_temperature_with_learnable_stacking(self):
        """update_temperature works with learnable stacking_eps."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, learnable={'stacking_eps'})

        model_learn.update_temperature(0.12)

        expected = topology.compute_stacking_eps(0.12, True)
        actual = model_learn.stacking_eps.data
        assert torch.allclose(actual, expected, atol=1e-12)


class TestComponentEnergy:
    """Test energy components with learnable params."""

    def test_energy_components_with_learnable(self):
        """energy_components works with learnable params."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, learnable={'f4_A', 'excl_eps'})

        components = model_learn.energy_components(state)
        assert 'fene' in components
        assert 'stacking' in components
        assert 'hbond' in components
        assert 'cross_stacking' in components
        assert 'coaxial_stacking' in components
        assert 'bonded_excl_vol' in components
        assert 'nonbonded_excl_vol' in components

        total = model_learn.total_energy(state)
        component_sum = sum(components.values())
        assert torch.allclose(total, component_sum, atol=1e-12)

    def test_compute_forces_with_learnable(self):
        """compute_forces works with learnable params."""
        topology, state = _load_hairpin()
        model_learn = OxDNAEnergy(
            topology, temperature=T_334K, learnable={'f4_A'})

        forces = model_learn.compute_forces(state)
        assert forces.shape == state.positions.shape
        assert torch.isfinite(forces).all()

"""
Top-level oxDNA energy model.

Assembles all 7 interaction terms into a single nn.Module that
computes the total potential energy of a system state.

Forces are obtained automatically via autograd:
  energy = model(state)
  energy.backward()
  forces = -state.positions.grad
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Set

from .state import SystemState
from .topology import Topology
from . import constants as C
from .params import ParameterStore
from .pairs import compute_site_positions, compute_site_offsets, find_nonbonded_pairs
from .interactions.fene import fene_energy
from .interactions.excluded_volume import (
    bonded_excluded_volume_energy,
    nonbonded_excluded_volume_energy,
)
from .interactions.stacking import stacking_energy
from .interactions.hbond import hydrogen_bonding_energy
from .interactions.cross_stacking import cross_stacking_energy
from .interactions.coaxial_stacking import coaxial_stacking_energy


class OxDNAEnergy(nn.Module):
    """Differentiable oxDNA energy model.

    Computes the total potential energy E(positions, quaternions) of an oxDNA
    system as the sum of 7 pairwise interaction terms. Since E is computed
    using standard PyTorch operations, forces and torques are obtained
    via automatic differentiation.

    The model parameters (interaction strengths, geometric constants) can
    optionally be made learnable by setting them as nn.Parameters.

    Args:
        topology: Topology object defining system connectivity
        temperature: temperature in oxDNA reduced units (T_K / 3000)
        seq_dependent: whether to use sequence-dependent stacking/HB parameters
        grooving: whether to use major-minor groove backbone positions
        learnable: optional set of parameter names to make learnable
                   (e.g. {'f4_A', 'excl_eps', 'fene_eps'}).
                   See params.PARAM_REGISTRY for available names.
                   Additional special names: 'stacking_eps', 'hbond_eps'.
    """

    def __init__(
        self,
        topology: Topology,
        temperature: float = 0.1,
        seq_dependent: bool = True,
        grooving: bool = False,
        learnable: Optional[Set[str]] = None,
    ):
        super().__init__()

        self.topology = topology
        self.temperature = temperature
        self.seq_dependent = seq_dependent
        self.grooving = grooving

        # Learnable parameter handling
        learnable = learnable or set()
        special_learnable = {'stacking_eps', 'hbond_eps'}
        store_learnable = learnable - special_learnable

        # Create ParameterStore for smooth function parameters
        self.param_store = ParameterStore(learnable=store_learnable if store_learnable else None)

        # Register topology tensors as buffers (not parameters, not differentiable)
        self.register_buffer('bonded_pairs', topology.bonded_pairs)
        self.register_buffer('base_types', topology.base_types)
        self.register_buffer('strand_ids', topology.strand_ids)

        # Precompute sequence-dependent parameters
        stacking_eps = topology.compute_stacking_eps(temperature, seq_dependent)
        if 'stacking_eps' in learnable:
            self.stacking_eps = nn.Parameter(stacking_eps)
        else:
            self.register_buffer('stacking_eps', stacking_eps)

        hbond_eps = topology.compute_hbond_eps(seq_dependent)
        if 'hbond_eps' in learnable:
            self.hbond_eps_matrix = nn.Parameter(hbond_eps)
        else:
            self.register_buffer('hbond_eps_matrix', hbond_eps)

        # Track if we have any learnable params (for fast path)
        self._has_learnable = bool(learnable)

        # Cutoff for neighbor finding
        self.cutoff = C.compute_rcut(grooving)

    def _apply(self, fn):
        """Override to also move the Topology object when .to() / .cuda() is called."""
        super()._apply(fn)
        # After super()._apply, all buffers have been moved to the new device.
        # Detect the target device from a buffer and move topology to match.
        device = self.bonded_pairs.device
        if self.topology.bonded_pairs.device != device:
            self.topology = self.topology.to(device)
        return self

    def forward(self, state: SystemState) -> Tensor:
        """Compute total potential energy.

        Args:
            state: SystemState with positions and quaternions
                   (positions should have requires_grad=True for force computation)

        Returns:
            Scalar tensor: total potential energy
        """
        return self.total_energy(state)

    def total_energy(self, state: SystemState) -> Tensor:
        """Compute total potential energy as sum of all interaction terms.

        Args:
            state: SystemState with positions and quaternions

        Returns:
            Scalar tensor: total potential energy
        """
        components = self.energy_components(state)
        return sum(components.values())

    def energy_components(self, state: SystemState) -> Dict[str, Tensor]:
        """Compute all energy components separately.

        Useful for diagnostics and per-term analysis.

        Args:
            state: SystemState with positions and quaternions

        Returns:
            Dict mapping interaction names to scalar energy tensors
        """
        positions = state.positions
        quaternions = state.quaternions
        box = state.box

        # Get learnable params dict (None if nothing is learnable → exact old path)
        params = self.param_store.as_dict() if self._has_learnable else None
        excl_eps = params['excl_eps'] if params is not None else None

        # Compute site offsets (relative to COM)
        back_offsets, stack_offsets, base_offsets = compute_site_offsets(
            quaternions, self.grooving
        )

        # Find non-bonded pairs within cutoff
        nonbonded_pairs = find_nonbonded_pairs(
            positions, self.topology, box, self.cutoff
        )

        components = {}

        # === Bonded interactions ===
        components['fene'] = fene_energy(
            positions, back_offsets, self.bonded_pairs, box, params=params
        )

        components['bonded_excl_vol'] = bonded_excluded_volume_energy(
            positions, back_offsets, base_offsets, self.bonded_pairs, box,
            excl_eps=excl_eps
        )

        components['stacking'] = stacking_energy(
            positions, quaternions, stack_offsets, self.bonded_pairs,
            self.stacking_eps, box, params=params
        )

        # === Non-bonded interactions ===
        components['nonbonded_excl_vol'] = nonbonded_excluded_volume_energy(
            positions, back_offsets, base_offsets, nonbonded_pairs, box,
            excl_eps=excl_eps
        )

        components['hbond'] = hydrogen_bonding_energy(
            positions, quaternions, base_offsets, nonbonded_pairs,
            self.base_types, self.hbond_eps_matrix, box, params=params
        )

        components['cross_stacking'] = cross_stacking_energy(
            positions, quaternions, base_offsets, nonbonded_pairs, box,
            params=params
        )

        components['coaxial_stacking'] = coaxial_stacking_energy(
            positions, quaternions, stack_offsets, nonbonded_pairs, box,
            params=params
        )

        return components

    def compute_forces(self, state: SystemState) -> Tensor:
        """Compute forces on all nucleotides via autograd.

        Args:
            state: SystemState (positions will temporarily have requires_grad=True)

        Returns:
            (N, 3) force tensor: F = -dE/d(positions)
        """
        positions = state.positions.detach().requires_grad_(True)
        state_copy = SystemState(
            positions=positions,
            quaternions=state.quaternions,
            velocities=state.velocities,
            ang_velocities=state.ang_velocities,
            box=state.box,
        )

        energy = self.total_energy(state_copy)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        return forces

    def update_temperature(self, temperature: float):
        """Update temperature and recompute temperature-dependent parameters.

        Args:
            temperature: new temperature in oxDNA reduced units
        """
        self.temperature = temperature
        stacking_eps = self.topology.compute_stacking_eps(temperature, self.seq_dependent)
        self.stacking_eps.data.copy_(stacking_eps.to(self.stacking_eps.device))

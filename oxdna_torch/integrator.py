"""
Differentiable integrators for oxDNA dynamics.

Supports backpropagation through time via:
1. Standard autograd (for short trajectories)
2. Gradient checkpointing (for long trajectories, trades compute for memory)

The integrator computes forces via autograd (F = -dE/dpos) and
propagates positions/velocities using velocity-Verlet or Langevin dynamics.
Quaternion updates use first-order integration.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, List, Tuple

from .state import SystemState
from .model import OxDNAEnergy
from .quaternion import quat_normalize, quat_multiply


class LangevinIntegrator(nn.Module):
    """Langevin dynamics integrator for oxDNA.

    Implements overdamped Langevin dynamics with:
    - Forces computed via autograd from the energy model
    - Translational and rotational diffusion
    - Optional gradient checkpointing for memory-efficient backprop

    The dynamics are:
      dx/dt = F/gamma + sqrt(2*kT/gamma) * noise
      dq/dt = torque_update + rotational_noise

    Args:
        energy_model: OxDNAEnergy module for computing potential energy
        dt: integration timestep
        gamma: friction coefficient
        temperature: temperature in oxDNA reduced units
        mass: particle mass (default 1.0)
    """

    def __init__(
        self,
        energy_model: OxDNAEnergy,
        dt: float = 0.003,
        gamma: float = 1.0,
        temperature: float = 0.1,
        mass: float = 1.0,
    ):
        super().__init__()
        self.energy_model = energy_model
        self.dt = dt
        self.gamma = gamma
        self.temperature = temperature
        self.mass = mass

        # Diffusion coefficient: D = kT / (gamma * mass)
        self.D_trans = temperature / (gamma * mass)
        # Translational noise amplitude: sqrt(2 * D * dt)
        self.noise_amp = (2.0 * self.D_trans * dt) ** 0.5

        # Rotational noise (simplified: same diffusion coefficient)
        self.D_rot = temperature / (gamma * mass)
        self.rot_noise_amp = (2.0 * self.D_rot * dt) ** 0.5

    def step(
        self,
        state: SystemState,
        stochastic: bool = True,
    ) -> SystemState:
        """Perform one integration step.

        Computes forces via autograd and updates positions/quaternions.

        Args:
            state: current SystemState
            stochastic: whether to add Langevin noise (False for NVE-like)

        Returns:
            New SystemState after one timestep
        """
        positions = state.positions
        quaternions = state.quaternions

        # Enable gradients for force computation
        pos_grad = positions.detach().requires_grad_(True)
        quat_grad = quaternions.detach().requires_grad_(True)

        state_grad = SystemState(
            positions=pos_grad,
            quaternions=quat_grad,
            box=state.box,
        )

        # Compute total energy
        energy = self.energy_model(state_grad)

        # Compute forces and torques via autograd
        grads = torch.autograd.grad(
            energy, [pos_grad, quat_grad],
            create_graph=True,  # Need this for backprop through time
        )

        forces = -grads[0]      # F = -dE/dpos, shape (N, 3)
        quat_grads = -grads[1]  # "torque" in quaternion space, shape (N, 4)

        # Update positions (overdamped Langevin)
        drift = forces * self.dt / (self.gamma * self.mass)
        new_positions = positions + drift

        if stochastic:
            noise = torch.randn_like(positions) * self.noise_amp
            new_positions = new_positions + noise

        # Update quaternions
        # Simple first-order update: q_new = q + dt * quat_grad_term
        # Project quaternion gradient to the tangent space of SO(3)
        quat_drift = quat_grads * self.dt / (self.gamma * self.mass)
        new_quaternions = quaternions + quat_drift

        if stochastic:
            rot_noise = torch.randn_like(quaternions) * self.rot_noise_amp
            new_quaternions = new_quaternions + rot_noise

        # Renormalize quaternions
        new_quaternions = quat_normalize(new_quaternions)

        # Apply periodic boundary conditions
        new_box = state.box
        if new_box is not None:
            new_positions = new_positions - new_box * torch.floor(new_positions / new_box)

        return SystemState(
            positions=new_positions,
            quaternions=new_quaternions,
            box=new_box,
        )

    def rollout(
        self,
        state: SystemState,
        n_steps: int,
        stochastic: bool = True,
        checkpoint_every: int = 0,
        save_every: int = 1,
    ) -> List[SystemState]:
        """Integrate for multiple steps, returning trajectory.

        Args:
            state: initial SystemState
            n_steps: number of integration steps
            stochastic: whether to add Langevin noise
            checkpoint_every: if > 0, use gradient checkpointing every N steps
                (saves memory at cost of recomputation during backward pass)
            save_every: save state every N steps (1 = save all)

        Returns:
            List of SystemState snapshots along the trajectory
        """
        trajectory = [state]

        if checkpoint_every > 0:
            # Use gradient checkpointing for memory efficiency
            current_state = state
            for start in range(0, n_steps, checkpoint_every):
                end = min(start + checkpoint_every, n_steps)
                n_chunk = end - start
                device = current_state.positions.device

                # Checkpoint this chunk
                current_state = grad_checkpoint(
                    self._integrate_chunk,
                    current_state.positions,
                    current_state.quaternions,
                    current_state.box if current_state.box is not None else torch.zeros(3, device=device),
                    torch.tensor(n_chunk, device=device),
                    torch.tensor(stochastic, device=device),
                    use_reentrant=False,
                )

                if (start + checkpoint_every) % save_every == 0 or end == n_steps:
                    trajectory.append(current_state)
        else:
            current_state = state
            for i in range(n_steps):
                current_state = self.step(current_state, stochastic=stochastic)
                if (i + 1) % save_every == 0:
                    trajectory.append(current_state)

        return trajectory

    def _integrate_chunk(
        self,
        positions: Tensor,
        quaternions: Tensor,
        box: Tensor,
        n_steps_tensor: Tensor,
        stochastic_tensor: Tensor,
    ) -> SystemState:
        """Integrate a chunk of steps (used with gradient checkpointing).

        Args are tensors to work with torch.utils.checkpoint.
        """
        n_steps = n_steps_tensor.item()
        stochastic = bool(stochastic_tensor.item())
        box_actual = box if box.any() else None

        state = SystemState(positions=positions, quaternions=quaternions, box=box_actual)

        for _ in range(n_steps):
            state = self.step(state, stochastic=stochastic)

        return state


class VelocityVerletIntegrator(nn.Module):
    """Velocity-Verlet integrator for NVE dynamics.

    Useful for testing energy conservation and validating forces.
    Does NOT add Langevin noise.

    WARNING: This integrator handles only translational degrees of freedom
    properly with velocity-Verlet. Quaternion integration uses a simplified
    first-order method. For production dynamics, use LangevinIntegrator.

    Args:
        energy_model: OxDNAEnergy module
        dt: timestep
        mass: particle mass
    """

    def __init__(
        self,
        energy_model: OxDNAEnergy,
        dt: float = 0.001,
        mass: float = 1.0,
    ):
        super().__init__()
        self.energy_model = energy_model
        self.dt = dt
        self.mass = mass

    def step(self, state: SystemState) -> SystemState:
        """Perform one velocity-Verlet step.

        Args:
            state: current SystemState (must have velocities)

        Returns:
            New SystemState
        """
        assert state.velocities is not None, "VelocityVerlet requires velocities"

        dt = self.dt
        positions = state.positions
        velocities = state.velocities
        quaternions = state.quaternions

        # Compute forces at current position
        pos_grad = positions.detach().requires_grad_(True)
        state_tmp = SystemState(positions=pos_grad, quaternions=quaternions, box=state.box)
        energy = self.energy_model(state_tmp)
        forces = -torch.autograd.grad(energy, pos_grad, create_graph=True)[0]

        # Half-step velocity update
        velocities_half = velocities + 0.5 * dt * forces / self.mass

        # Full-step position update
        new_positions = positions + dt * velocities_half

        # Apply PBC
        if state.box is not None:
            new_positions = new_positions - state.box * torch.floor(new_positions / state.box)

        # Compute forces at new position
        new_pos_grad = new_positions.detach().requires_grad_(True)
        state_tmp2 = SystemState(
            positions=new_pos_grad, quaternions=quaternions, box=state.box
        )
        energy2 = self.energy_model(state_tmp2)
        new_forces = -torch.autograd.grad(energy2, new_pos_grad, create_graph=True)[0]

        # Full velocity update
        new_velocities = velocities_half + 0.5 * dt * new_forces / self.mass

        return SystemState(
            positions=new_positions,
            quaternions=quaternions,  # Not updated in this simple version
            velocities=new_velocities,
            ang_velocities=state.ang_velocities,
            box=state.box,
        )

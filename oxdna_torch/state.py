"""
System state representation for differentiable oxDNA.
"""

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemState:
    """Complete state of an oxDNA system.

    All tensors can have requires_grad=True for backpropagation.
    Positions and quaternions are the primary degrees of freedom;
    velocities and angular velocities are used by the integrator.

    Attributes:
        positions: (N, 3) center-of-mass positions in oxDNA units
        quaternions: (N, 4) unit quaternions [w, x, y, z] for orientations
        velocities: (N, 3) translational velocities (optional)
        ang_velocities: (N, 3) angular velocities in lab frame (optional)
        box: (3,) periodic box dimensions (None for non-periodic)
    """
    positions: Tensor
    quaternions: Tensor
    velocities: Optional[Tensor] = None
    ang_velocities: Optional[Tensor] = None
    box: Optional[Tensor] = None

    @property
    def n_nucleotides(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    def to(self, device: torch.device) -> 'SystemState':
        """Move all tensors to specified device."""
        return SystemState(
            positions=self.positions.to(device),
            quaternions=self.quaternions.to(device),
            velocities=self.velocities.to(device) if self.velocities is not None else None,
            ang_velocities=self.ang_velocities.to(device) if self.ang_velocities is not None else None,
            box=self.box.to(device) if self.box is not None else None,
        )

    def detach(self) -> 'SystemState':
        """Detach all tensors from the computation graph."""
        return SystemState(
            positions=self.positions.detach(),
            quaternions=self.quaternions.detach(),
            velocities=self.velocities.detach() if self.velocities is not None else None,
            ang_velocities=self.ang_velocities.detach() if self.ang_velocities is not None else None,
            box=self.box.detach() if self.box is not None else None,
        )

    def clone(self) -> 'SystemState':
        """Deep clone of the state."""
        return SystemState(
            positions=self.positions.clone(),
            quaternions=self.quaternions.clone(),
            velocities=self.velocities.clone() if self.velocities is not None else None,
            ang_velocities=self.ang_velocities.clone() if self.ang_velocities is not None else None,
            box=self.box.clone() if self.box is not None else None,
        )

    def requires_grad_(self, requires_grad: bool = True) -> 'SystemState':
        """Set requires_grad on positions and quaternions."""
        self.positions.requires_grad_(requires_grad)
        self.quaternions.requires_grad_(requires_grad)
        return self

"""
oxdna_torch: Differentiable oxDNA potential in PyTorch.

A PyTorch reimplementation of the oxDNA coarse-grained DNA model
that supports backpropagation through time for:
  - Parameter learning
  - Hybrid neural network + physics models
  - Inverse sequence design

Reference:
  Ouldridge et al., J. Chem. Phys. 134, 085101 (2011)
  Sulc et al., J. Chem. Phys. 137, 135101 (2012) [sequence dependence]
"""

from .state import SystemState
from .topology import Topology
from .io import load_system, read_topology, read_configuration, write_configuration
from .model import OxDNAEnergy
from .params import ParameterStore

__all__ = [
    'SystemState',
    'Topology',
    'OxDNAEnergy',
    'ParameterStore',
    'load_system',
    'read_topology',
    'read_configuration',
    'write_configuration',
]

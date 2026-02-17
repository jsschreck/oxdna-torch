# oxdna_torch

`oxdna_torch` is a high-performance, fully differentiable PyTorch reimplementation of the **oxDNA coarse-grained model**.

By leveraging PyTorch's autograd engine, this package allows for the calculation of forces and torques via automatic differentiation, enabling backpropagation through physics-based trajectories. This makes it a powerful tool for:

* **Machine Learning Integration**: Combining physical potentials with neural networks.
* **Parameter Optimization**: Learning or refining oxDNA interaction parameters directly from experimental or simulation data.
* **Inverse Design**: Performing sequence design by backpropagating through structural properties.

## Features

* **Complete oxDNA1 Potential**: Includes all 7 interaction terms: FENE, Stacking, Hydrogen Bonding, Cross Stacking, Coaxial Stacking, and Excluded Volume (Bonded and Non-bonded).
* **Fully Differentiable**: Every potential term is implemented using pure PyTorch operations, ensuring gradients are preserved throughout the simulation.
* **GPU Acceleration**: Native support for CUDA, allowing for massive parallelization of energy and force calculations.
* **Differentiable Integrators**: Includes Langevin and Velocity-Verlet integrators that support backpropagation through time (BPTT).
* **Memory Efficiency**: Optional gradient checkpointing for long trajectories to trade computation for reduced memory usage.
* **Parameter Store**: A centralized registry to manage learnable parameters versus frozen constants.

## Installation

```bash
git clone https://github.com/jsschreck/oxdna_torch.git
cd oxdna_torch
pip install -e .

```

## Quick Start

The following example demonstrates how to load a system, compute energies and forces, and run a short trajectory on a GPU.

```python
import torch
from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.integrator import LangevinIntegrator

# 1. Load system topology and configuration
# You can load directly to a specific device (e.g., 'cuda' or 'cpu')
topology, state = load_system("topology.top", "config.conf", device='cuda')

# 2. Initialize the Energy Model
# Temperature is provided in oxDNA reduced units (T_kelvin / 3000)
model = OxDNAEnergy(
    topology, 
    temperature=0.1113, # ~334K
    seq_dependent=True
).to('cuda')

# 3. Compute Energy and Forces
# Positions and quaternions in 'state' can track gradients
energy = model(state)
forces = model.compute_forces(state)

print(f"Total Potential Energy: {energy.item()}")

# 4. Run Dynamics
integrator = LangevinIntegrator(
    model, 
    dt=0.003, 
    gamma=1.0, 
    temperature=0.1113
)

# Perform a 100-step simulation
trajectory = integrator.rollout(state, n_steps=100)

```

## Repository Structure

* `oxdna_torch/model.py`: The top-level `OxDNAEnergy` module that assembles all interaction terms.
* `oxdna_torch/interactions/`: Individual modules for each of the 7 oxDNA potential terms.
* `oxdna_torch/integrator.py`: Differentiable Langevin and Velocity-Verlet dynamics.
* `oxdna_torch/params.py`: Management of learnable vs. fixed physical parameters.
* `oxdna_torch/io.py`: Support for standard oxDNA `.top` and `.conf` file formats.
* `oxdna_torch/quaternion.py`: Quaternion utilities for rigid body rotations of nucleotides.

## References

* Ouldridge et al., *J. Chem. Phys.* 134, 085101 (2011).
* Sulc et al., *J. Chem. Phys.* 137, 135101 (2012) (Sequence dependence).
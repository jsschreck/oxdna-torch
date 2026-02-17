"""
oxdna_torch demo: Differentiable oxDNA potential in PyTorch.

This script demonstrates the core capabilities using the HAIRPIN example
that ships with oxDNA. Run from the oxDNA root directory:

    python examples/pytorch_demo.py

It shows:
  1. Loading an oxDNA system (topology + configuration)
  2. Computing energy and its per-term breakdown
  3. Computing forces via autograd
  4. Running a short Langevin dynamics simulation
  5. Backpropagating through time (the key differentiable physics feature)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.integrator import LangevinIntegrator


def main():
    # ----------------------------------------------------------------
    # 1. Load system
    # ----------------------------------------------------------------
    print("=" * 60)
    print("1. LOADING SYSTEM")
    print("=" * 60)

    topology, state = load_system(
        "examples/HAIRPIN/initial.top",
        "examples/HAIRPIN/initial.conf",
    )

    print(f"   Nucleotides : {topology.n_nucleotides}")
    print(f"   Strands     : {topology.n_strands}")
    print(f"   Bonded pairs: {topology.n_bonded}")
    print(f"   Box size    : {state.box.tolist()}")

    # Print the sequence
    base_map = {0: "A", 1: "C", 2: "G", 3: "T"}
    seq = "".join(base_map[b.item()] for b in topology.base_types)
    print(f"   Sequence    : {seq}")
    print()

    # ----------------------------------------------------------------
    # 2. Create the energy model
    # ----------------------------------------------------------------
    print("=" * 60)
    print("2. CREATING ENERGY MODEL")
    print("=" * 60)

    # Temperature: 334 K in oxDNA reduced units (T_reduced = T_kelvin / 3000)
    temperature = 334.0 / 3000.0
    print(f"   Temperature : {temperature:.6f} (= 334 K)")

    model = OxDNAEnergy(
        topology,
        temperature=temperature,
        seq_dependent=False,   # use average-sequence model (matches reference)
        grooving=False,        # no major-minor groove distinction
    )
    print(f"   Cutoff dist : {model.cutoff:.4f}")
    print()

    # ----------------------------------------------------------------
    # 3. Compute energy breakdown
    # ----------------------------------------------------------------
    print("=" * 60)
    print("3. ENERGY BREAKDOWN")
    print("=" * 60)

    components = model.energy_components(
        SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
    )

    for name, val in components.items():
        print(f"   {name:25s} : {val.item():+.6f}")

    total = sum(components.values())
    print(f"   {'TOTAL':25s} : {total.item():+.6f}")
    print(f"   {'per nucleotide':25s} : {total.item() / topology.n_nucleotides:+.6f}")
    print(f"   {'reference (from file)':25s} : -0.365026")
    print()

    # ----------------------------------------------------------------
    # 4. Compute forces via autograd
    # ----------------------------------------------------------------
    print("=" * 60)
    print("4. FORCES VIA AUTOGRAD")
    print("=" * 60)

    # Enable gradient tracking on positions
    pos = state.positions.clone().detach().requires_grad_(True)
    quat = state.quaternions.clone().detach()

    energy = model(SystemState(positions=pos, quaternions=quat, box=state.box))
    energy.backward()

    forces = -pos.grad  # F = -dE/dr

    print(f"   Total energy       : {energy.item():+.6f}")
    print(f"   Force on nucl. 0   : [{forces[0, 0].item():+.4f}, "
          f"{forces[0, 1].item():+.4f}, {forces[0, 2].item():+.4f}]")
    print(f"   Max force magnitude: {forces.norm(dim=-1).max().item():.4f}")
    print(f"   Any NaN in forces  : {torch.isnan(forces).any().item()}")
    print()

    # ----------------------------------------------------------------
    # 5. Short Langevin dynamics
    # ----------------------------------------------------------------
    print("=" * 60)
    print("5. LANGEVIN DYNAMICS (50 steps)")
    print("=" * 60)

    integrator = LangevinIntegrator(
        energy_model=model,
        dt=0.001,
        gamma=1.0,
        temperature=temperature,
    )

    current = SystemState(
        positions=state.positions.clone().detach(),
        quaternions=state.quaternions.clone().detach(),
        box=state.box,
    )

    energies = []
    for step in range(50):
        e = model(SystemState(
            positions=current.positions.detach(),
            quaternions=current.quaternions.detach(),
            box=current.box,
        ))
        energies.append(e.item())
        current = integrator.step(current, stochastic=True)

    print(f"   Initial energy: {energies[0]:+.4f}")
    print(f"   Final energy  : {energies[-1]:+.4f}")
    print(f"   Min energy    : {min(energies):+.4f}")
    print(f"   Max energy    : {max(energies):+.4f}")
    print()

    # ----------------------------------------------------------------
    # 6. Backprop through time
    # ----------------------------------------------------------------
    print("=" * 60)
    print("6. BACKPROP THROUGH TIME (5 steps)")
    print("=" * 60)
    print("   This is the key feature: gradients flow from the final")
    print("   energy back through the integration steps to the initial")
    print("   positions. This enables learning force field parameters,")
    print("   training hybrid NN+physics models, and inverse design.")
    print()

    # Start with initial positions that require gradients
    init_pos = state.positions.clone().detach().requires_grad_(True)
    init_quat = state.quaternions.clone().detach()

    s = SystemState(positions=init_pos, quaternions=init_quat, box=state.box)

    # Integrate forward for 5 steps (deterministic, no noise)
    n_bptt_steps = 5
    for step in range(n_bptt_steps):
        s = integrator.step(s, stochastic=False)

    # Compute energy at the final state
    final_energy = model(s)

    # Backpropagate through the entire trajectory
    final_energy.backward()

    print(f"   Steps integrated  : {n_bptt_steps}")
    print(f"   Final energy      : {final_energy.item():+.6f}")
    print(f"   Grad wrt init pos : exists={init_pos.grad is not None}")
    print(f"   Grad norm         : {init_pos.grad.norm().item():.4f}")
    print(f"   Any NaN in grads  : {torch.isnan(init_pos.grad).any().item()}")
    print()

    # ----------------------------------------------------------------
    # 7. Verify forces with finite differences
    # ----------------------------------------------------------------
    print("=" * 60)
    print("7. GRADIENT VERIFICATION (finite differences)")
    print("=" * 60)

    eps_fd = 1e-6
    nucl_idx, dim_idx = 3, 1  # Check nucleotide 3, y-component

    p_plus = state.positions.clone().detach()
    p_plus[nucl_idx, dim_idx] += eps_fd
    e_plus = model(SystemState(positions=p_plus, quaternions=state.quaternions.detach(), box=state.box)).item()

    p_minus = state.positions.clone().detach()
    p_minus[nucl_idx, dim_idx] -= eps_fd
    e_minus = model(SystemState(positions=p_minus, quaternions=state.quaternions.detach(), box=state.box)).item()

    fd_force = -(e_plus - e_minus) / (2 * eps_fd)

    # Autograd force (recompute)
    p_check = state.positions.clone().detach().requires_grad_(True)
    e_check = model(SystemState(positions=p_check, quaternions=state.quaternions.detach(), box=state.box))
    e_check.backward()
    ag_force = -p_check.grad[nucl_idx, dim_idx].item()

    print(f"   Nucleotide {nucl_idx}, dim {dim_idx} (y):")
    print(f"   Finite diff force : {fd_force:+.8f}")
    print(f"   Autograd force    : {ag_force:+.8f}")
    print(f"   Relative error    : {abs(fd_force - ag_force) / (abs(ag_force) + 1e-10):.2e}")
    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

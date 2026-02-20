"""
I/O for oxDNA topology and configuration files.

Supports the "old" oxDNA file formats:

Topology file (*.top):
  Line 1: N_nucleotides N_strands
  Lines 2..N+1: strand_id base_type n3_neighbor n5_neighbor
    - strand_id: 1-indexed
    - base_type: A, C, G, T
    - n3_neighbor: 0-indexed nucleotide index, -1 for no neighbor
    - n5_neighbor: 0-indexed nucleotide index, -1 for no neighbor

Configuration file (*.conf / *.dat):
  Line 1: t = <timestep>
  Line 2: b = <Lx> <Ly> <Lz>
  Line 3: E = <Epot> <Ekin> <Etot>
  Lines 4..N+3: <px> <py> <pz> <a1x> <a1y> <a1z> <a3x> <a3y> <a3z> <vx> <vy> <vz> <Lx> <Ly> <Lz>
    - p: center of mass position
    - a1: principal axis (along backbone direction)
    - a3: base normal direction
    - v: velocity (often zero in initial configs)
    - L: angular momentum (often zero in initial configs)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple

from .state import SystemState
from .topology import Topology
from .quaternion import rotmat_to_quat
from . import constants as C


def read_topology(filepath: Union[str, Path]) -> Topology:
    """Read an oxDNA topology file.

    Args:
        filepath: path to .top file

    Returns:
        Topology object with connectivity and base type info
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line: N_nucleotides N_strands
    header = lines[0].strip().split()
    n_nucleotides = int(header[0])
    n_strands = int(header[1])

    strand_ids = torch.zeros(n_nucleotides, dtype=torch.long)
    base_types = torch.zeros(n_nucleotides, dtype=torch.long)
    bonded_neighbors = torch.full((n_nucleotides, 2), -1, dtype=torch.long)  # [n3, n5]

    for i in range(n_nucleotides):
        parts = lines[i + 1].strip().split()
        strand_id = int(parts[0]) - 1  # Convert to 0-indexed
        base_char = parts[1]
        n3 = int(parts[2])
        n5 = int(parts[3])

        strand_ids[i] = strand_id
        base_types[i] = C.BASE_CHAR_TO_INT[base_char]
        bonded_neighbors[i, 0] = n3  # n3 neighbor index (already 0-indexed, -1 for none)
        bonded_neighbors[i, 1] = n5  # n5 neighbor index

    return Topology(
        n_nucleotides=n_nucleotides,
        n_strands=n_strands,
        strand_ids=strand_ids,
        base_types=base_types,
        bonded_neighbors=bonded_neighbors,
    )


def read_configuration(
    filepath: Union[str, Path],
    topology: Optional[Topology] = None,
) -> SystemState:
    """Read an oxDNA configuration file.

    The configuration stores positions and orientations as a1, a3 vectors.
    We reconstruct the full rotation matrix R = [a1, a2, a3] where a2 = a3 x a1,
    then convert to quaternions.

    Args:
        filepath: path to .conf or .dat file
        topology: optional Topology (used to verify nucleotide count)

    Returns:
        SystemState with positions, quaternions, velocities, and box
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Line 1: timestep
    # t = <value>
    # Line 2: box
    # b = <Lx> <Ly> <Lz>
    box_parts = lines[1].strip().split()
    box = torch.tensor([float(box_parts[2]), float(box_parts[3]), float(box_parts[4])],
                        dtype=torch.float64)

    # Line 3: energy (informational, not needed for state)
    # Lines 4+: nucleotide data
    n_nucleotides = len(lines) - 3

    if topology is not None:
        assert n_nucleotides == topology.n_nucleotides, \
            f"Configuration has {n_nucleotides} nucleotides but topology has {topology.n_nucleotides}"

    positions = torch.zeros(n_nucleotides, 3, dtype=torch.float64)
    a1_vecs = torch.zeros(n_nucleotides, 3, dtype=torch.float64)
    a3_vecs = torch.zeros(n_nucleotides, 3, dtype=torch.float64)
    velocities = torch.zeros(n_nucleotides, 3, dtype=torch.float64)
    ang_velocities = torch.zeros(n_nucleotides, 3, dtype=torch.float64)

    for i in range(n_nucleotides):
        parts = lines[i + 3].strip().split()
        values = [float(x) for x in parts]

        positions[i] = torch.tensor(values[0:3])
        a1_vecs[i] = torch.tensor(values[3:6])
        a3_vecs[i] = torch.tensor(values[6:9])
        velocities[i] = torch.tensor(values[9:12])
        ang_velocities[i] = torch.tensor(values[12:15])

    # Build rotation matrices: R = [a1 | a2 | a3] as columns
    # a2 = a3 x a1
    a2_vecs = torch.linalg.cross(a3_vecs, a1_vecs, dim=-1)

    # Normalize to ensure proper rotation matrices
    a1_vecs = a1_vecs / torch.norm(a1_vecs, dim=-1, keepdim=True)
    a2_vecs = a2_vecs / torch.norm(a2_vecs, dim=-1, keepdim=True)
    a3_vecs = a3_vecs / torch.norm(a3_vecs, dim=-1, keepdim=True)

    # Rotation matrix: columns are a1, a2, a3
    # R[i] @ [1,0,0] = a1[i], R[i] @ [0,1,0] = a2[i], R[i] @ [0,0,1] = a3[i]
    rot_matrices = torch.stack([a1_vecs, a2_vecs, a3_vecs], dim=-1)  # (N, 3, 3)

    # Convert to quaternions
    quaternions = rotmat_to_quat(rot_matrices)

    return SystemState(
        positions=positions,
        quaternions=quaternions,
        velocities=velocities,
        ang_velocities=ang_velocities,
        box=box,
    )


def write_configuration(
    filepath: Union[str, Path],
    state: SystemState,
    topology: Optional[Topology] = None,
    timestep: int = 0,
    Epot: float = 0.0,
    Ekin: float = 0.0,
    append: bool = False,
) -> None:
    """Write an oxDNA configuration file.

    The format is identical to the reference oxDNA binary output and can be
    read back by both this library and the C++ oxDNA tools.

    Args:
        filepath: output path
        state: SystemState with positions and quaternions
        topology: optional Topology (used only for nucleotide count validation)
        timestep: simulation timestep written to the header
        Epot: potential energy written to the ``E =`` header line
        Ekin: kinetic energy written to the ``E =`` header line
        append: if True, append to an existing file (for trajectory files);
                if False (default), overwrite / create the file
    """
    from .quaternion import quat_to_rotmat

    filepath = Path(filepath)

    R = quat_to_rotmat(state.quaternions)  # (N, 3, 3)

    # Extract orientation vectors (columns of R)
    a1 = R[:, :, 0]  # (N, 3)
    a3 = R[:, :, 2]  # (N, 3)

    pos = state.positions.detach().cpu().numpy()
    a1_np = a1.detach().cpu().numpy()
    a3_np = a3.detach().cpu().numpy()

    vel = state.velocities.detach().cpu().numpy() if state.velocities is not None \
        else np.zeros_like(pos)
    ang_vel = state.ang_velocities.detach().cpu().numpy() if state.ang_velocities is not None \
        else np.zeros_like(pos)

    box = state.box.detach().cpu().numpy() if state.box is not None \
        else np.array([50.0, 50.0, 50.0])

    Etot = Epot + Ekin
    mode = 'a' if append else 'w'
    with open(filepath, mode) as f:
        f.write(f"t = {timestep}\n")
        f.write(f"b = {box[0]} {box[1]} {box[2]}\n")
        f.write(f"E = {Epot} {Ekin} {Etot}\n")

        for i in range(state.n_nucleotides):
            parts = []
            parts.extend(f"{x:.15g}" for x in pos[i])
            parts.extend(f"{x:.15g}" for x in a1_np[i])
            parts.extend(f"{x:.15g}" for x in a3_np[i])
            parts.extend(f"{x:.15g}" for x in vel[i])
            parts.extend(f"{x:.15g}" for x in ang_vel[i])
            f.write(" ".join(parts) + "\n")


def load_system(
    topology_path: Union[str, Path],
    config_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[Topology, SystemState]:
    """Convenience function to load both topology and configuration.

    Args:
        topology_path: path to .top file
        config_path: path to .conf or .dat file
        device: optional device to move tensors to (e.g., 'cuda')

    Returns:
        (topology, state) tuple
    """
    topology = read_topology(topology_path)
    state = read_configuration(config_path, topology)
    if device is not None:
        topology = topology.to(device)
        state = state.to(device)
    return topology, state

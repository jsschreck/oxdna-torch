"""
Quaternion utilities for rigid body orientation.

Convention: q = [w, x, y, z] where w is the scalar part.
All functions operate on batched quaternions of shape (..., 4).
"""

import torch
from torch import Tensor


def quat_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit length. q: (..., 4)."""
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)


def quat_conjugate(q: Tensor) -> Tensor:
    """Quaternion conjugate (inverse for unit quaternions). q: (..., 4)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions. q1, q2: (..., 4).

    Returns q1 * q2 using the Hamilton product convention.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert unit quaternion to 3x3 rotation matrix.

    q: (..., 4) -> R: (..., 3, 3)

    The rotation matrix R rotates a vector v as: v_rotated = R @ v
    This is equivalent to the quaternion rotation: q * [0,v] * q_conj

    In oxDNA, the orientation matrix columns are [a1, a2, a3] where:
      a1 = R[:, 0]  (column 0, the principal axis)
      a2 = R[:, 1]  (column 1)
      a3 = R[:, 2]  (column 2)
    """
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)

    # Precompute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Rotation matrix rows
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return R


def quat_rotate_vec(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v by quaternion q.

    q: (..., 4), v: (..., 3) -> (..., 3)

    Uses the formula: v' = q * [0,v] * q_conj
    which is equivalent to: v' = R(q) @ v
    but avoids building the full rotation matrix.
    """
    # Faster than building rotation matrix for single vector rotation
    q_vec = q[..., 1:4]  # (..., 3)
    q_w = q[..., 0:1]    # (..., 1)

    # t = 2 * cross(q_vec, v)
    t = 2.0 * torch.linalg.cross(q_vec, v, dim=-1)

    # v' = v + w*t + cross(q_vec, t)
    return v + q_w * t + torch.linalg.cross(q_vec, t, dim=-1)


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert 3x3 rotation matrix to unit quaternion.

    R: (..., 3, 3) -> q: (..., 4)

    Uses Shepperd's method for numerical stability.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (B,)

    # Four possible cases for numerical stability
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4*w
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    # Ensure positive w convention
    sign = torch.sign(q[:, 0]).unsqueeze(-1)
    sign[sign == 0] = 1.0
    q = q * sign

    q = quat_normalize(q)
    return q.reshape(*batch_shape, 4)


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """Create quaternion from rotation axis and angle.

    axis: (..., 3) unit vector, angle: (...) in radians -> q: (..., 4)
    """
    half_angle = angle.unsqueeze(-1) * 0.5
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    return torch.cat([w, xyz], dim=-1)


def quat_angular_velocity_update(q: Tensor, omega: Tensor, dt: float) -> Tensor:
    """Update quaternion given angular velocity.

    q: (..., 4), omega: (..., 3) angular velocity, dt: timestep
    -> q_new: (..., 4) updated quaternion

    Uses first-order integration: q(t+dt) = q(t) + 0.5*dt * [0, omega] * q(t)
    Then renormalize.
    """
    omega_quat = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
    q_dot = 0.5 * quat_multiply(omega_quat, q)
    q_new = q + dt * q_dot
    return quat_normalize(q_new)

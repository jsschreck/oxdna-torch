"""
Tests for oxdna_torch.quaternion module.
"""

import torch
import math
import pytest

from oxdna_torch.quaternion import (
    quat_normalize,
    quat_conjugate,
    quat_multiply,
    quat_to_rotmat,
    quat_rotate_vec,
    rotmat_to_quat,
    quat_from_axis_angle,
    quat_angular_velocity_update,
)


class TestQuatNormalize:
    def test_already_normalized(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        qn = quat_normalize(q)
        assert torch.allclose(q, qn)

    def test_unnormalized(self):
        q = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        qn = quat_normalize(q)
        assert torch.allclose(qn, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))

    def test_batch(self):
        q = torch.randn(10, 4, dtype=torch.float64)
        qn = quat_normalize(q)
        norms = torch.norm(qn, dim=-1)
        assert torch.allclose(norms, torch.ones(10, dtype=torch.float64), atol=1e-12)


class TestQuatConjugate:
    def test_identity(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        qc = quat_conjugate(q)
        assert torch.allclose(q, qc)

    def test_general(self):
        q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        qc = quat_conjugate(q)
        expected = torch.tensor([0.5, -0.5, -0.5, -0.5], dtype=torch.float64)
        assert torch.allclose(qc, expected)

    def test_q_times_conj_is_identity(self):
        q = quat_normalize(torch.randn(4, dtype=torch.float64))
        qc = quat_conjugate(q)
        product = quat_multiply(q, qc)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(product, identity, atol=1e-12)


class TestQuatMultiply:
    def test_identity_left(self):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q = quat_normalize(torch.randn(4, dtype=torch.float64))
        result = quat_multiply(identity, q)
        assert torch.allclose(result, q, atol=1e-12)

    def test_identity_right(self):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q = quat_normalize(torch.randn(4, dtype=torch.float64))
        result = quat_multiply(q, identity)
        assert torch.allclose(result, q, atol=1e-12)

    def test_i_times_j_equals_k(self):
        # i*j = k in quaternion algebra
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        result = quat_multiply(i, j)
        assert torch.allclose(result, k, atol=1e-12)

    def test_j_times_i_equals_neg_k(self):
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        neg_k = torch.tensor([0.0, 0.0, 0.0, -1.0], dtype=torch.float64)
        result = quat_multiply(j, i)
        assert torch.allclose(result, neg_k, atol=1e-12)

    def test_batch(self):
        q1 = quat_normalize(torch.randn(5, 4, dtype=torch.float64))
        q2 = quat_normalize(torch.randn(5, 4, dtype=torch.float64))
        result = quat_multiply(q1, q2)
        # Result should also be unit quaternions
        norms = torch.norm(result, dim=-1)
        assert torch.allclose(norms, torch.ones(5, dtype=torch.float64), atol=1e-12)


class TestQuatToRotmat:
    def test_identity_quaternion(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        R = quat_to_rotmat(q.unsqueeze(0))[0]
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-12)

    def test_90deg_around_z(self):
        # 90 degrees around z: q = [cos(45), 0, 0, sin(45)]
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=torch.float64)
        R = quat_to_rotmat(q.unsqueeze(0))[0]
        # Should rotate x -> y, y -> -x
        x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        rotated = R @ x
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(rotated, expected, atol=1e-12)

    def test_180deg_around_x(self):
        angle = math.pi
        q = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0], dtype=torch.float64)
        R = quat_to_rotmat(q.unsqueeze(0))[0]
        # Should flip y and z
        y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        rotated = R @ y
        expected = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(rotated, expected, atol=1e-12)

    def test_rotation_matrix_is_orthogonal(self):
        q = quat_normalize(torch.randn(10, 4, dtype=torch.float64))
        R = quat_to_rotmat(q)
        # R^T R should be identity
        RtR = torch.bmm(R.transpose(-1, -2), R)
        I = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        assert torch.allclose(RtR, I, atol=1e-12)

    def test_determinant_is_one(self):
        q = quat_normalize(torch.randn(10, 4, dtype=torch.float64))
        R = quat_to_rotmat(q)
        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(10, dtype=torch.float64), atol=1e-12)

    def test_columns_are_orthonormal(self):
        q = quat_normalize(torch.randn(5, 4, dtype=torch.float64))
        R = quat_to_rotmat(q)
        a1, a2, a3 = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        # Norms
        assert torch.allclose(a1.norm(dim=-1), torch.ones(5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(a2.norm(dim=-1), torch.ones(5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(a3.norm(dim=-1), torch.ones(5, dtype=torch.float64), atol=1e-12)
        # Orthogonality
        assert torch.allclose((a1 * a2).sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose((a1 * a3).sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose((a2 * a3).sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)


class TestQuatRotateVec:
    def test_identity_rotation(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        result = quat_rotate_vec(q, v)
        assert torch.allclose(result, v, atol=1e-12)

    def test_matches_rotmat(self):
        """quat_rotate_vec should give same result as R @ v."""
        q = quat_normalize(torch.randn(10, 4, dtype=torch.float64))
        v = torch.randn(10, 3, dtype=torch.float64)
        # Via quaternion rotation
        result_quat = quat_rotate_vec(q, v)
        # Via rotation matrix
        R = quat_to_rotmat(q)
        result_mat = torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result_quat, result_mat, atol=1e-12)

    def test_90deg_z_rotation(self):
        angle = math.pi / 2
        q = torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]], dtype=torch.float64)
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        result = quat_rotate_vec(q, v)
        expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12)


class TestRotmatToQuat:
    def test_identity(self):
        R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        q = rotmat_to_quat(R)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(q, expected, atol=1e-12)

    def test_roundtrip_quat_to_rotmat_to_quat(self):
        q_orig = quat_normalize(torch.randn(20, 4, dtype=torch.float64))
        # Ensure positive w convention
        signs = torch.sign(q_orig[:, 0]).unsqueeze(-1)
        signs[signs == 0] = 1.0
        q_orig = q_orig * signs

        R = quat_to_rotmat(q_orig)
        q_back = rotmat_to_quat(R)

        # Quaternions should match (up to sign — both q and -q give same rotation)
        # We enforce positive w, so they should be identical
        assert torch.allclose(q_orig, q_back, atol=1e-10)

    def test_roundtrip_rotmat_to_quat_to_rotmat(self):
        q = quat_normalize(torch.randn(20, 4, dtype=torch.float64))
        R_orig = quat_to_rotmat(q)
        q_back = rotmat_to_quat(R_orig)
        R_back = quat_to_rotmat(q_back)
        assert torch.allclose(R_orig, R_back, atol=1e-10)


class TestQuatFromAxisAngle:
    def test_zero_angle(self):
        axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        angle = torch.tensor([0.0], dtype=torch.float64)
        q = quat_from_axis_angle(axis, angle)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(q, expected, atol=1e-12)

    def test_180_around_z(self):
        axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi], dtype=torch.float64)
        q = quat_from_axis_angle(axis, angle)
        # q = [cos(pi/2), 0, 0, sin(pi/2)] = [0, 0, 0, 1]
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        assert torch.allclose(q, expected, atol=1e-12)

    def test_rotation_matches_rotmat(self):
        axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi / 3], dtype=torch.float64)
        q = quat_from_axis_angle(axis, angle)
        R = quat_to_rotmat(q)
        # Rotate y by 60 deg around x
        v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        rotated = (R[0] @ v)
        expected = torch.tensor([0.0, math.cos(math.pi / 3), math.sin(math.pi / 3)], dtype=torch.float64)
        assert torch.allclose(rotated, expected, atol=1e-12)


class TestQuatDifferentiability:
    def test_rotmat_grad_flows(self):
        q = quat_normalize(torch.randn(4, dtype=torch.float64)).requires_grad_(True)
        R = quat_to_rotmat(q.unsqueeze(0))
        loss = R.sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_rotate_vec_grad_flows(self):
        q = quat_normalize(torch.randn(1, 4, dtype=torch.float64)).requires_grad_(True)
        v = torch.randn(1, 3, dtype=torch.float64)
        result = quat_rotate_vec(q, v)
        loss = result.sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

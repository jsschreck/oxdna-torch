"""
Tests for oxdna_torch.smooth module (f1, f2, f4, f5 potential functions).
"""

import torch
import math
import pytest

from oxdna_torch.smooth import f1, f2, f4, f4_of_cos, f4_of_cos_cxst_t1, f5, repulsive_lj
from oxdna_torch import constants as C


class TestF1:
    """Tests for the Morse-like radial potential f1."""

    def test_zero_outside_range(self):
        """f1 should be zero outside [RCLOW, RCHIGH]."""
        eps = torch.tensor([1.0], dtype=torch.float64)
        # Below RCLOW
        r = torch.tensor([C.HYDR_RCLOW - 0.01], dtype=torch.float64)
        assert f1(r, C.HYDR_F1, eps).item() == 0.0
        # Above RCHIGH
        r = torch.tensor([C.HYDR_RCHIGH + 0.01], dtype=torch.float64)
        assert f1(r, C.HYDR_F1, eps).item() == 0.0

    def test_minimum_near_r0(self):
        """f1 should have its minimum near R0."""
        eps = torch.tensor(1.0, dtype=torch.float64).expand(100)
        r = torch.linspace(C.HYDR_RCLOW + 0.01, C.HYDR_RCHIGH - 0.01, 100, dtype=torch.float64)
        vals = f1(r, C.HYDR_F1, eps)
        min_idx = vals.argmin()
        min_r = r[min_idx].item()
        # Minimum should be near R0 = 0.4
        assert abs(min_r - C.HYDR_R0) < 0.05, f"Minimum at r={min_r}, expected near {C.HYDR_R0}"

    def test_negative_at_minimum(self):
        """f1 should be negative at its minimum (attractive well)."""
        eps = torch.tensor([1.0], dtype=torch.float64)
        r = torch.tensor([C.HYDR_R0], dtype=torch.float64)
        val = f1(r, C.HYDR_F1, eps)
        assert val.item() < 0, f"f1 at R0 should be negative, got {val.item()}"

    def test_continuity_at_boundaries(self):
        """f1 should be continuous at RLOW, RHIGH boundaries."""
        eps = torch.tensor([1.0], dtype=torch.float64)
        delta = 1e-8

        # At RLOW boundary (HYDR)
        r_below = torch.tensor([C.HYDR_RLOW - delta], dtype=torch.float64)
        r_above = torch.tensor([C.HYDR_RLOW + delta], dtype=torch.float64)
        val_below = f1(r_below, C.HYDR_F1, eps)
        val_above = f1(r_above, C.HYDR_F1, eps)
        assert abs(val_below.item() - val_above.item()) < 1e-4

        # At RHIGH boundary (HYDR)
        r_below = torch.tensor([C.HYDR_RHIGH - delta], dtype=torch.float64)
        r_above = torch.tensor([C.HYDR_RHIGH + delta], dtype=torch.float64)
        val_below = f1(r_below, C.HYDR_F1, eps)
        val_above = f1(r_above, C.HYDR_F1, eps)
        assert abs(val_below.item() - val_above.item()) < 1e-4

    def test_stacking_type(self):
        """f1 with stacking parameters should also work."""
        eps = torch.tensor([1.5], dtype=torch.float64)
        r = torch.tensor([C.STCK_R0], dtype=torch.float64)
        val = f1(r, C.STCK_F1, eps)
        assert val.item() < 0

    def test_differentiable(self):
        """f1 should be differentiable everywhere in its range."""
        eps = torch.tensor(1.0, dtype=torch.float64).expand(50)
        r = torch.linspace(C.HYDR_RCLOW + 0.01, C.HYDR_RCHIGH - 0.01, 50,
                           dtype=torch.float64).requires_grad_(True)
        val = f1(r, C.HYDR_F1, eps).sum()
        val.backward()
        assert r.grad is not None
        assert not torch.isnan(r.grad).any()

    def test_eps_scaling(self):
        """Doubling eps should double the energy."""
        r = torch.tensor([0.5], dtype=torch.float64)
        eps1 = torch.tensor([1.0], dtype=torch.float64)
        eps2 = torch.tensor([2.0], dtype=torch.float64)
        val1 = f1(r, C.HYDR_F1, eps1)
        val2 = f1(r, C.HYDR_F1, eps2)
        # f1 = eps * morse - shift; shift also scales with eps
        # So it should roughly scale linearly
        ratio = val2.item() / val1.item()
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"


class TestF2:
    """Tests for the harmonic radial potential f2."""

    def test_zero_outside_range(self):
        """f2 should be zero outside [RCLOW, RCHIGH]."""
        r = torch.tensor([C.CRST_RCLOW - 0.01], dtype=torch.float64)
        assert f2(r, C.CRST_F2).item() == 0.0
        r = torch.tensor([C.CRST_RCHIGH + 0.01], dtype=torch.float64)
        assert f2(r, C.CRST_F2).item() == 0.0

    def test_minimum_at_r0(self):
        """f2 should have minimum near R0."""
        r = torch.linspace(C.CRST_RCLOW + 0.01, C.CRST_RCHIGH - 0.01, 200, dtype=torch.float64)
        vals = f2(r, C.CRST_F2)
        min_idx = vals.argmin()
        min_r = r[min_idx].item()
        assert abs(min_r - C.CRST_R0) < 0.02, f"Minimum at r={min_r}, expected near {C.CRST_R0}"

    def test_negative_at_minimum(self):
        """f2 should be negative at its minimum."""
        r = torch.tensor([C.CRST_R0], dtype=torch.float64)
        val = f2(r, C.CRST_F2)
        assert val.item() < 0, f"f2 at R0 should be negative, got {val.item()}"

    def test_differentiable(self):
        r = torch.linspace(C.CRST_RCLOW + 0.01, C.CRST_RCHIGH - 0.01, 50,
                           dtype=torch.float64).requires_grad_(True)
        val = f2(r, C.CRST_F2).sum()
        val.backward()
        assert not torch.isnan(r.grad).any()

    def test_cxst_type(self):
        """f2 with coaxial stacking parameters should work."""
        r = torch.tensor([C.CXST_R0], dtype=torch.float64)
        val = f2(r, C.CXST_F2)
        assert val.item() < 0


class TestF4:
    """Tests for the angular modulation function f4."""

    def test_maximum_at_t0(self):
        """f4 should be 1.0 at theta = T0."""
        for i in range(13):
            t0 = C.F4_THETA_T0[i].item()
            theta = torch.tensor([t0], dtype=torch.float64)
            val = f4(theta, i)
            assert abs(val.item() - 1.0) < 1e-10, f"f4 type {i} at T0={t0}: expected 1.0, got {val.item()}"

    def test_zero_outside_tc(self):
        """f4 should be zero when |theta - T0| > TC."""
        for i in range(13):
            t0 = C.F4_THETA_T0[i].item()
            tc = C.F4_THETA_TC[i].item()
            theta = torch.tensor([t0 + tc + 0.01], dtype=torch.float64)
            val = f4(theta, i)
            assert val.item() == 0.0, f"f4 type {i} outside TC: expected 0, got {val.item()}"

    def test_symmetric_around_t0(self):
        """f4 should be symmetric around T0."""
        for i in range(13):
            t0 = C.F4_THETA_T0[i].item()
            tc = C.F4_THETA_TC[i].item()
            offset = tc * 0.5
            theta_plus = torch.tensor([t0 + offset], dtype=torch.float64)
            theta_minus = torch.tensor([t0 - offset], dtype=torch.float64)
            val_plus = f4(theta_plus, i)
            val_minus = f4(theta_minus, i)
            assert abs(val_plus.item() - val_minus.item()) < 1e-10, \
                f"f4 type {i} not symmetric: f4(T0+d)={val_plus.item()}, f4(T0-d)={val_minus.item()}"

    def test_values_between_zero_and_one(self):
        """f4 should be in [0, 1] for all inputs."""
        theta = torch.linspace(0, math.pi, 500, dtype=torch.float64)
        for i in range(13):
            vals = f4(theta, i)
            assert (vals >= -1e-10).all(), f"f4 type {i} has negative values"
            assert (vals <= 1.0 + 1e-10).all(), f"f4 type {i} exceeds 1.0"

    def test_differentiable(self):
        theta = torch.linspace(0.01, math.pi - 0.01, 50, dtype=torch.float64).requires_grad_(True)
        for i in range(13):
            val = f4(theta, i).sum()
            val.backward()
            assert not torch.isnan(theta.grad).any(), f"NaN gradient for f4 type {i}"
            theta.grad.zero_()


class TestF4OfCos:
    """Tests for f4 evaluated on cosine of angle."""

    def test_matches_f4_direct(self):
        """f4_of_cos(cos(theta)) should equal f4(theta)."""
        from oxdna_torch.utils import safe_acos
        theta = torch.linspace(0.1, math.pi - 0.1, 50, dtype=torch.float64)
        cos_theta = torch.cos(theta)
        for i in range(13):
            val_direct = f4(theta, i)
            val_cos = f4_of_cos(cos_theta, i)
            assert torch.allclose(val_direct, val_cos, atol=1e-8), \
                f"f4 vs f4_of_cos mismatch for type {i}"

    def test_differentiable_through_acos(self):
        cos_theta = torch.linspace(-0.9, 0.9, 20, dtype=torch.float64).requires_grad_(True)
        val = f4_of_cos(cos_theta, 0).sum()
        val.backward()
        assert not torch.isnan(cos_theta.grad).any()


class TestF4CxstT1:
    """Tests for the special coaxial stacking theta1 function."""

    def test_symmetric_contribution(self):
        """f4_cxst_t1 should be f4(acos(t)) + f4(2pi - acos(t))."""
        cos_theta = torch.tensor([0.0], dtype=torch.float64)
        val = f4_of_cos_cxst_t1(cos_theta, C.CXST_F4_THETA1)
        # acos(0) = pi/2, 2pi - pi/2 = 3pi/2
        from oxdna_torch.utils import safe_acos
        theta = safe_acos(cos_theta)
        expected = f4(theta, C.CXST_F4_THETA1) + f4(2 * math.pi - theta, C.CXST_F4_THETA1)
        assert torch.allclose(val, expected, atol=1e-10)


class TestF5:
    """Tests for the azimuthal modulation function f5."""

    def test_one_for_positive_cos(self):
        """f5 should be 1.0 when cos_phi >= 0."""
        cos_phi = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        for i in range(4):
            vals = f5(cos_phi, i)
            assert torch.allclose(vals, torch.ones_like(vals), atol=1e-10), \
                f"f5 type {i} not 1.0 for positive cos_phi"

    def test_zero_below_xc(self):
        """f5 should be 0 when cos_phi < XC."""
        for i in range(4):
            xc = C.F5_PHI_XC[i].item()
            cos_phi = torch.tensor([xc - 0.01], dtype=torch.float64)
            val = f5(cos_phi, i)
            assert val.item() == 0.0, f"f5 type {i} not zero below XC"

    def test_values_between_zero_and_one(self):
        """f5 should be in [0, 1]."""
        cos_phi = torch.linspace(-1.0, 1.0, 500, dtype=torch.float64)
        for i in range(4):
            vals = f5(cos_phi, i)
            assert (vals >= -1e-10).all(), f"f5 type {i} has negative values"
            assert (vals <= 1.0 + 1e-10).all(), f"f5 type {i} exceeds 1.0"

    def test_differentiable(self):
        cos_phi = torch.linspace(-0.9, 0.9, 50, dtype=torch.float64).requires_grad_(True)
        for i in range(4):
            val = f5(cos_phi, i).sum()
            val.backward()
            assert not torch.isnan(cos_phi.grad).any()
            cos_phi.grad.zero_()


class TestRepulsiveLJ:
    """Tests for the repulsive Lennard-Jones potential."""

    def test_zero_beyond_cutoff(self):
        """Should be zero for r > rc."""
        r_sq = torch.tensor([C.EXCL_RC1 ** 2 + 0.01], dtype=torch.float64)
        val = repulsive_lj(r_sq, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1)
        assert val.item() == 0.0

    def test_positive_inside_cutoff(self):
        """Should be positive (repulsive) inside cutoff."""
        r_sq = torch.tensor([(C.EXCL_S1 * 0.9) ** 2], dtype=torch.float64)
        val = repulsive_lj(r_sq, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1)
        assert val.item() > 0, f"Expected positive energy, got {val.item()}"

    def test_very_close_is_very_repulsive(self):
        """Energy should increase dramatically at very small r."""
        r_close = torch.tensor([(C.EXCL_S1 * 0.5) ** 2], dtype=torch.float64)
        r_far = torch.tensor([(C.EXCL_S1 * 0.9) ** 2], dtype=torch.float64)
        val_close = repulsive_lj(r_close, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1)
        val_far = repulsive_lj(r_far, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1)
        assert val_close.item() > val_far.item()

    def test_differentiable(self):
        r_sq = torch.tensor([C.EXCL_R1 ** 2 * 0.8], dtype=torch.float64).requires_grad_(True)
        val = repulsive_lj(r_sq, C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1)
        val.backward()
        assert not torch.isnan(r_sq.grad).any()

    def test_all_site_parameter_sets(self):
        """All four excluded volume parameter sets should give positive energy inside cutoff."""
        params = [
            (C.EXCL_S1, C.EXCL_R1, C.EXCL_B1, C.EXCL_RC1),
            (C.EXCL_S2, C.EXCL_R2, C.EXCL_B2, C.EXCL_RC2),
            (C.EXCL_S3, C.EXCL_R3, C.EXCL_B3, C.EXCL_RC3),
            (C.EXCL_S4, C.EXCL_R4, C.EXCL_B4, C.EXCL_RC4),
        ]
        for sigma, rstar, b, rc in params:
            r_sq = torch.tensor([(sigma * 0.95) ** 2], dtype=torch.float64)
            val = repulsive_lj(r_sq, sigma, rstar, b, rc)
            assert val.item() > 0, f"sigma={sigma}: expected positive energy"

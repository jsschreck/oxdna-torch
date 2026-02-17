"""
Numerical stability utilities for differentiable oxDNA.
"""

import torch
from torch import Tensor

# Small epsilon for numerical stability
EPS = 1e-7


def safe_acos(x: Tensor) -> Tensor:
    """Numerically stable arccos that clamps inputs to [-1+eps, 1-eps].

    This avoids NaN gradients at the boundaries of arccos where
    d/dx acos(x) = -1/sqrt(1-x^2) diverges.
    """
    return torch.acos(torch.clamp(x, -1.0 + EPS, 1.0 - EPS))


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Numerically stable vector norm.

    Avoids zero-gradient issues at the origin by adding a small epsilon
    to the squared norm before taking the square root.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + EPS * EPS)


def safe_normalize(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable vector normalization."""
    return x / safe_norm(x, dim=dim, keepdim=True)


def dot(a: Tensor, b: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Batched dot product along specified dimension."""
    return torch.sum(a * b, dim=dim, keepdim=keepdim)


def cross(a: Tensor, b: Tensor) -> Tensor:
    """Batched cross product. a, b: (..., 3)."""
    return torch.linalg.cross(a, b, dim=-1)

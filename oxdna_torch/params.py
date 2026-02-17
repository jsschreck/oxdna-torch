"""
Parameter store for learnable oxDNA parameters.

Manages the distinction between frozen constants (buffers) and
learnable parameters (nn.Parameter). Provides dict-like access
for smooth functions to use tensor values with gradient flow.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Set, Dict

from . import constants as C


# Registry of all parameterizable constants with their default values and shapes
PARAM_REGISTRY = {
    # f1 parameters (2 types: HYDR=0, STCK=1)
    'f1_A':      lambda: C.F1_A.clone(),
    'f1_R0':     lambda: C.F1_R0.clone(),
    'f1_RC':     lambda: C.F1_RC.clone(),
    'f1_BLOW':   lambda: C.F1_BLOW.clone(),
    'f1_BHIGH':  lambda: C.F1_BHIGH.clone(),
    'f1_RLOW':   lambda: C.F1_RLOW.clone(),
    'f1_RHIGH':  lambda: C.F1_RHIGH.clone(),
    'f1_RCLOW':  lambda: C.F1_RCLOW.clone(),
    'f1_RCHIGH': lambda: C.F1_RCHIGH.clone(),

    # f2 parameters (2 types: CRST=0, CXST=1)
    'f2_K':      lambda: C.F2_K.clone(),
    'f2_R0':     lambda: C.F2_R0.clone(),
    'f2_RC':     lambda: C.F2_RC.clone(),
    'f2_BLOW':   lambda: C.F2_BLOW.clone(),
    'f2_BHIGH':  lambda: C.F2_BHIGH.clone(),
    'f2_RLOW':   lambda: C.F2_RLOW.clone(),
    'f2_RHIGH':  lambda: C.F2_RHIGH.clone(),
    'f2_RCLOW':  lambda: C.F2_RCLOW.clone(),
    'f2_RCHIGH': lambda: C.F2_RCHIGH.clone(),

    # f4 parameters (13 types)
    'f4_A':      lambda: C.F4_THETA_A.clone(),
    'f4_B':      lambda: C.F4_THETA_B.clone(),
    'f4_T0':     lambda: C.F4_THETA_T0.clone(),
    'f4_TS':     lambda: C.F4_THETA_TS.clone(),
    'f4_TC':     lambda: C.F4_THETA_TC.clone(),

    # f5 parameters (4 types)
    'f5_A':      lambda: C.F5_PHI_A.clone(),
    'f5_B':      lambda: C.F5_PHI_B.clone(),
    'f5_XC':     lambda: C.F5_PHI_XC.clone(),
    'f5_XS':     lambda: C.F5_PHI_XS.clone(),

    # FENE
    'fene_eps':   lambda: torch.tensor(C.FENE_EPS, dtype=torch.float64),
    'fene_r0':    lambda: torch.tensor(C.FENE_R0_OXDNA, dtype=torch.float64),
    'fene_delta': lambda: torch.tensor(C.FENE_DELTA, dtype=torch.float64),

    # Excluded volume
    'excl_eps':   lambda: torch.tensor(C.EXCL_EPS, dtype=torch.float64),
}


class ParameterStore(nn.Module):
    """Stores oxDNA model parameters as either buffers or nn.Parameters.

    Usage:
        store = ParameterStore(learnable={'f4_A', 'f4_T0', 'excl_eps'})
        params = store.as_dict()  # Dict[str, Tensor] for passing to smooth fns

    Args:
        learnable: set of parameter names to register as nn.Parameter
                   (all others become buffers)
    """

    def __init__(self, learnable: Optional[Set[str]] = None):
        super().__init__()
        self._learnable_names = learnable or set()

        for name in self._learnable_names:
            if name not in PARAM_REGISTRY:
                raise ValueError(
                    f"Unknown parameter '{name}'. "
                    f"Available: {sorted(PARAM_REGISTRY.keys())}"
                )

        for name, factory in PARAM_REGISTRY.items():
            val = factory()
            if name in self._learnable_names:
                setattr(self, name, nn.Parameter(val))
            else:
                self.register_buffer(name, val)

    def as_dict(self) -> Dict[str, Tensor]:
        """Return all parameters as a plain dict for passing to smooth functions."""
        return {name: getattr(self, name) for name in PARAM_REGISTRY}

    @property
    def learnable_names(self) -> Set[str]:
        return self._learnable_names

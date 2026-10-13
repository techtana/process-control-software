"""Component-1-only dynamics / state-space layer (§1.3)."""

from .subspace import subspace_order
from .state_space import identify_first_order_dynamics, assemble_state_space

__all__ = ["subspace_order", "identify_first_order_dynamics", "assemble_state_space"]

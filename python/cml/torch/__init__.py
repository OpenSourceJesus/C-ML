"""PyTorch-like namespace for C-ML (ExecuTorch-inspired runtime façade)."""

from cml.torch.runtime import RuntimeModule, export_pte, load_pte, load_aot
from cml.torch.memory import MemoryManager
from cml.torch.tensor import zeros, ones, randn, options

__all__ = [
    "RuntimeModule",
    "export_pte",
    "load_pte",
    "load_aot",
    "MemoryManager",
    "zeros",
    "ones",
    "randn",
    "options",
]

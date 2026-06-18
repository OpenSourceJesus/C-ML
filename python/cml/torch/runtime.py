"""Runtime module loading and execution (.cpte / AOT)."""

from __future__ import annotations
from typing import Optional
from cml._cml_lib import ffi, lib
from cml.core import Tensor, Module
from cml.torch.memory import MemoryManager


class RuntimeModule:
    """ExecuTorch-style loaded program wrapper."""

    def __init__(self, ptr):
        self._rt = ptr
        self._memory: Optional[MemoryManager] = None

    @classmethod
    def from_module(cls, module: Module) -> "RuntimeModule":
        rt = lib.torch_runtime_from_module(module._module)
        if rt == ffi.NULL:
            raise RuntimeError("Failed to wrap module")
        return cls(rt)

    @classmethod
    def load_pte(cls, path: str) -> "RuntimeModule":
        rt = lib.torch_runtime_load_pte(path.encode())
        if rt == ffi.NULL:
            raise RuntimeError(f"Failed to load PTE: {path}")
        return cls(rt)

    @classmethod
    def load_aot(cls, path: str) -> "RuntimeModule":
        rt = lib.torch_runtime_load_aot(path.encode())
        if rt == ffi.NULL:
            raise RuntimeError(f"Failed to load AOT model: {path}")
        return cls(rt)

    def set_memory(self, memory: MemoryManager) -> None:
        lib.torch_runtime_set_memory(self._rt, memory._mgr)
        self._memory = memory

    def forward(self, x: Tensor) -> Tensor:
        out = lib.torch_runtime_forward(self._rt, x._tensor)
        if out == ffi.NULL:
            raise RuntimeError("Runtime forward failed")
        return Tensor(out)

    def __del__(self):
        if hasattr(self, "_rt") and self._rt != ffi.NULL:
            lib.torch_runtime_free(self._rt)
            self._rt = ffi.NULL


def export_pte(module: Module, sample_input: Tensor, path: str,
               include_weights: bool = True) -> None:
    opts = ffi.new("TorchPTEExportOptions*")
    opts.method_name = ffi.new("char[]", b"forward")
    opts.backend = 0
    opts.include_weights = include_weights
    opts.compute_memory_plan = True
    opts.aot_output_path = ffi.NULL
    rc = lib.torch_runtime_export_pte(
        module._module, sample_input._tensor, path.encode(), opts)
    if rc != 0:
        raise RuntimeError(f"PTE export failed: {path}")


def load_pte(path: str) -> RuntimeModule:
    return RuntimeModule.load_pte(path)


def load_aot(path: str) -> RuntimeModule:
    return RuntimeModule.load_aot(path)

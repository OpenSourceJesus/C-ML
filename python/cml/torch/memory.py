"""User-provided memory arenas for edge deployment."""

from __future__ import annotations
from cml._cml_lib import ffi, lib


class MemoryManager:
    """Wraps a C-ML TorchMemoryManager arena."""

    def __init__(self, size: int):
        self._mgr = lib.torch_memory_create(size)
        if self._mgr == ffi.NULL:
            raise MemoryError(f"Failed to allocate {size}-byte arena")

    @classmethod
    def from_buffer(cls, buffer, size: int) -> "MemoryManager":
        mgr = cls.__new__(cls)
        mgr._mgr = lib.torch_memory_from_buffer(buffer, size)
        if mgr._mgr == ffi.NULL:
            raise MemoryError("Failed to wrap external buffer")
        return mgr

    @property
    def used(self) -> int:
        return int(lib.torch_memory_used(self._mgr))

    @property
    def peak(self) -> int:
        return int(lib.torch_memory_peak(self._mgr))

    def __del__(self):
        if hasattr(self, "_mgr") and self._mgr != ffi.NULL:
            lib.torch_memory_free(self._mgr)
            self._mgr = ffi.NULL

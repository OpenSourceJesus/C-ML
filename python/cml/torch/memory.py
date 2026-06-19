"""User-provided memory arenas for edge deployment."""

from __future__ import annotations
from cml._cml_lib import ffi, lib


class MemoryManager:
    """Wraps a C-ML TorchMemoryManager arena.

    Not thread-safe: do not share one instance across threads while another
    thread may call close() or rely on garbage-collection teardown.
    Prefer explicit close() or a with-block in multi-threaded hosts.
    """

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

    def close(self) -> None:
        if getattr(self, "_mgr", ffi.NULL) != ffi.NULL:
            lib.torch_memory_free(self._mgr)
            self._mgr = ffi.NULL

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        self.close()

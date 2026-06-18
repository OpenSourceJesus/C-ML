"""Tensor helpers via torch_c C API."""

from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU


class TensorOptions:
    def __init__(self, dtype=DTYPE_FLOAT32, device=DEVICE_CPU, requires_grad=False):
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def _to_c(self):
        opts = lib.torch_options()
        opts = lib.torch_options_dtype(opts, self.dtype)
        opts = lib.torch_options_device(opts, self.device)
        opts = lib.torch_options_requires_grad(opts, self.requires_grad)
        return opts


def options(dtype=DTYPE_FLOAT32, device=DEVICE_CPU, requires_grad=False) -> TensorOptions:
    return TensorOptions(dtype=dtype, device=device, requires_grad=requires_grad)


def _make_tensor(c_tensor) -> Tensor:
    return Tensor(c_tensor)


def zeros(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_zeros(c_shape, len(shape), ffi.addressof(opts._to_c()))
    return _make_tensor(t)


def ones(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_ones(c_shape, len(shape), ffi.addressof(opts._to_c()))
    return _make_tensor(t)


def randn(shape: Sequence[int], opts: Optional[TensorOptions] = None) -> Tensor:
    opts = opts or TensorOptions()
    c_shape = ffi.new("int[]", list(shape))
    t = lib.torch_randn(c_shape, len(shape), ffi.addressof(opts._to_c()))
    return _make_tensor(t)

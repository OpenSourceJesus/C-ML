/*
 * torch_c_internal.h — Internal helpers for torch_c hot paths (not public API).
 */

#ifndef CML_TORCH_C_INTERNAL_H
#define CML_TORCH_C_INTERNAL_H

#include "torch/torch_c.h"
#include "autograd/autograd.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "ops/uops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lock-free cached defaults (synced in torch_set_default_*). */
extern DType g_torch_default_dtype;
extern DeviceType g_torch_default_device;

static inline void torch_opts_sync_config(TorchTensorOptions* opts) {
    opts->config.dtype      = opts->has_dtype ? opts->dtype : g_torch_default_dtype;
    opts->config.device     = opts->has_device ? opts->device : g_torch_default_device;
    opts->config.has_dtype  = true;
    opts->config.has_device = true;
}

static inline TensorConfig torch_config_default(void) {
    TensorConfig cfg = {
        .dtype       = g_torch_default_dtype,
        .device      = g_torch_default_device,
        .has_dtype   = true,
        .has_device  = true,
    };
    return cfg;
}

static inline const TensorConfig* torch_resolve_config(const TorchTensorOptions* opts,
                                                       TensorConfig* scratch) {
    if (opts)
        return &opts->config;
    *scratch = torch_config_default();
    return scratch;
}

typedef Tensor* (*TorchCreateFn)(int* shape, int ndim, const TensorConfig* config);

static inline Tensor* torch_create_tensor(TorchCreateFn fn, int* shape, int ndim,
                                          const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = fn(shape, ndim, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_C_INTERNAL_H */

/*
 * torch_c.c — PyTorch-like C API implementation for C-ML
 *
 * Thin façade over cml.h / tensor / nn / autograd / AOT runtime.
 */

#include "torch/torch_c.h"

#include "cml.h"
#include "autograd/autograd.h"
#include "autograd/forward_ops.h"
#include "backend/device.h"
#include "core/config.h"
#include "core/error_stack.h"
#include "ops/uops.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

int torch_init(void) { return cml_init(); }

int torch_cleanup(void) { return cml_cleanup(); }

void torch_get_version(int* major, int* minor, int* patch, const char** version_string) {
    cml_get_version(major, minor, patch, version_string);
}

/* ------------------------------------------------------------------ */
/* TensorOptions                                                       */
/* ------------------------------------------------------------------ */

TorchTensorOptions torch_options(void) {
    TorchTensorOptions opts = {0};
    opts.dtype         = cml_get_default_dtype();
    opts.device        = cml_get_default_device();
    opts.requires_grad = false;
    opts.has_dtype     = false;
    opts.has_device    = false;
    return opts;
}

TorchTensorOptions torch_options_dtype(TorchTensorOptions opts, DType dtype) {
    opts.dtype     = dtype;
    opts.has_dtype = true;
    return opts;
}

TorchTensorOptions torch_options_device(TorchTensorOptions opts, DeviceType device) {
    opts.device     = device;
    opts.has_device = true;
    return opts;
}

TorchTensorOptions torch_options_requires_grad(TorchTensorOptions opts, bool requires_grad) {
    opts.requires_grad = requires_grad;
    return opts;
}

TensorConfig torch_options_to_config(const TorchTensorOptions* opts) {
    TensorConfig cfg = {0};
    if (!opts) {
        cfg.dtype     = cml_get_default_dtype();
        cfg.device    = cml_get_default_device();
        cfg.has_dtype = true;
        cfg.has_device = true;
        return cfg;
    }
    cfg.dtype      = opts->has_dtype ? opts->dtype : cml_get_default_dtype();
    cfg.device     = opts->has_device ? opts->device : cml_get_default_device();
    cfg.has_dtype  = true;
    cfg.has_device = true;
    return cfg;
}

static Tensor* torch_apply_options(Tensor* t, const TorchTensorOptions* opts) {
    if (t && opts && opts->requires_grad)
        cml_set_requires_grad(t, true);
    return t;
}

/* ------------------------------------------------------------------ */
/* Device & dtype                                                      */
/* ------------------------------------------------------------------ */

bool torch_cuda_is_available(void) { return device_cuda_available(); }

int torch_cuda_device_count(void) { return device_cuda_get_count(); }

DeviceType torch_get_default_device(void) { return cml_get_default_device(); }

void torch_set_default_device(DeviceType device) { cml_set_default_device(device); }

DType torch_get_default_dtype(void) { return cml_get_default_dtype(); }

void torch_set_default_dtype(DType dtype) { cml_set_default_dtype(dtype); }

void torch_manual_seed(uint64_t seed) { cml_manual_seed(seed); }

/* ------------------------------------------------------------------ */
/* Tensor lifecycle & accessors                                        */
/* ------------------------------------------------------------------ */

void torch_tensor_retain(Tensor* t) {
    if (t)
        t->ref_count++;
}

void torch_tensor_free(Tensor* t) { tensor_free(t); }

int torch_tensor_ndim(const Tensor* t) { return t ? t->ndim : 0; }

size_t torch_tensor_numel(const Tensor* t) { return t ? t->numel : 0; }

DType torch_tensor_dtype(const Tensor* t) { return t ? t->dtype : DTYPE_FLOAT32; }

DeviceType torch_tensor_device(const Tensor* t) { return t ? t->device : DEVICE_CPU; }

bool torch_tensor_is_contiguous(const Tensor* t) { return t ? t->is_contiguous : false; }

bool torch_tensor_requires_grad(const Tensor* t) { return t ? cml_requires_grad((Tensor*)t) : false; }

void torch_tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (t)
        cml_set_requires_grad(t, requires_grad);
}

const int* torch_tensor_sizes(const Tensor* t) { return t ? t->shape : NULL; }

void* torch_tensor_data_ptr(Tensor* t) { return t ? tensor_data_ptr(t) : NULL; }

float torch_tensor_item_float(Tensor* t) {
    if (!t || t->numel != 1)
        return 0.0f;
    return tensor_get_float(t, 0);
}

void torch_tensor_set_item_float(Tensor* t, float value) {
    if (t && t->numel == 1)
        tensor_set_float(t, 0, value);
}

Tensor* torch_empty(int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_empty(shape, ndim, &cfg), opts);
}

Tensor* torch_zeros(int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_zeros(shape, ndim, &cfg), opts);
}

Tensor* torch_ones(int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_ones(shape, ndim, &cfg), opts);
}

Tensor* torch_full(int* shape, int ndim, const TorchTensorOptions* opts, float value) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_full(shape, ndim, &cfg, value), opts);
}

Tensor* torch_rand(int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_rand(shape, ndim, &cfg), opts);
}

Tensor* torch_randn(int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_randn(shape, ndim, &cfg), opts);
}

Tensor* torch_eye(int n, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_eye(n, &cfg), opts);
}

Tensor* torch_arange(float start, float end, float step, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_arange(start, end, step, &cfg), opts);
}

Tensor* torch_linspace(float start, float end, int steps, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_linspace(start, end, steps, &cfg), opts);
}

Tensor* torch_from_blob(void* data, int* shape, int ndim, const TorchTensorOptions* opts) {
    TensorConfig cfg = torch_options_to_config(opts);
    return torch_apply_options(cml_from_blob(data, shape, ndim, &cfg), opts);
}

Tensor* torch_zeros_like(Tensor* t) { return t ? cml_zeros_like(t) : NULL; }

Tensor* torch_ones_like(Tensor* t) { return t ? cml_ones_like(t) : NULL; }

Tensor* torch_randn_like(Tensor* t) { return t ? cml_randn_like(t) : NULL; }

/* ------------------------------------------------------------------ */
/* Tensor operations                                                   */
/* ------------------------------------------------------------------ */

Tensor* torch_add(Tensor* a, Tensor* b) { return cml_add(a, b); }
Tensor* torch_sub(Tensor* a, Tensor* b) { return cml_sub(a, b); }
Tensor* torch_mul(Tensor* a, Tensor* b) { return cml_mul(a, b); }
Tensor* torch_div(Tensor* a, Tensor* b) { return cml_div(a, b); }
Tensor* torch_matmul(Tensor* a, Tensor* b) { return cml_matmul(a, b); }
Tensor* torch_pow(Tensor* a, Tensor* b) { return cml_pow(a, b); }

Tensor* torch_sum(Tensor* a, int dim, bool keepdim) { return cml_sum(a, dim, keepdim); }
Tensor* torch_mean(Tensor* a, int dim, bool keepdim) { return cml_mean(a, dim, keepdim); }
Tensor* torch_max(Tensor* a, int dim, bool keepdim) { return cml_max(a, dim, keepdim); }
Tensor* torch_min(Tensor* a, int dim, bool keepdim) { return cml_min(a, dim, keepdim); }

Tensor* torch_relu(Tensor* a) { return cml_relu(a); }
Tensor* torch_sigmoid(Tensor* a) { return cml_sigmoid(a); }
Tensor* torch_tanh(Tensor* a) { return cml_tanh(a); }
Tensor* torch_softmax(Tensor* a, int dim) { return cml_softmax(a, dim); }
Tensor* torch_gelu(Tensor* a) { return uop_gelu(a); }

Tensor* torch_reshape(Tensor* a, int* new_shape, int new_ndim) {
    return cml_reshape(a, new_shape, new_ndim);
}
Tensor* torch_transpose(Tensor* a, int dim0, int dim1) { return cml_transpose(a, dim0, dim1); }
Tensor* torch_squeeze(Tensor* a, int dim) { return cml_squeeze(a, dim); }
Tensor* torch_unsqueeze(Tensor* a, int dim) { return cml_unsqueeze(a, dim); }
Tensor* torch_cat(Tensor** tensors, int num_tensors, int dim) {
    return cml_concat(tensors, num_tensors, dim);
}
Tensor* torch_stack(Tensor** tensors, int num_tensors, int dim) {
    return cml_stack(tensors, num_tensors, dim);
}
Tensor* torch_clone(Tensor* a) { return cml_clone(a); }
Tensor* torch_detach(Tensor* a) { return cml_detach(a); }
Tensor* torch_contiguous(Tensor* a) { return cml_contiguous(a); }

/* ------------------------------------------------------------------ */
/* Autograd                                                            */
/* ------------------------------------------------------------------ */

void torch_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    cml_backward(tensor, gradient, retain_graph, create_graph);
}

void torch_zero_grad(Tensor* tensor) { cml_zero_grad(tensor); }

void torch_no_grad(void) { cml_no_grad(); }

void torch_enable_grad(void) { cml_enable_grad(); }

bool torch_is_grad_enabled(void) { return cml_is_grad_enabled(); }

Tensor* torch_get_grad(Tensor* t) { return tensor_get_grad(t); }

/* ------------------------------------------------------------------ */
/* nn.Module API                                                       */
/* ------------------------------------------------------------------ */

Tensor* torch_module_forward(Module* module, Tensor* input) {
    return cml_nn_module_forward(module, input);
}

void torch_module_train(Module* module) { cml_nn_module_train(module); }

void torch_module_eval(Module* module) { cml_nn_module_eval(module); }

bool torch_module_is_training(Module* module) { return cml_nn_module_is_training(module); }

void torch_module_zero_grad(Module* module) { module_zero_grad(module); }

Linear* torch_nn_linear(int in_features, int out_features, bool bias) {
    return cml_nn_linear(in_features, out_features, cml_get_default_dtype(),
                         cml_get_default_device(), bias);
}

ReLU* torch_nn_relu(void) { return cml_nn_relu(false); }

Sequential* torch_nn_sequential(void) { return cml_nn_sequential(); }

void torch_nn_sequential_add(Sequential* seq, Module* layer) {
    cml_nn_sequential_add(seq, layer);
}

Tensor* torch_nn_sequential_forward(Sequential* seq, Tensor* input) {
    return cml_nn_sequential_forward(seq, input);
}

/* ------------------------------------------------------------------ */
/* State dict                                                          */
/* ------------------------------------------------------------------ */

StateDict* torch_module_state_dict(Module* module, const char* prefix) {
    return nn_get_state_dict(module, prefix);
}

int torch_module_load_state_dict(Module* module, const StateDict* sd, bool strict) {
    return nn_load_state_dict(module, sd, strict);
}

Tensor* torch_state_dict_get(const StateDict* sd, const char* key) {
    return nn_state_dict_get(sd, key);
}

void torch_state_dict_free(StateDict* sd) { nn_state_dict_free(sd); }

/* ------------------------------------------------------------------ */
/* Optimizers                                                          */
/* ------------------------------------------------------------------ */

Optimizer* torch_optim_adam(Module* model, float lr, float weight_decay, float beta1, float beta2,
                            float eps) {
    return cml_optim_adam_for_model(model, lr, weight_decay, beta1, beta2, eps);
}

Optimizer* torch_optim_sgd(Module* model, float lr, float momentum, float weight_decay) {
    return cml_optim_sgd_for_model(model, lr, momentum, weight_decay);
}

void torch_optim_zero_grad(Optimizer* optimizer) { cml_optim_zero_grad(optimizer); }

void torch_optim_step(Optimizer* optimizer) { cml_optim_step(optimizer); }

void torch_optim_free(Optimizer* optimizer) { optimizer_free(optimizer); }

/* ------------------------------------------------------------------ */
/* Loss functions                                                      */
/* ------------------------------------------------------------------ */

Tensor* torch_nn_mse_loss(Tensor* input, Tensor* target) {
    return cml_nn_mse_loss(input, target);
}

Tensor* torch_nn_cross_entropy_loss(Tensor* input, Tensor* target) {
    return cml_nn_cross_entropy_loss(input, target);
}

/* ------------------------------------------------------------------ */
/* Runtime module                                                      */
/* ------------------------------------------------------------------ */

TorchRuntimeModule* torch_runtime_from_module(Module* module) {
    if (!module)
        return NULL;
    TorchRuntimeModule* rt = (TorchRuntimeModule*)calloc(1, sizeof(TorchRuntimeModule));
    if (!rt)
        return NULL;
    rt->kind          = TORCH_RUNTIME_EAGER;
    rt->eager_module  = module;
    rt->aot_model     = NULL;
    rt->owns_eager    = false;
    return rt;
}

TorchRuntimeModule* torch_runtime_load_aot(const char* path) {
    if (!path)
        return NULL;
    CMLAOTModel* aot = cml_aot_load(path);
    if (!aot)
        return NULL;
    TorchRuntimeModule* rt = (TorchRuntimeModule*)calloc(1, sizeof(TorchRuntimeModule));
    if (!rt) {
        cml_aot_free(aot);
        return NULL;
    }
    rt->kind         = TORCH_RUNTIME_AOT;
    rt->eager_module = NULL;
    rt->aot_model    = aot;
    rt->pte_model    = NULL;
    rt->memory       = NULL;
    rt->owns_eager   = false;
    return rt;
}

TorchRuntimeModule* torch_runtime_load_pte(const char* path) {
    if (!path)
        return NULL;
    CMLPTEModel* pte = torch_pte_load(path);
    if (!pte)
        return NULL;

    TorchRuntimeModule* rt = (TorchRuntimeModule*)calloc(1, sizeof(TorchRuntimeModule));
    if (!rt) {
        torch_pte_free(pte);
        return NULL;
    }
    rt->kind      = TORCH_RUNTIME_PTE;
    rt->pte_model = pte;
    rt->memory    = NULL;

    int arena = torch_pte_get_required_arena_size(pte);
    if (arena > 0) {
        rt->memory       = torch_memory_create((size_t)arena);
        rt->owns_memory  = true;
        pte->memory      = rt->memory;
    }

    return rt;
}

int torch_runtime_export_pte(Module* module, Tensor* sample_input, const char* path,
                             const TorchPTEExportOptions* opts) {
    return torch_pte_export_module(module, sample_input, path, opts);
}

void torch_runtime_set_memory(TorchRuntimeModule* runtime, TorchMemoryManager* memory) {
    if (!runtime)
        return;
    runtime->memory = memory;
    runtime->owns_memory = false;
    if (runtime->pte_model)
        runtime->pte_model->memory = memory;
}

Tensor* torch_runtime_forward(TorchRuntimeModule* runtime, Tensor* input) {
    if (!runtime || !input)
        return NULL;

    if (runtime->kind == TORCH_RUNTIME_EAGER)
        return module_forward(runtime->eager_module, input);

    if (runtime->kind == TORCH_RUNTIME_AOT && runtime->aot_model) {
        Tensor* inputs[]  = {input};
        Tensor* outputs[1];
        if (cml_aot_execute(runtime->aot_model, inputs, 1, outputs, 1) != 0)
            return NULL;
        return outputs[0];
    }

    if (runtime->kind == TORCH_RUNTIME_PTE && runtime->pte_model) {
        Tensor* inputs[]  = {input};
        Tensor* outputs[1] = {NULL};
        if (torch_pte_execute(runtime->pte_model, inputs, 1, outputs, 1) != 0)
            return NULL;
        return outputs[0];
    }

    return NULL;
}

void torch_runtime_free(TorchRuntimeModule* runtime) {
    if (!runtime)
        return;
    if (runtime->kind == TORCH_RUNTIME_AOT && runtime->aot_model)
        cml_aot_free(runtime->aot_model);
    if (runtime->kind == TORCH_RUNTIME_PTE && runtime->pte_model)
        torch_pte_free(runtime->pte_model);
    if (runtime->owns_memory && runtime->memory)
        torch_memory_free(runtime->memory);
    if (runtime->owns_eager && runtime->eager_module)
        module_free(runtime->eager_module);
    free(runtime);
}

/* ------------------------------------------------------------------ */
/* IR helpers                                                          */
/* ------------------------------------------------------------------ */

void torch_reset_ir(void) { cml_reset_ir_context(); }

void torch_reset_ir_soft(void) { cml_reset_ir_graph_only(); }

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

const char* torch_get_last_error(void) { return error_stack_get_last_message(); }

int torch_get_last_error_code(void) { return error_stack_get_last_code(); }

bool torch_has_error(void) { return error_stack_has_errors(); }

/*
 * pte.c — C-ML PyTorch Edge (.cpte) portable program format
 */

#include "torch/pte.h"
#include "torch/selective_build.h"
#include "torch/memory.h"
#include "torch/delegate.h"
#include "torch/torch_c.h"
#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/context.h"
#include "ops/ir/aot.h"
#include "ops/uops.h"
#include "nn/state.h"
#include "core/logging.h"
#include "core/error_stack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
} PTEFileSection;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_sections;
    uint32_t flags;
    uint64_t total_size;
} PTEFileHeader;

static int pte_write_u32(FILE* f, uint32_t v) { return fwrite(&v, 1, 4, f) == 4 ? 0 : -1; }
static int pte_write_u64(FILE* f, uint64_t v) { return fwrite(&v, 1, 8, f) == 8 ? 0 : -1; }

static int pte_read_u32(FILE* f, uint32_t* v) { return fread(v, 1, 4, f) == 4 ? 0 : -1; }
static int pte_read_u64(FILE* f, uint64_t* v) { return fread(v, 1, 8, f) == 8 ? 0 : -1; }

TorchPTEExportOptions torch_pte_default_export_options(void) {
    TorchPTEExportOptions opts = {0};
    opts.method_name           = "forward";
    opts.backend               = CML_BACKEND_CPU_FALLBACK;
    opts.include_weights         = true;
    opts.compute_memory_plan     = true;
    opts.aot_output_path         = NULL;
    return opts;
}

static size_t pte_dtype_nbytes(DType dtype) {
    switch (dtype) {
    case DTYPE_FLOAT64:
    case DTYPE_INT64:
    case DTYPE_UINT64:
        return 8;
    case DTYPE_FLOAT16:
    case DTYPE_BFLOAT16:
    case DTYPE_INT16:
    case DTYPE_UINT16:
        return 2;
    case DTYPE_INT8:
    case DTYPE_UINT8:
    case DTYPE_BOOL:
        return 1;
    default:
        return 4;
    }
}

static int pte_collect_ir(CMLGraph_t ir, CMLPTEInstruction** out_instrs, int* out_count,
                          CMLPTEMemoryPlan* plan, Module* module) {
    if (!ir || !out_instrs || !out_count)
        return -1;

    int cap   = ir->node_count > 0 ? ir->node_count : 16;
    CMLPTEInstruction* instrs = calloc((size_t)cap, sizeof(CMLPTEInstruction));
    if (!instrs)
        return -1;

    /* Map parameter tensors to constant indices (negative slot encoding) */
    StateDict* sd = module ? nn_get_state_dict(module, "") : NULL;
    int num_consts = sd ? sd->count : 0;

    int idx = 0;
    uint64_t arena = 0;

    for (struct IRNode* node = ir->head; node; node = node->next) {
        if (idx >= cap) {
            cap *= 2;
            CMLPTEInstruction* tmp = realloc(instrs, (size_t)cap * sizeof(CMLPTEInstruction));
            if (!tmp) {
                free(instrs);
                if (sd) nn_state_dict_free(sd);
                return -1;
            }
            instrs = tmp;
        }

        CMLPTEInstruction* ins = &instrs[idx];
        memset(ins, 0, sizeof(*ins));
        ins->kind         = CML_PTE_INSTR_KERNEL;
        ins->kernel_id    = (uint16_t)node->type;
        ins->output_index = idx;

        if (node->output) {
            ins->output_ndim = node->output->ndim;
            for (int d = 0; d < node->output->ndim && d < 8; d++)
                ins->output_shape[d] = node->output->shape[d];
            ins->output_dtype = (int32_t)node->output->dtype;
            arena += node->output->numel * pte_dtype_nbytes(node->output->dtype);
        } else if (node->output_ndim > 0) {
            ins->output_ndim = node->output_ndim;
            for (int d = 0; d < node->output_ndim && d < 8; d++)
                ins->output_shape[d] = node->output_shape[d];
            ins->output_dtype = (int32_t)node->output_dtype;
        }

        ins->num_args = (uint16_t)(node->num_inputs < 8 ? node->num_inputs : 8);
        for (int a = 0; a < ins->num_args && node->inputs; a++) {
            Tensor* in = node->inputs[a];
            int slot   = 0;

            /* Match against prior IR outputs */
            int pidx = 0;
            for (struct IRNode* prev = ir->head; prev && prev != node; prev = prev->next, pidx++) {
                if (prev->output == in) {
                    slot = 1 + pidx; /* +1: slot 0 reserved for graph input */
                    break;
                }
            }

            /* Match against constants/parameters */
            if (slot == 0 && sd && in) {
                for (int c = 0; c < sd->count; c++) {
                    if (sd->entries[c].value == in) {
                        slot = -(c + 1);
                        break;
                    }
                }
            }

            ins->arg_indices[a] = slot;
        }

        idx++;
    }

    if (sd)
        nn_state_dict_free(sd);

    if (plan) {
        plan->num_buffers       = 1;
        plan->total_bytes       = arena;
        plan->peak_bytes        = arena;
        plan->buffer_sizes[0]   = arena;
        plan->buffer_offsets[0] = 0;
    }

    *out_instrs = instrs;
    *out_count  = idx;
    return 0;
}

static int pte_collect_constants(Module* module, CMLPTEConstant** out_consts, uint8_t** out_data,
                                 size_t* out_data_size, int* out_count) {
    StateDict* sd = nn_get_state_dict(module, "");
    if (!sd)
        return -1;

    CMLPTEConstant* consts = calloc((size_t)sd->count, sizeof(CMLPTEConstant));
    if (!consts) {
        nn_state_dict_free(sd);
        return -1;
    }

    size_t total = 0;
    for (int i = 0; i < sd->count; i++) {
        Tensor* t = sd->entries[i].value;
        if (!t)
            continue;
        tensor_ensure_executed(t);
        total += t->numel * pte_dtype_nbytes(t->dtype);
    }

    uint8_t* blob = total > 0 ? malloc(total) : NULL;
    if (total > 0 && !blob) {
        free(consts);
        nn_state_dict_free(sd);
        return -1;
    }

    size_t offset = 0;
    int n = 0;
    for (int i = 0; i < sd->count; i++) {
        Tensor* t = sd->entries[i].value;
        if (!t)
            continue;

        CMLPTEConstant* c = &consts[n++];
        strncpy(c->name, sd->entries[i].key ? sd->entries[i].key : "", sizeof(c->name) - 1);
        c->ndim   = t->ndim;
        c->dtype  = (int32_t)t->dtype;
        c->offset = offset;
        for (int d = 0; d < t->ndim && d < 8; d++)
            c->shape[d] = t->shape[d];

        size_t nbytes = t->numel * pte_dtype_nbytes(t->dtype);
        c->nbytes = nbytes;
        if (blob && t->data)
            memcpy(blob + offset, t->data, nbytes);
        offset += nbytes;
    }

    nn_state_dict_free(sd);
    *out_consts     = consts;
    *out_data       = blob;
    *out_data_size  = offset;
    *out_count      = n;
    return 0;
}

int torch_pte_export_module(Module* module, Tensor* sample_input, const char* path,
                            const TorchPTEExportOptions* opts) {
    if (!module || !sample_input || !path)
        return -1;

    TorchPTEExportOptions o = opts ? *opts : torch_pte_default_export_options();

    cml_ir_reset_global_context();
    Tensor* output = module_forward(module, sample_input);
    if (!output) {
        LOG_ERROR("PTE export: forward pass failed");
        return -1;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        LOG_ERROR("PTE export: no IR captured");
        return -1;
    }

    CMLPTEInstruction* instrs = NULL;
    int num_instrs            = 0;
    CMLPTEMemoryPlan plan     = {0};
    if (pte_collect_ir(ir, &instrs, &num_instrs, o.compute_memory_plan ? &plan : NULL, module) != 0) {
        LOG_ERROR("PTE export: IR collection failed");
        return -1;
    }

    CMLPTEConstant* consts = NULL;
    uint8_t* const_data    = NULL;
    size_t const_data_size = 0;
    int num_consts         = 0;
    if (o.include_weights &&
        pte_collect_constants(module, &consts, &const_data, &const_data_size, &num_consts) != 0) {
        LOG_ERROR("PTE export: constant collection failed");
        free(instrs);
        return -1;
    }

    CMLPTEMetadata meta = {0};
    strncpy(meta.method_name, o.method_name ? o.method_name : "forward", sizeof(meta.method_name) - 1);
    meta.backend_id       = (uint32_t)o.backend;
    meta.num_inputs       = 1;
    meta.num_outputs      = 1;
    meta.num_instructions = (uint32_t)num_instrs;
    meta.num_constants    = (uint32_t)num_consts;
    meta.arena_size       = plan.peak_bytes;

    /* Build section payloads in memory */
    size_t prog_size = (size_t)num_instrs * sizeof(CMLPTEInstruction);
    size_t const_meta_size = (size_t)num_consts * sizeof(CMLPTEConstant);
    size_t meta_size = sizeof(CMLPTEMetadata);
    size_t plan_size = sizeof(CMLPTEMemoryPlan);

    uint32_t num_sections = o.include_weights ? 4u : 3u;
    PTEFileSection sections[4];
    memset(sections, 0, sizeof(sections));

    size_t cursor = sizeof(PTEFileHeader) + num_sections * sizeof(PTEFileSection);

    sections[0].type   = CML_PTE_SECTION_PROGRAM;
    sections[0].offset = cursor;
    sections[0].size   = prog_size;
    cursor += prog_size;

    sections[1].type   = CML_PTE_SECTION_CONSTANTS;
    sections[1].offset = cursor;
    sections[1].size   = const_meta_size + const_data_size;
    cursor += sections[1].size;

    sections[2].type   = CML_PTE_SECTION_MEMORY_PLAN;
    sections[2].offset = cursor;
    sections[2].size   = plan_size;
    cursor += plan_size;

    sections[3].type   = CML_PTE_SECTION_METADATA;
    sections[3].offset = cursor;
    sections[3].size   = meta_size;
    cursor += meta_size;

    FILE* f = fopen(path, "wb");
    if (!f) {
        free(instrs);
        free(consts);
        free(const_data);
        return -1;
    }

    PTEFileHeader hdr = {
        .magic         = CML_PTE_MAGIC,
        .version       = CML_PTE_VERSION,
        .num_sections  = num_sections,
        .flags         = 0,
        .total_size    = cursor,
    };
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(sections, sizeof(PTEFileSection), num_sections, f);
    fwrite(instrs, 1, prog_size, f);

    if (num_consts > 0) {
        fwrite(consts, 1, const_meta_size, f);
        if (const_data_size > 0)
            fwrite(const_data, 1, const_data_size, f);
    }

    fwrite(&plan, 1, plan_size, f);
    fwrite(&meta, 1, meta_size, f);
    fclose(f);

    /* Write selective-build manifest alongside .cpte */
    TorchSelectiveBuildConfig sb = torch_selective_build_none();
    for (int i = 0; i < num_instrs; i++)
        torch_selective_build_enable_op(&sb, (UOpType)instrs[i].kernel_id);
    char sb_path[512];
    snprintf(sb_path, sizeof(sb_path), "%s.kernels", path);
    torch_selective_build_save(&sb, sb_path);

    if (o.aot_output_path) {
        AOTCompileOptions aot = cml_aot_default_options();
        cml_aot_compile(ir, o.aot_output_path, &aot);
    }

    free(instrs);
    free(consts);
    free(const_data);

    LOG_INFO("PTE exported: %s (%u instructions, %u constants, arena=%llu bytes)", path,
             meta.num_instructions, meta.num_constants, (unsigned long long)meta.arena_size);
    return 0;
}

CMLPTEModel* torch_pte_load(const char* path) {
    if (!path)
        return NULL;

    FILE* f = fopen(path, "rb");
    if (!f)
        return NULL;

    PTEFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != CML_PTE_MAGIC) {
        fclose(f);
        return NULL;
    }

    PTEFileSection* sections = calloc(hdr.num_sections, sizeof(PTEFileSection));
    if (!sections) {
        fclose(f);
        return NULL;
    }
    if (fread(sections, sizeof(PTEFileSection), hdr.num_sections, f) != hdr.num_sections) {
        free(sections);
        fclose(f);
        return NULL;
    }

    CMLPTEModel* model = calloc(1, sizeof(CMLPTEModel));
    if (!model) {
        free(sections);
        fclose(f);
        return NULL;
    }
    model->path = strdup(path);

    /* Read metadata first */
    for (uint32_t s = 0; s < hdr.num_sections; s++) {
        if (sections[s].type != CML_PTE_SECTION_METADATA)
            continue;
        fseek(f, (long)sections[s].offset, SEEK_SET);
        fread(&model->meta, 1, sizeof(CMLPTEMetadata), f);
    }

    for (uint32_t s = 0; s < hdr.num_sections; s++) {
        fseek(f, (long)sections[s].offset, SEEK_SET);
        if (sections[s].type == CML_PTE_SECTION_PROGRAM) {
            int n = (int)(sections[s].size / sizeof(CMLPTEInstruction));
            model->instructions = calloc((size_t)n, sizeof(CMLPTEInstruction));
            if (model->instructions)
                fread(model->instructions, sizeof(CMLPTEInstruction), (size_t)n, f);
        } else if (sections[s].type == CML_PTE_SECTION_CONSTANTS) {
            uint32_t nc = model->meta.num_constants;
            size_t meta_bytes = (size_t)nc * sizeof(CMLPTEConstant);
            if (nc > 0) {
                model->constants = calloc(nc, sizeof(CMLPTEConstant));
                fread(model->constants, sizeof(CMLPTEConstant), nc, f);
            }
            size_t data_bytes = sections[s].size > meta_bytes ? sections[s].size - meta_bytes : 0;
            if (data_bytes > 0) {
                model->constant_data = malloc(data_bytes);
                if (model->constant_data)
                    fread(model->constant_data, 1, data_bytes, f);
                model->constant_data_size = data_bytes;
            }
        } else if (sections[s].type == CML_PTE_SECTION_MEMORY_PLAN) {
            fread(&model->memory_plan, 1, sizeof(CMLPTEMemoryPlan), f);
        }
    }

    free(sections);
    fclose(f);

    TorchSelectiveBuildConfig sb = torch_selective_build_none();
    for (uint32_t i = 0; i < model->meta.num_instructions; i++)
        torch_selective_build_enable_op(&sb, (UOpType)model->instructions[i].kernel_id);
    torch_selective_build_apply(&sb);

    LOG_INFO("PTE loaded: %s (%u instructions, %u constants)", path,
             model->meta.num_instructions, model->meta.num_constants);
    return model;
}

void torch_pte_free(CMLPTEModel* model) {
    if (!model)
        return;
    free(model->path);
    free(model->instructions);
    free(model->constants);
    free(model->constant_data);
    free(model);
}

int torch_pte_get_required_arena_size(const CMLPTEModel* model) {
    if (!model)
        return 0;
    if (model->memory_plan.peak_bytes > 0)
        return (int)model->memory_plan.peak_bytes;
    return (int)model->meta.arena_size;
}

/* Map UOpType to cml function for linear interpreter */
static Tensor* pte_exec_kernel(UOpType op, Tensor** args, int num_args,
                               const CMLPTEInstruction* ins) {
    if (!torch_selective_build_is_op_enabled(op))
        return NULL;

    switch (op) {
    case UOP_ADD:
        return num_args >= 2 ? cml_add(args[0], args[1]) : NULL;
    case UOP_SUB:
        return num_args >= 2 ? cml_sub(args[0], args[1]) : NULL;
    case UOP_MUL:
        return num_args >= 2 ? cml_mul(args[0], args[1]) : NULL;
    case UOP_DIV:
        return num_args >= 2 ? cml_div(args[0], args[1]) : NULL;
    case UOP_MATMUL:
        return num_args >= 2 ? cml_matmul(args[0], args[1]) : NULL;
    case UOP_RELU:
        return num_args >= 1 ? cml_relu(args[0]) : NULL;
    case UOP_SIGMOID:
        return num_args >= 1 ? cml_sigmoid(args[0]) : NULL;
    case UOP_TANH:
        return num_args >= 1 ? cml_tanh(args[0]) : NULL;
    case UOP_SUM:
        return num_args >= 1 ? cml_sum(args[0], -1, false) : NULL;
    case UOP_MEAN:
        return num_args >= 1 ? cml_mean(args[0], -1, false) : NULL;
    case UOP_RESHAPE: {
        if (num_args < 1 || ins->output_ndim <= 0)
            return NULL;
        int shape[8];
        for (int d = 0; d < ins->output_ndim; d++)
            shape[d] = ins->output_shape[d];
        return cml_reshape(args[0], shape, ins->output_ndim);
    }
    case UOP_LINEAR:
        return num_args >= 3 ? uop_linear(args[0], args[1], args[2])
               : num_args >= 2 ? uop_linear(args[0], args[1], NULL)
                               : NULL;
    default:
        LOG_WARNING("PTE execute: unsupported kernel %d", (int)op);
        return NULL;
    }
}

int torch_pte_execute(CMLPTEModel* model, Tensor** inputs, int num_inputs,
                      Tensor** outputs, int num_outputs) {
    if (!model || !inputs || num_inputs < 1 || !outputs || num_outputs < 1)
        return -1;

    int n = (int)model->meta.num_instructions;
    int nc = (int)model->meta.num_constants;

    /* Materialize constant tensors from blob */
    Tensor** const_tensors = NULL;
    if (nc > 0) {
        const_tensors = calloc((size_t)nc, sizeof(Tensor*));
        TorchTensorOptions opts = torch_options();
        for (int c = 0; c < nc; c++) {
            CMLPTEConstant* meta = &model->constants[c];
            int shape[8];
            for (int d = 0; d < meta->ndim; d++)
                shape[d] = meta->shape[d];
            opts = torch_options_dtype(opts, (DType)meta->dtype);
            Tensor* t = torch_empty(shape, meta->ndim, &opts);
            if (t && model->constant_data && meta->offset + meta->nbytes <= model->constant_data_size) {
                void* dst = torch_tensor_data_ptr(t);
                if (dst)
                    memcpy(dst, model->constant_data + meta->offset, (size_t)meta->nbytes);
            }
            const_tensors[c] = t;
        }
    }

    Tensor** intermediates = calloc((size_t)n, sizeof(Tensor*));
    if (!intermediates) {
        if (const_tensors) {
            for (int c = 0; c < nc; c++)
                torch_tensor_free(const_tensors[c]);
            free(const_tensors);
        }
        return -1;
    }

    for (uint32_t i = 0; i < model->meta.num_instructions; i++) {
        const CMLPTEInstruction* ins = &model->instructions[i];
        Tensor* args[8] = {0};

        for (int a = 0; a < ins->num_args && a < 8; a++) {
            int slot = ins->arg_indices[a];
            if (slot == 0)
                args[a] = inputs[0];
            else if (slot < 0 && const_tensors)
                args[a] = const_tensors[-slot - 1];
            else if (slot > 0 && slot <= n)
                args[a] = intermediates[slot - 1];
        }

        if (ins->kind == CML_PTE_INSTR_DELEGATE) {
            TorchDelegate* d = torch_delegate_find_by_backend((CMLBackendType)ins->delegate_id);
            if (d && d->execute) {
                Tensor* out = NULL;
                d->execute(d, NULL, 0, args, ins->num_args, &out, 1);
                intermediates[i] = out;
            }
            continue;
        }

        intermediates[i] = pte_exec_kernel((UOpType)ins->kernel_id, args, ins->num_args, ins);
    }

    outputs[0] = n > 0 ? intermediates[n - 1] : inputs[0];

    for (int i = 0; i < n - 1; i++) {
        if (intermediates[i] && intermediates[i] != outputs[0])
            tensor_free(intermediates[i]);
    }

    if (const_tensors) {
        for (int c = 0; c < nc; c++)
            torch_tensor_free(const_tensors[c]);
        free(const_tensors);
    }
    free(intermediates);
    return outputs[0] ? 0 : -1;
}

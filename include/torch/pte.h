/*
 * pte.h — C-ML PyTorch Edge (.cpte) portable program format
 *
 * ExecuTorch PTE analogue: ahead-of-time serialized program with linear
 * instruction list, constant/weight segments, memory plan, and backend metadata.
 */

#ifndef CML_TORCH_PTE_H
#define CML_TORCH_PTE_H

#include "core/export.h"
#include "ops/ir/dispatch.h"
#include "tensor/tensor.h"
#include "nn.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_PTE_MAGIC   0x45545043u /* "CPTE" */
#define CML_PTE_VERSION 1u

typedef enum {
    CML_PTE_SECTION_PROGRAM     = 1,
    CML_PTE_SECTION_CONSTANTS   = 2,
    CML_PTE_SECTION_MEMORY_PLAN = 3,
    CML_PTE_SECTION_METADATA    = 4,
} CMLPTESectionType;

typedef enum {
    CML_PTE_INSTR_KERNEL  = 1,
    CML_PTE_INSTR_DELEGATE = 2,
    CML_PTE_INSTR_MOVE    = 3,
} CMLPTEInstrKind;

typedef struct CMLPTEInstruction {
    uint16_t kind;
    uint16_t kernel_id;   /* UOpType for kernel calls */
    uint16_t num_args;
    uint16_t delegate_id;
    int32_t  arg_indices[8];
    int32_t  output_index;
    int32_t  output_ndim;
    int32_t  output_shape[8];
    int32_t  output_dtype;
} CMLPTEInstruction;

typedef struct CMLPTEMetadata {
    char     method_name[64];
    uint32_t backend_id;      /* CMLBackendType */
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t num_instructions;
    uint32_t num_constants;
    uint64_t arena_size;
} CMLPTEMetadata;

typedef struct CMLPTEConstant {
    char     name[128];
    int32_t  ndim;
    int32_t  shape[8];
    int32_t  dtype;
    uint64_t offset;
    uint64_t nbytes;
} CMLPTEConstant;

typedef struct CMLPTEMemoryPlan {
    uint32_t num_buffers;
    uint64_t total_bytes;
    uint64_t peak_bytes;
    uint64_t buffer_sizes[64];
    uint64_t buffer_offsets[64];
} CMLPTEMemoryPlan;

typedef struct TorchPTEExportOptions {
    const char*      method_name;
    CMLBackendType   backend;
    bool             include_weights;
    bool             compute_memory_plan;
    const char*      aot_output_path; /* optional: also emit .so alongside .cpte */
} TorchPTEExportOptions;

typedef struct CMLPTEModel {
    char*                  path;
    CMLPTEMetadata         meta;
    CMLPTEInstruction*     instructions;
    CMLPTEConstant*        constants;
    CMLPTEMemoryPlan       memory_plan;
    uint8_t*               constant_data;
    size_t                 constant_data_size;
    struct TorchMemoryManager* memory;
    struct TorchDelegateRegistry* delegates;
} CMLPTEModel;

CML_API TorchPTEExportOptions torch_pte_default_export_options(void);

CML_API int torch_pte_export_module(Module* module, Tensor* sample_input, const char* path,
                                    const TorchPTEExportOptions* opts);

CML_API CMLPTEModel* torch_pte_load(const char* path);
CML_API void         torch_pte_free(CMLPTEModel* model);

CML_API int  torch_pte_execute(CMLPTEModel* model, Tensor** inputs, int num_inputs,
                               Tensor** outputs, int num_outputs);
CML_API int  torch_pte_get_required_arena_size(const CMLPTEModel* model);

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_PTE_H */

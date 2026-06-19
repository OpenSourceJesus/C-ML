#include "torch/torch_c.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void test_memory_arena(void) {
    printf("  test_memory_arena...");
    TorchMemoryManager* mgr = torch_memory_create(1024 * 1024);
    assert(mgr != NULL);

    void* a = torch_memory_alloc(mgr, 4096);
    void* b = torch_memory_alloc(mgr, 8192);
    assert(a && b);
    assert(torch_memory_used(mgr) >= 4096 + 8192);

    torch_memory_reset(mgr);
    assert(torch_memory_used(mgr) == 0);

    torch_memory_free(mgr);
    printf(" PASSED\n");
}

static void test_selective_build(void) {
    printf("  test_selective_build...");
    TorchSelectiveBuildConfig cfg;
    assert(torch_selective_build_from_string("add,mul,matmul,relu", &cfg) == 0);
    assert(cfg.enabled[UOP_ADD]);
    assert(cfg.enabled[UOP_MATMUL]);
    assert(!cfg.enabled[UOP_DIV]);

    torch_selective_build_apply(&cfg);
    assert(torch_selective_build_is_op_enabled(UOP_ADD));
    assert(!torch_selective_build_is_op_enabled(UOP_DIV));

    torch_selective_build_reset();
    assert(torch_selective_build_is_op_enabled(UOP_DIV));
    printf(" PASSED\n");
}

static void test_delegate(void) {
    printf("  test_delegate...");
    TorchDelegate* cpu = torch_delegate_cpu();
    assert(cpu != NULL);
    assert(torch_delegate_find("cpu") != NULL);
    assert(torch_delegate_supports_op(cpu, UOP_ADD));
    printf(" PASSED\n");
}

static void test_pte_roundtrip(void) {
    printf("  test_pte_roundtrip...");

    char path[] = "/tmp/cml_test_XXXXXX.cpte";
    int fd = mkstemps(path, 5);
    assert(fd >= 0);
    close(fd);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(4, 2, true));

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {1, 4};
    Tensor* sample = torch_randn(shape, 2, &opts);

    TorchPTEExportOptions export_opts = torch_pte_default_export_options();
    assert(torch_pte_export_module((Module*)model, sample, path, &export_opts) == 0);

    TorchRuntimeModule* rt = torch_runtime_load_pte(path);
    assert(rt != NULL);
    assert(torch_runtime_get_kind(rt) == TORCH_RUNTIME_PTE);
    assert(torch_runtime_has_memory(rt) || torch_runtime_pte_arena_size(rt) == 0);

    Tensor* out = torch_runtime_forward(rt, sample);
    assert(out != NULL);
    assert(torch_tensor_numel(out) == 2);

    torch_tensor_free(out);
    torch_runtime_free(rt);
    torch_tensor_free(sample);
    module_free((Module*)model);
    torch_reset_ir();
    unlink(path);

    char sb_path[512];
    snprintf(sb_path, sizeof(sb_path), "%s.kernels", path);
    unlink(sb_path);

    printf(" PASSED\n");
}

int main(void) {
    torch_init();
    printf("Running torch runtime tests:\n");

    test_memory_arena();
    test_selective_build();
    test_delegate();
    test_pte_roundtrip();

    torch_cleanup();
    printf("All torch runtime tests passed.\n");
    return 0;
}

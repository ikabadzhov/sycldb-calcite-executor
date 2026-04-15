#pragma once

#include <sycl/sycl.hpp>
#include "selection.hpp"
#include "aggregation.hpp"

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL [[sycl::external]]
#endif

extern "C" {
    struct SYCLDBContext {
        const int* col_ptrs[8];
        const int* col_ptrs2[8];
        int* res_ptrs[8];
        const bool* ht_ptrs[8];
        const int* ht_int_ptrs[8];
        int values[8];
        int params[64];
        uint32_t bypass_write_mask;

        // Flattened members for direct SROA
        const int* p0; const int* p1; const int* p2; const int* p3;
        const int* p4; const int* p5; const int* p6; const int* p7;
        const bool* h0; const bool* h1; const bool* h2; const bool* h3;
    };

#define DECLARE_JIT_SLOTS(name) \
    SYCL_EXTERNAL void name##_0(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_1(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_2(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_3(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_4(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_5(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_6(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_7(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc);

    DECLARE_JIT_SLOTS(selection_literal_jit)
    DECLARE_JIT_SLOTS(selection_columns_jit)
    DECLARE_JIT_SLOTS(perform_op_columns_jit)
    DECLARE_JIT_SLOTS(perform_op_literal_first_jit)
    DECLARE_JIT_SLOTS(perform_op_literal_second_jit)
    DECLARE_JIT_SLOTS(filter_join_jit)
    DECLARE_JIT_SLOTS(full_join_jit)
    DECLARE_JIT_SLOTS(aggregate_jit)

    // Call sequence dispatcher (implemented by JIT)
    SYCL_EXTERNAL void execute_sycldb_ops_agg(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc);

    void force_emit_kernels(sycl::queue& q);
}

#pragma once

#include <sycl/sycl.hpp>
#include "selection.hpp"
#include "aggregation.hpp"

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL [[sycl::external]]
#endif

extern "C" {
    struct SYCLDBContext {
        const int* col_ptrs[32];
        int* res_ptrs[32];
        int values[32];
        const bool* ht_ptrs[32];
        const int* ht_int_ptrs[32];
        int params[256]; // generic parameters: 32 slots * 8 params each
    };

#define DECLARE_JIT_SLOTS(name) \
    SYCL_EXTERNAL void name##_0(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_1(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_2(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_3(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_4(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_5(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_6(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc); \
    SYCL_EXTERNAL void name##_7(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc);

    DECLARE_JIT_SLOTS(selection_literal_jit)
    DECLARE_JIT_SLOTS(selection_columns_jit)
    DECLARE_JIT_SLOTS(perform_op_columns_jit)
    DECLARE_JIT_SLOTS(perform_op_literal_first_jit)
    DECLARE_JIT_SLOTS(perform_op_literal_second_jit)
    DECLARE_JIT_SLOTS(filter_join_jit)
    DECLARE_JIT_SLOTS(full_join_jit)
    DECLARE_JIT_SLOTS(aggregate_jit)

    // Call sequence dispatcher (implemented by JIT)
    SYCL_EXTERNAL void execute_sycldb_ops_agg(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc);

    void force_emit_kernels(sycl::queue& q);
}

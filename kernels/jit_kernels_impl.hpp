#pragma once

#include "jit_kernels.hpp"
#include "join.hpp"

extern "C" {

#define JIT_IMPL_SLOT_INLINE(name, i) \
    SYCL_EXTERNAL __attribute__((used)) inline void name##_##i(sycl::item<1> idx, SYCLDBContext ctx, bool& pass, uint64_t& acc) { \
        name##_impl(idx, ctx, i, pass, acc); \
    }

#define JIT_IMPL_ALL_SLOTS_INLINE(name) \
    JIT_IMPL_SLOT_INLINE(name, 0) \
    JIT_IMPL_SLOT_INLINE(name, 1) \
    JIT_IMPL_SLOT_INLINE(name, 2) \
    JIT_IMPL_SLOT_INLINE(name, 3) \
    JIT_IMPL_SLOT_INLINE(name, 4) \
    JIT_IMPL_SLOT_INLINE(name, 5) \
    JIT_IMPL_SLOT_INLINE(name, 6) \
    JIT_IMPL_SLOT_INLINE(name, 7)

    inline void selection_literal_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) pass = logical((logical_op)ctx.params[i*8+1], pass, compare((comp_op)ctx.params[i*8+0], ctx.col_ptrs[i][idx], ctx.values[i]));
    }
    JIT_IMPL_ALL_SLOTS_INLINE(selection_literal_jit)

    inline void selection_columns_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) pass = logical((logical_op)ctx.params[i*8+1], pass, compare((comp_op)ctx.params[i*8+0], ctx.col_ptrs[i][idx], ctx.col_ptrs[i+8][idx]));
    }
    JIT_IMPL_ALL_SLOTS_INLINE(selection_columns_jit)

    inline void perform_op_columns_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            ctx.res_ptrs[i][idx] = element_operation(ctx.col_ptrs[i][idx], ctx.col_ptrs[i+8][idx], (BinaryOp)ctx.params[i*8+2]);
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_columns_jit)

    inline void perform_op_literal_first_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            ctx.res_ptrs[i][idx] = element_operation(ctx.values[i], ctx.col_ptrs[i][idx], (BinaryOp)ctx.params[i*8+2]);
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_literal_first_jit)

    inline void perform_op_literal_second_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            ctx.res_ptrs[i][idx] = element_operation(ctx.col_ptrs[i][idx], ctx.values[i], (BinaryOp)ctx.params[i*8+2]);
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_literal_second_jit)

    inline void filter_join_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int val = ctx.col_ptrs[i][idx];
            if (val >= ctx.params[i*8+3] && val <= ctx.params[i*8+4]) {
                pass = ctx.ht_ptrs[i][HASH(val, ctx.params[i*8+5], ctx.params[i*8+3])];
            } else {
                pass = false;
            }
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(filter_join_jit)

    inline void full_join_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int val = ctx.col_ptrs[i][idx];
            int ht_min = ctx.params[i*8+3];
            int ht_max = ctx.params[i*8+4];
            int ht_len = ctx.params[i*8+5];
            if (val >= ht_min && val <= ht_max) {
                int hash = HASH(val, ht_len, ht_min) << 1;
                if (ctx.ht_int_ptrs[i][hash] == 1) {
                    ctx.res_ptrs[i][idx] = ctx.ht_int_ptrs[i][hash+1];
                } else {
                    pass = false;
                }
            } else {
                pass = false;
            }
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(full_join_jit)

    inline void aggregate_jit_impl(sycl::item<1> idx, SYCLDBContext ctx, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            acc += (uint64_t)ctx.col_ptrs[i][idx];
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(aggregate_jit)
}

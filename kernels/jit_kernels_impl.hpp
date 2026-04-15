#include "jit_kernels.hpp"
#include "selection.hpp"
#include "join.hpp"
#include <sycl/sycl.hpp>

extern "C" {

#define JIT_IMPL_SLOT_INLINE(name, i) \
    SYCL_EXTERNAL __attribute__((always_inline)) inline void name##_##i(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc) { \
        name##_impl(idx, ctx, regs, i, pass, acc); \
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

    inline bool inline_compare(int v1, int v2, int op) {
        if (op == 0) return (v1 == v2);
        if (op == 2) return (v1 < v2);
        if (op == 3) return (v1 <= v2);
        if (op == 4) return (v1 > v2);
        if (op == 5) return (v1 >= v2);
        return (v1 != v2);
    }

#define GET_COL_PTR(ctx, i) ((i==0)?ctx.p0 : (i==1)?ctx.p1 : (i==2)?ctx.p2 : (i==3)?ctx.p3 : (i==4)?ctx.p4 : (i==5)?ctx.p5 : (i==6)?ctx.p6 : (i==7)?ctx.p7 : ctx.col_ptrs[i])
#define GET_HT_PTR(ctx, i) ((i==0)?ctx.h0 : (i==1)?ctx.h1 : (i==2)?ctx.h2 : (i==3)?ctx.h3 : ctx.ht_ptrs[i])

    inline void selection_literal_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            int r_wr = ctx.params[i*8+7];
            const int* col = (const int*)__builtin_assume_aligned(GET_COL_PTR(ctx, i), 64);
            int col_val = (r_rd >= 0) ? regs[r_rd] : col[idx[0]];
            if (r_wr >= 0) regs[r_wr] = col_val;
            bool res = inline_compare(col_val, ctx.values[i], ctx.params[i*8+0]);
            if (ctx.params[i*8+1] == 0) pass = pass && res; 
            else if (ctx.params[i*8+1] == 1) pass = pass || res;
            else pass = res;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(selection_literal_jit)

    inline void selection_columns_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd1 = ctx.params[i*8+6];
            int r_rd2 = ctx.params[i*8+5];
            int v1 = (r_rd1 >= 0) ? regs[r_rd1] : ctx.col_ptrs[i][idx[0]];
            int v2 = (r_rd2 >= 0) ? regs[r_rd2] : ctx.col_ptrs2[i][idx[0]];
            bool res = inline_compare(v1, v2, ctx.params[i*8+0]);
            if (ctx.params[i*8+1] == 0) pass = pass && res;
            else if (ctx.params[i*8+1] == 1) pass = pass || res;
            else pass = res;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(selection_columns_jit)

    inline void perform_op_columns_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd1 = ctx.params[i*8+6];
            int r_rd2 = ctx.params[i*8+5];
            int v1 = (r_rd1 >= 0) ? regs[r_rd1] : ctx.col_ptrs[i][idx[0]];
            int v2 = (r_rd2 >= 0) ? regs[r_rd2] : ctx.col_ptrs2[i][idx[0]];
            int op = ctx.params[i*8+2];
            int res = 0;
            if (op == 0) res = v1 + v2; else if (op == 1) res = v1 - v2;
            else if (op == 2) res = v1 * v2; else if (op == 3 && v2 != 0) res = v1 / v2;
            regs[i] = res;
            if (!(ctx.bypass_write_mask & (1 << i))) ctx.res_ptrs[i][idx[0]] = res;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_columns_jit)

    inline void perform_op_literal_first_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            int v = (r_rd >= 0) ? regs[r_rd] : ctx.col_ptrs[i][idx[0]];
            int res = 0; int op = ctx.params[i*8+2];
            int lit = ctx.values[i];
            if (op == 0) res = lit + v; else if (op == 1) res = lit - v;
            else if (op == 2) res = lit * v; else if (op == 3 && v != 0) res = lit / v;
            regs[i] = res;
            if (!(ctx.bypass_write_mask & (1 << i))) ctx.res_ptrs[i][idx[0]] = res;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_literal_first_jit)

    inline void perform_op_literal_second_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            int v = (r_rd >= 0) ? regs[r_rd] : ctx.col_ptrs[i][idx[0]];
            int res = 0; int op = ctx.params[i*8+2];
            int lit = ctx.values[i];
            if (op == 0) res = v + lit; else if (op == 1) res = v - lit;
            else if (op == 2) res = v * lit; else if (op == 3 && lit != 0) res = v / lit;
            regs[i] = res;
            if (!(ctx.bypass_write_mask & (1 << i))) ctx.res_ptrs[i][idx[0]] = res;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(perform_op_literal_second_jit)

    inline void filter_join_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            const int* col = (const int*)__builtin_assume_aligned(GET_COL_PTR(ctx, i), 64);
            const bool* ht = GET_HT_PTR(ctx, i);
            unsigned int val = (unsigned int)((r_rd >= 0) ? regs[r_rd] : col[idx[0]]);
            unsigned int min_val = (unsigned int)ctx.params[i*8+3];
            unsigned int v = (min_val == 0) ? val : (val - min_val);
            pass = (v < (unsigned int)ctx.params[i*8+5]) && ht[v];
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(filter_join_jit)

    inline void full_join_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            int val = (r_rd >= 0) ? regs[r_rd] : ctx.col_ptrs[i][idx[0]];
            int min_val = ctx.params[i*8+3];
            int v = (min_val == 0) ? val : (val - min_val);
            if (v >= 0 && v < ctx.params[i*8+5]) {
                int res = ctx.ht_int_ptrs[i][v];
                regs[i] = res;
                if (!(ctx.bypass_write_mask & (1 << i))) ctx.res_ptrs[i][idx[0]] = res;
            } else pass = false;
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(full_join_jit)

    inline void aggregate_jit_impl(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, int i, bool& pass, uint64_t& acc) {
        if (pass) {
            int r_rd = ctx.params[i*8+6];
            const int* col = (const int*)__builtin_assume_aligned(GET_COL_PTR(ctx, i), 64);
            acc += (uint64_t)((r_rd >= 0) ? regs[r_rd] : col[idx[0]]);
        }
    }
    JIT_IMPL_ALL_SLOTS_INLINE(aggregate_jit)

    SYCL_EXTERNAL inline void execute_sycldb_ops_agg(sycl::item<1> idx, const SYCLDBContext& ctx, int* regs, bool& pass, uint64_t& acc) { }
}

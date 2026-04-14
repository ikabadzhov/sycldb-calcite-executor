#include "jit_kernels_impl.hpp"

extern "C" {
    // A dummy kernel to force the compiler to include these functions in device code
    void force_emit_kernels(sycl::queue& q) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::item<1> idx) {
                bool p = false;
                uint64_t a = 0;
                SYCLDBContext c;
                selection_literal_jit_0(idx, c, p, a);
                selection_columns_jit_0(idx, c, p, a);
                perform_op_columns_jit_0(idx, c, p, a);
                filter_join_jit_0(idx, c, p, a);
                full_join_jit_0(idx, c, p, a);
                aggregate_jit_0(idx, c, p, a);
            });
        });
    }
}

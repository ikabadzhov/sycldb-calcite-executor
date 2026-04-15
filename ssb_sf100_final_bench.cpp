#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hipSYCL/sycl/jit.hpp>

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL [[sycl::external]]
#endif

namespace acpp_jit = sycl::AdaptiveCpp_jit;

const size_t N_FACT = 600043265;

struct SSBContext {
    const uint32_t *lo_date, *lo_qty, *lo_price, *lo_disc;
    uint32_t offset;
};

extern "C" void execute_ssb_ops(sycl::id<1> idx, SSBContext ctx, bool& pass, long long& acc);

extern "C" {
    SYCL_EXTERNAL void execute_single_row(sycl::id<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        size_t i = idx[0] + ctx.offset;
        uint32_t d = ctx.lo_date[i];
        uint32_t qt = ctx.lo_qty[i];
        uint32_t di = ctx.lo_disc[i];
        if (d >= 19930101 && d <= 19931231 && qt < 25 && di >= 1 && di <= 3) {
            acc = (long long)ctx.lo_price[i] * di;
        } else {
            pass = false;
        }
    }
}

template<typename T>
T* map_col(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) return nullptr;
    struct stat sb; fstat(fd, &sb);
    T* addr = (T*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return addr;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Starting Reference Segmented Benchmark..." << std::endl;

    std::string b = "/media/ssb/s100_columnar/";
    auto mc = [&](const std::string& name, size_t size) {
        uint32_t* h = (uint32_t*)map_col<int32_t>(b+name);
        if (!h) { std::cerr << "Fail: " << name << std::endl; exit(1); }
        uint32_t* d = sycl::malloc_device<uint32_t>(size, q);
        q.memcpy(d, h, size*4).wait();
        return d;
    };

    SSBContext ctx;
    ctx.lo_date = mc("LINEORDER5", N_FACT);
    ctx.lo_qty = mc("LINEORDER8", N_FACT);
    ctx.lo_price = mc("LINEORDER9", N_FACT);
    ctx.lo_disc = mc("LINEORDER11", N_FACT);

    uint64_t *acc_res = sycl::malloc_shared<uint64_t>(1, q);

    using JOp = acpp_jit::dynamic_function_definition<void, sycl::id<1>, SSBContext, bool&, long long&>;
    std::vector<JOp> ops = { JOp{execute_single_row} };
    acpp_jit::dynamic_function_config cfg;
    cfg.define_as_call_sequence(&execute_ssb_ops, ops);

    std::vector<uint64_t> segs = { 256*1024*1024, 256*1024*1024, N_FACT - 512*1024*1024 };

    for(int r=0; r<3; ++r) {
        *acc_res = 0;
        uint64_t offset = 0;
        printf("Repetition %d\n", r+1);
        for(int s=0; s<3; ++s) {
            ctx.offset = offset;
            auto t0 = std::chrono::high_resolution_clock::now();
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(segs[s]), sycl::reduction(acc_res, sycl::plus<uint64_t>()), 
                    cfg.apply([=](sycl::id<1> idx, auto& red) {
                        bool pass = true;
                        long long acc = 0;
                        execute_ssb_ops(idx, ctx, pass, acc);
                        red += acc;
                    }));
            }).wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            printf("  Segment %d: %.4f ms\n", s, std::chrono::duration<double, std::milli>(t1-t0).count());
            offset += segs[s];
        }
    }
    std::cout << "Result: " << *acc_res << std::endl;
    return 0;
}

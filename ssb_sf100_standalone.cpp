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
#define SYCL_EXTERNAL
#endif

namespace acpp_jit = sycl::AdaptiveCpp_jit;

const size_t N_FACT = 600043265;
const size_t N_CUST = 3000000;
const size_t N_PART = 1400000;
const size_t N_SUPP = 200000;

struct SSBContext {
    const uint32_t *lo_date, *lo_qty, *lo_price, *lo_disc, *lo_pk, *lo_sk, *lo_ck, *lo_rev, *lo_cost;
    const uint8_t *p_filter, *s_filter, *c_filter;
    uint32_t d_min, d_max;
    uint32_t disc_min, disc_max;
    uint32_t qty_min, qty_max;
    uint32_t y_min, y_max;
};

// JIT anchor
extern "C" void execute_ssb_ops(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc);

// Global JIT functions
extern "C" {
    SYCL_EXTERNAL void filter_q1(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        size_t i = idx.get_id(0);
        uint32_t d = ctx.lo_date[i], di = ctx.lo_disc[i], qt = ctx.lo_qty[i];
        pass &= (d >= ctx.d_min && d <= ctx.d_max && di >= ctx.disc_min && di <= ctx.disc_max && qt >= ctx.qty_min && qt <= ctx.qty_max);
    }
    SYCL_EXTERNAL void join_part(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(!pass) return;
        uint32_t pk = ctx.lo_pk[idx.get_id(0)];
        pass &= (pk <= N_PART && ctx.p_filter[pk]);
    }
    SYCL_EXTERNAL void join_supp(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(!pass) return;
        uint32_t sk = ctx.lo_sk[idx.get_id(0)];
        pass &= (sk <= N_SUPP && ctx.s_filter[sk]);
    }
    SYCL_EXTERNAL void agg_revenue_q1(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(pass) acc += (long long)ctx.lo_price[idx.get_id(0)] * ctx.lo_disc[idx.get_id(0)];
    }
    SYCL_EXTERNAL void agg_revenue_simple(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(pass) acc += (long long)ctx.lo_rev[idx.get_id(0)];
    }
}

template<typename T>
T* map_col(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) { std::cerr << "Failed: " << path << std::endl; exit(1); }
    struct stat sb; fstat(fd, &sb);
    T* addr = (T*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return addr;
}

struct State {
    int32_t *c_reg, *p_cat, *s_reg;
    uint32_t *lo_d, *lo_q, *lo_p, *lo_di, *lo_pk, *lo_sk, *lo_ck, *lo_r, *lo_co;
    State(sycl::queue& q) {
        std::string b = "/media/ssb/s100_columnar/";
        auto mc = [&](const std::string& name, size_t size) {
            uint32_t* h = (uint32_t*)map_col<int32_t>(b+name);
            uint32_t* d = sycl::malloc_device<uint32_t>(size, q);
            q.memcpy(d, h, size*4).wait();
            return d;
        };
        lo_d = mc("LINEORDER5", N_FACT); lo_pk = mc("LINEORDER3", N_FACT);
        lo_sk = mc("LINEORDER4", N_FACT); lo_ck = mc("LINEORDER2", N_FACT);
        lo_q = mc("LINEORDER8", N_FACT); lo_p = mc("LINEORDER9", N_FACT);
        lo_di = mc("LINEORDER11", N_FACT); lo_r = mc("LINEORDER12", N_FACT);
        lo_co = mc("LINEORDER13", N_FACT);
        c_reg = (int32_t*)mc("CUSTOMER5", N_CUST);
        p_cat = (int32_t*)mc("PART3", N_PART);
        s_reg = (int32_t*)mc("SUPPLIER5", N_SUPP);
    }
};

typedef acpp_jit::dynamic_function_definition<void, sycl::item<1>, SSBContext, bool&, long long&> df_def;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Benchmarking Standalone SF100 JIT..." << std::endl;
    State s(q);

    uint8_t *pf = sycl::malloc_device<uint8_t>(N_PART+1, q);
    uint8_t *sf = sycl::malloc_device<uint8_t>(N_SUPP+1, q);
    uint8_t *h_pf = (uint8_t*)malloc(N_PART+1);
    uint8_t *h_sf = (uint8_t*)malloc(N_SUPP+1);

    auto benchmark = [&](std::string name, std::vector<df_def> ops, SSBContext& ctx) {
        long long *d_res = sycl::malloc_shared<long long>(1, q);
        acpp_jit::dynamic_function_config cfg;
        cfg.define_as_call_sequence(&execute_ssb_ops, ops);
        auto run = [&]() {
            q.parallel_for(sycl::range<1>{N_FACT}, sycl::reduction(d_res, sycl::plus<>()), cfg.apply([=](sycl::item<1> idx, auto& acc) {
                bool pass = true; long long l_acc = 0;
                execute_ssb_ops(idx, ctx, pass, l_acc); acc += l_acc;
            })).wait();
        };
        run(); run(); // warm
        auto t0 = std::chrono::high_resolution_clock::now();
        run();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << name << "," << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
        sycl::free(d_res, q);
    };

    SSBContext ctx;
    ctx.lo_date = s.lo_d; ctx.lo_pk = s.lo_pk; ctx.lo_sk = s.lo_sk; ctx.lo_ck = s.lo_ck;
    ctx.lo_qty = s.lo_q; ctx.lo_price = s.lo_p; ctx.lo_disc = s.lo_di; ctx.lo_rev = s.lo_r; ctx.lo_cost = s.lo_co;
    ctx.p_filter = pf; ctx.s_filter = sf;

    // Q1.1
    ctx.d_min = 19930101; ctx.d_max = 19931231; ctx.disc_min = 1; ctx.disc_max = 3; ctx.qty_min = 0; ctx.qty_max = 25;
    benchmark("Q1.1", {df_def{filter_q1}, df_def{agg_revenue_q1}}, ctx);

    // Q2.1
    int32_t *h_p_cat = (int32_t*)map_col<int32_t>("/media/ssb/s100_columnar/PART3");
    int32_t *h_s_reg = (int32_t*)map_col<int32_t>("/media/ssb/s100_columnar/SUPPLIER5");
    memset(h_pf, 0, N_PART+1); memset(h_sf, 0, N_SUPP+1);
    for(size_t i=0; i<N_PART; ++i) if(h_p_cat[i] == 12) h_pf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 1) h_sf[i+1] = 1;
    q.memcpy(pf, h_pf, N_PART+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    benchmark("Q2.1", {df_def{join_part}, df_def{join_supp}, df_def{agg_revenue_simple}}, ctx);

    return 0;
}

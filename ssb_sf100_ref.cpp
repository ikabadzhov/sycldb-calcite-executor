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

extern "C" void execute_ssb_ops(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc);

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
    SYCL_EXTERNAL void join_cust(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(!pass) return;
        uint32_t ck = ctx.lo_ck[idx.get_id(0)];
        pass &= (ck <= N_CUST && ctx.c_filter[ck]);
    }
    SYCL_EXTERNAL void filter_year(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(!pass) return;
        uint32_t yr = ctx.lo_date[idx.get_id(0)] / 10000;
        pass &= (yr >= ctx.y_min && yr <= ctx.y_max);
    }
    SYCL_EXTERNAL void filter_date_exact(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(!pass) return;
        uint32_t d = ctx.lo_date[idx.get_id(0)];
        pass &= (d >= ctx.d_min && d <= ctx.d_max);
    }
    SYCL_EXTERNAL void agg_revenue_q1(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(pass) {
            size_t i = idx.get_id(0);
            acc += (long long)ctx.lo_price[i] * ctx.lo_disc[i];
        }
    }
    SYCL_EXTERNAL void agg_revenue_simple(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(pass) acc += (long long)ctx.lo_rev[idx.get_id(0)];
    }
    SYCL_EXTERNAL void agg_profit(sycl::item<1> idx, SSBContext ctx, bool& pass, long long& acc) {
        if(pass) {
            size_t i = idx.get_id(0);
            acc += (long long)((int)ctx.lo_rev[i] - (int)ctx.lo_cost[i]);
        }
    }
}

template<typename T>
T* map_col(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) exit(1);
    struct stat sb; fstat(fd, &sb);
    T* addr = (T*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return addr;
}

struct State {
    int32_t *c_reg, *c_nat, *c_city, *p_mfgr, *p_cat, *p_brand, *s_reg, *s_nat, *s_city;
    uint32_t *lo_d, *lo_q, *lo_p, *lo_di, *lo_pk, *lo_sk, *lo_ck, *lo_r, *lo_co;
    State(sycl::queue& q) {
        std::string b = "/media/ssb/s100_columnar/";
        auto map_and_copy = [&](const std::string& name, size_t size) {
            uint32_t* h = (uint32_t*)map_col<int32_t>(b+name);
            uint32_t* d = sycl::malloc_device<uint32_t>(size, q);
            q.memcpy(d, h, size*4).wait();
            return d;
        };
        auto map_and_copy_i32 = [&](const std::string& name, size_t size) {
            int32_t* h = map_col<int32_t>(b+name);
            int32_t* d = sycl::malloc_device<int32_t>(size, q);
            q.memcpy(d, h, size*4).wait();
            return d;
        };
        lo_d = map_and_copy("LINEORDER5", N_FACT); lo_pk = map_and_copy("LINEORDER3", N_FACT);
        lo_sk = map_and_copy("LINEORDER4", N_FACT); lo_ck = map_and_copy("LINEORDER2", N_FACT);
        lo_q = map_and_copy("LINEORDER8", N_FACT); lo_p = map_and_copy("LINEORDER9", N_FACT);
        lo_di = map_and_copy("LINEORDER11", N_FACT); lo_r = map_and_copy("LINEORDER12", N_FACT);
        lo_co = map_and_copy("LINEORDER13", N_FACT);
        c_reg = map_and_copy_i32("CUSTOMER5", N_CUST); c_nat = map_and_copy_i32("CUSTOMER4", N_CUST); c_city = map_and_copy_i32("CUSTOMER3", N_CUST);
        p_mfgr = map_and_copy_i32("PART2", N_PART); p_cat = map_and_copy_i32("PART3", N_PART); p_brand = map_and_copy_i32("PART4", N_PART);
        s_reg = map_and_copy_i32("SUPPLIER5", N_SUPP); s_nat = map_and_copy_i32("SUPPLIER4", N_SUPP); s_city = map_and_copy_i32("SUPPLIER3", N_SUPP);
    }
};

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Running SSB JIT GPU on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    State s(q);

    auto benchmark = [&](std::string name, std::vector<acpp_jit::dynamic_function_definition<void, sycl::item<1>, SSBContext, bool&, long long&>> ops, SSBContext& ctx) {
        long long *d_res = sycl::malloc_shared<long long>(1, q);
        acpp_jit::dynamic_function_config cfg;
        cfg.define_as_call_sequence(&execute_ssb_ops, ops);
        auto run = [&]() {
            q.parallel_for(sycl::range<1>{N_FACT}, sycl::reduction(d_res, sycl::plus<>()), cfg.apply([=](sycl::item<1> idx, auto& acc) {
                bool pass = true; long long l_acc = 0;
                execute_ssb_ops(idx, ctx, pass, l_acc); acc += l_acc;
            })).wait();
        };
        *d_res = 0; run(); *d_res = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        run();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << name << "," << std::chrono::duration<double, std::milli>(t1 - t0).count() << ", res=" << *d_res << std::endl;
        sycl::free(d_res, q);
    };

    uint8_t *pf = sycl::malloc_device<uint8_t>(N_PART + 1, q), *sf = sycl::malloc_device<uint8_t>(N_SUPP + 1, q), *cf = sycl::malloc_device<uint8_t>(N_CUST + 1, q);
    uint8_t *h_pf = new uint8_t[N_PART + 1], *h_sf = new uint8_t[N_SUPP + 1], *h_cf = new uint8_t[N_CUST + 1];
    int32_t *h_p_cat = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/PART3");
    int32_t *h_s_reg = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/SUPPLIER5");
    int32_t *h_p_brand = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/PART4");
    int32_t *h_c_reg = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/CUSTOMER5");
    int32_t *h_c_nat = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/CUSTOMER4");
    int32_t *h_s_nat = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/SUPPLIER4");
    int32_t *h_c_city = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/CUSTOMER3");
    int32_t *h_s_city = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/SUPPLIER3");
    int32_t *h_p_mfgr = (int32_t*)map_col<int32_t>("/media/ssb/s20_columnar/PART2");

    auto reset_filters = [&](){ std::fill(h_pf, h_pf+N_PART+1, 0); std::fill(h_sf, h_sf+N_SUPP+1, 0); std::fill(h_cf, h_cf+N_CUST+1, 0); };

    // Q1.x
    SSBContext c11{s.lo_d, s.lo_q, s.lo_p, s.lo_di, 0, 0, 0, 0, 0, 0, 0, 0, 19930101, 19931231, 1, 3, 0, 24, 0, 0};
    benchmark("Q1.1", {{&filter_q1}, {&agg_revenue_q1}}, c11);
    SSBContext c12{s.lo_d, s.lo_q, s.lo_p, s.lo_di, 0, 0, 0, 0, 0, 0, 0, 0, 19940101, 19940131, 4, 6, 26, 35, 0, 0};
    benchmark("Q1.2", {{&filter_q1}, {&agg_revenue_q1}}, c12);
    SSBContext c13{s.lo_d, s.lo_q, s.lo_p, s.lo_di, 0, 0, 0, 0, 0, 0, 0, 0, 19940206, 19940212, 5, 7, 26, 35, 0, 0};
    benchmark("Q1.3", {{&filter_q1}, {&agg_revenue_q1}}, c13);

    // Q2.x
    reset_filters();
    for(size_t i=0; i<N_PART; ++i) if(h_p_cat[i] == 1) h_pf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 1) h_sf[i+1] = 1;
    q.memcpy(pf, h_pf, N_PART+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    SSBContext ctx21{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, 0, s.lo_r, 0, pf, sf, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    benchmark("Q2.1", {{&join_part}, {&join_supp}, {&agg_revenue_simple}}, ctx21);

    reset_filters();
    for(size_t i=0; i<N_PART; ++i) if(h_p_brand[i] >= 260 && h_p_brand[i] <= 267) h_pf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 2) h_sf[i+1] = 1;
    q.memcpy(pf, h_pf, N_PART+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    SSBContext ctx22{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, 0, s.lo_r, 0, pf, sf, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    benchmark("Q2.2", {{&join_part}, {&join_supp}, {&agg_revenue_simple}}, ctx22);

    reset_filters();
    for(size_t i=0; i<N_PART; ++i) if(h_p_brand[i] == 278) h_pf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 3) h_sf[i+1] = 1;
    q.memcpy(pf, h_pf, N_PART+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    SSBContext ctx23{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, 0, s.lo_r, 0, pf, sf, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    benchmark("Q2.3", {{&join_part}, {&join_supp}, {&agg_revenue_simple}}, ctx23);

    // Q3.x
    reset_filters();
    for(size_t i=0; i<N_CUST; ++i) if(h_c_reg[i] == 2) h_cf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 2) h_sf[i+1] = 1;
    q.memcpy(cf, h_cf, N_CUST+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    SSBContext ctx31{s.lo_d, 0, 0, 0, 0, s.lo_sk, s.lo_ck, s.lo_r, 0, 0, sf, cf, 0, 0, 0, 0, 0, 0, 1992, 1997};
    benchmark("Q3.1", {{&filter_year}, {&join_cust}, {&join_supp}, {&agg_revenue_simple}}, ctx31);

    reset_filters();
    for(size_t i=0; i<N_CUST; ++i) if(h_c_nat[i] == 24) h_cf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_nat[i] == 24) h_sf[i+1] = 1;
    q.memcpy(cf, h_cf, N_CUST+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    benchmark("Q3.2", {{&filter_year}, {&join_cust}, {&join_supp}, {&agg_revenue_simple}}, ctx31);

    reset_filters();
    for(size_t i=0; i<N_CUST; ++i) if(h_c_city[i] == 201 || h_c_city[i] == 205) h_cf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_city[i] == 201 || h_s_city[i] == 205) h_sf[i+1] = 1;
    q.memcpy(cf, h_cf, N_CUST+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    benchmark("Q3.3", {{&filter_year}, {&join_cust}, {&join_supp}, {&agg_revenue_simple}}, ctx31);

    reset_filters();
    for(size_t i=0; i<N_CUST; ++i) if(h_c_city[i] == 201 || h_c_city[i] == 205) h_cf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_city[i] == 205) h_sf[i+1] = 1;
    q.memcpy(cf, h_cf, N_CUST+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait();
    SSBContext ctx34{s.lo_d, 0, 0, 0, 0, s.lo_sk, s.lo_ck, s.lo_r, 0, 0, sf, cf, 19971201, 19971231, 0, 0, 0, 0, 0, 0};
    benchmark("Q3.4", {{&filter_date_exact}, {&join_cust}, {&join_supp}, {&agg_revenue_simple}}, ctx34);

    // Q4.x
    reset_filters();
    for(size_t i=0; i<N_CUST; ++i) if(h_c_reg[i] == 1) h_cf[i+1] = 1;
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_reg[i] == 1) h_sf[i+1] = 1;
    for(size_t i=0; i<N_PART; ++i) if(h_p_mfgr[i] == 0 || h_p_mfgr[i] == 1) h_pf[i+1] = 1;
    q.memcpy(cf, h_cf, N_CUST+1).wait(); q.memcpy(sf, h_sf, N_SUPP+1).wait(); q.memcpy(pf, h_pf, N_PART+1).wait();
    SSBContext ctx41{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, s.lo_ck, s.lo_r, s.lo_co, pf, sf, cf, 0, 0, 0, 0, 0, 0, 1992, 1998};
    benchmark("Q4.1", {{&filter_year}, {&join_cust}, {&join_supp}, {&join_part}, {&agg_profit}}, ctx41);

    SSBContext ctx42{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, s.lo_ck, s.lo_r, s.lo_co, pf, sf, cf, 0, 0, 0, 0, 0, 0, 1997, 1998};
    benchmark("Q4.2", {{&filter_year}, {&join_cust}, {&join_supp}, {&join_part}, {&agg_profit}}, ctx42);

    reset_filters();
    for(size_t i=0; i<N_SUPP; ++i) if(h_s_nat[i] == 24) h_sf[i+1] = 1;
    for(size_t i=0; i<N_PART; ++i) if(h_p_cat[i] == 3) h_pf[i+1] = 1;
    q.memcpy(sf, h_sf, N_SUPP+1).wait(); q.memcpy(pf, h_pf, N_PART+1).wait();
    SSBContext ctx43{s.lo_d, 0, 0, 0, s.lo_pk, s.lo_sk, 0, s.lo_r, s.lo_co, pf, sf, 0, 0, 0, 0, 0, 0, 0, 1997, 1997};
    benchmark("Q4.3", {{&filter_year}, {&join_supp}, {&join_part}, {&agg_profit}}, ctx43);

    return 0;
}

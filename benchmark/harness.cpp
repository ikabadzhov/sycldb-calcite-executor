#include <sycl/sycl.hpp>
#include <AdaptiveCpp/sycl/jit.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// --- SHARED STATE ---
struct QueryState {
    bool pass;
};

// --- JIT PLACEHOLDERS ---
extern "C" {
    void execute_q11(sycl::item<1> idx, QueryState& state, 
                           int* lo_discount, int* lo_quantity, int* lo_orderdate, 
                           int* lo_extendedprice, bool* date_mask, 
                           int d_min, int d_max, uint64_t* out) { }

    void execute_q21(sycl::item<1> idx, QueryState& state, 
                           int* lo_partkey, int* lo_suppkey, int* lo_revenue,
                           bool* p_mask, bool* s_mask, size_t mask_size, uint64_t* out) { }
}

// --- KERNEL PRIMITIVES ---
extern "C" {
    void p11_discount(sycl::item<1> idx, QueryState& state, int* lo_discount, int*, int*, int*, bool*, int, int, uint64_t*) {
        state.pass = (lo_discount[idx.get_id(0)] >= 1 && lo_discount[idx.get_id(0)] <= 3);
    }
    void p11_quantity(sycl::item<1> idx, QueryState& state, int*, int* lo_quantity, int*, int*, bool*, int, int, uint64_t*) {
        if(state.pass) state.pass = (lo_quantity[idx.get_id(0)] < 25);
    }
    void p11_date(sycl::item<1> idx, QueryState& state, int*, int*, int* lo_orderdate, int*, bool* mask, int d_min, int d_max, uint64_t*) {
        if(state.pass) {
            int d = lo_orderdate[idx.get_id(0)];
            state.pass = (d >= d_min && d <= d_max && mask[d - d_min]);
        }
    }
    void p11_aggr(sycl::item<1> idx, QueryState& state, int* lo_discount, int*, int*, int* lo_extendedprice, bool*, int, int, uint64_t* out) {
        if(state.pass) {
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_out(*out);
            atomic_out += (uint64_t)lo_extendedprice[idx.get_id(0)] * (uint64_t)lo_discount[idx.get_id(0)];
        }
    }

    void p21_part(sycl::item<1> idx, QueryState& state, int* lo_partkey, int*, int*, bool* p_mask, bool*, size_t mask_size, uint64_t*) {
        state.pass = (lo_partkey[idx.get_id(0)] < (int)mask_size) ? p_mask[lo_partkey[idx.get_id(0)]] : false;
    }
    void p21_supp(sycl::item<1> idx, QueryState& state, int*, int* lo_suppkey, int*, bool*, bool* s_mask, size_t mask_size, uint64_t*) {
        if(state.pass) state.pass = (lo_suppkey[idx.get_id(0)] < (int)mask_size) ? s_mask[lo_suppkey[idx.get_id(0)]] : false;
    }
    void p21_aggr(sycl::item<1> idx, QueryState& state, int*, int*, int* lo_revenue, bool*, bool*, size_t, uint64_t* out) {
        if(state.pass) {
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_out(*out);
            atomic_out += (uint64_t)lo_revenue[idx.get_id(0)];
        }
    }
}

size_t get_rows(const std::string& path, size_t elem_size) {
    if (!fs::exists(path)) return 0;
    return fs::file_size(path) / elem_size;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    int dev_idx = (argc > 1) ? std::stoi(argv[1]) : 0;
    if (dev_idx >= (int)devices.size()) return 1;
    sycl::queue q(devices[dev_idx]);
    std::string device_name = devices[dev_idx].get_info<sycl::info::device::name>();
    
    std::string data_root = std::getenv("SSB_PATH") ? std::getenv("SSB_PATH") : "/media/ssb/s40_columnar/";
    if (data_root.back() != '/') data_root += "/";
    size_t rows = get_rows(data_root + "LINEORDER5", 4);
    if (rows == 0) return 1;

    int *lo_orderdate = sycl::malloc_device<int>(rows, q), *lo_partkey = sycl::malloc_device<int>(rows, q),
        *lo_suppkey = sycl::malloc_device<int>(rows, q), *lo_quantity = sycl::malloc_device<int>(rows, q),
        *lo_extendedprice = sycl::malloc_device<int>(rows, q), *lo_discount = sycl::malloc_device<int>(rows, q),
        *lo_revenue = sycl::malloc_device<int>(rows, q);
    bool *d_flags = sycl::malloc_device<bool>(rows, q);
    uint64_t *d_out = sycl::malloc_device<uint64_t>(1, q);

    auto load = [&](const std::string& n, int* d) {
        std::ifstream is(data_root + n, std::ios::binary); if(!is) return;
        std::vector<int> b(10000000);
        for(size_t i=0; i<rows; i+=b.size()) { size_t c = std::min(b.size(), rows-i); is.read((char*)b.data(), c*4); q.memcpy(d+i, b.data(), c*4).wait(); }
    };
    load("LINEORDER5", lo_orderdate); load("LINEORDER3", lo_partkey); load("LINEORDER4", lo_suppkey);
    load("LINEORDER8", lo_quantity); load("LINEORDER9", lo_extendedprice); load("LINEORDER11", lo_discount); load("LINEORDER12", lo_revenue);

    size_t m_sz = 700000000;
    bool *d_pm = sycl::malloc_device<bool>(m_sz, q), *d_sm = sycl::malloc_device<bool>(m_sz, q);
    q.fill(d_pm, true, m_sz); q.fill(d_sm, true, m_sz);

    size_t dr = 2556; std::vector<int> hk(dr), hy(dr);
    { std::ifstream is(data_root+"DDATE0", std::ios::binary); is.read((char*)hk.data(), dr*4);
      std::ifstream is2(data_root+"DDATE4", std::ios::binary); is2.read((char*)hy.data(), dr*4); }
    int d_min=99999999, d_max=0; for(int i=0; i<dr; ++i) { if(hk[i]<d_min)d_min=hk[i]; if(hk[i]>d_max)d_max=hk[i]; }
    bool* d_dm = sycl::malloc_device<bool>(d_max-d_min+1, q);
    { bool* hm = new bool[d_max-d_min+1]; std::fill(hm, hm+d_max-d_min+1, false);
      for(int i=0; i<dr; ++i) if(hy[i]==1993) hm[hk[i]-d_min]=true; q.memcpy(d_dm, hm, d_max-d_min+1).wait(); delete[] hm; }

    int reps = (argc > 4) ? std::stoi(argv[4]) : 20;

    sycl::jit::dynamic_function_config cfg_q11, cfg_q21;
    
    // Simplest JIT call sequence definition
    cfg_q11.define_as_call_sequence(&execute_q11, {&p11_discount, &p11_quantity, &p11_date, &p11_aggr});
    cfg_q21.define_as_call_sequence(&execute_q21, {&p21_part, &p21_supp, &p21_aggr});

    auto run_q11_unfused = [&]() {
        q.fill(d_out, 0ULL, 1).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { QueryState s; p11_discount(idx, s, lo_discount, 0,0,0,0,0,0,0); d_flags[idx.get_id(0)] = s.pass; }).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { if(d_flags[idx.get_id(0)]) { QueryState s; s.pass=true; p11_quantity(idx, s, 0, lo_quantity, 0,0,0,0,0,0); d_flags[idx.get_id(0)] = s.pass; } }).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { if(d_flags[idx.get_id(0)]) { QueryState s; s.pass=true; p11_date(idx, s, 0,0, lo_orderdate, 0, d_dm, d_min, d_max, 0); d_flags[idx.get_id(0)] = s.pass; } }).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { if(d_flags[idx.get_id(0)]) { QueryState s; s.pass=true; p11_aggr(idx, s, lo_discount, 0,0, lo_extendedprice, 0,0,0, d_out); } }).wait();
    };

    auto run_q11_fused = [&]() {
        q.fill(d_out, 0ULL, 1).wait();
        q.parallel_for(sycl::range<1>(rows), cfg_q11.apply([=](sycl::item<1> idx) {
            QueryState s; s.pass = true;
            execute_q11(idx, s, lo_discount, lo_quantity, lo_orderdate, lo_extendedprice, d_dm, d_min, d_max, d_out);
        })).wait();
    };

    auto run_q21_unfused = [&]() {
        q.fill(d_out, 0ULL, 1).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { QueryState s; p21_part(idx, s, lo_partkey, 0,0, d_pm, 0, m_sz, 0); d_flags[idx.get_id(0)] = s.pass; }).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { if(d_flags[idx.get_id(0)]) { QueryState s; s.pass=true; p21_supp(idx, s, 0, lo_suppkey, 0,0, d_sm, m_sz, 0); d_flags[idx.get_id(0)] = s.pass; } }).wait();
        q.parallel_for(rows, [=](sycl::item<1> idx) { if(d_flags[idx.get_id(0)]) { QueryState s; s.pass=true; p21_aggr(idx, s, 0,0, lo_revenue, 0,0,0, d_out); } }).wait();
    };

    auto run_q21_fused = [&]() {
        q.fill(d_out, 0ULL, 1).wait();
        q.parallel_for(sycl::range<1>(rows), cfg_q21.apply([=](sycl::item<1> idx) {
            QueryState s; s.pass = true;
            execute_q21(idx, s, lo_partkey, lo_suppkey, lo_revenue, d_pm, d_sm, m_sz, d_out);
        })).wait();
    };

    auto run_hardcoded = [&](const std::string& qy) {
        q.fill(d_out, 0ULL, 1).wait();
        if(qy=="Q1.1") q.parallel_for(rows, [=](sycl::id<1> i) {
            if (lo_discount[i]>=1 && lo_discount[i]<=3 && lo_quantity[i]<25) {
                int d = lo_orderdate[i]; if(d>=d_min && d<=d_max && d_dm[d-d_min]) {
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ao(*d_out);
                    ao += (uint64_t)lo_extendedprice[i] * (uint64_t)lo_discount[i];
                }
            }
        }).wait();
        else q.parallel_for(rows, [=](sycl::id<1> i) {
            if (lo_partkey[i]<(int)m_sz && lo_suppkey[i]<(int)m_sz && d_pm[lo_partkey[i]] && d_sm[lo_suppkey[i]]) {
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ao(*d_out); ao += (uint64_t)lo_revenue[i];
            }
        }).wait();
    };

    std::vector<std::string> queries = {"Q1.1", "Q2.1"};
    for(auto query_str : queries) {
        if (argc > 2 && std::string(argv[2]) != "ALL" && std::string(argv[2]) != query_str) continue;
        for(auto mode : {"unfused", "fused", "hardcoded"}) {
            if (argc > 3 && std::string(argv[3]) != "ALL" && std::string(argv[3]) != mode) continue;
            for(int i=0; i<2; ++i) { 
                if(query_str=="Q1.1") { if(std::string(mode)=="unfused") run_q11_unfused(); else if(std::string(mode)=="fused") run_q11_fused(); else run_hardcoded("Q1.1"); }
                else { if(std::string(mode)=="unfused") run_q21_unfused(); else if(std::string(mode)=="fused") run_q21_fused(); else run_hardcoded("Q2.1"); }
            }
            q.wait();
            for(int i=0; i<reps; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                if(query_str=="Q1.1") { if(std::string(mode)=="unfused") run_q11_unfused(); else if(std::string(mode)=="fused") run_q11_fused(); else run_hardcoded("Q1.1"); }
                else { if(std::string(mode)=="unfused") run_q21_unfused(); else if(std::string(mode)=="fused") run_q21_fused(); else run_hardcoded("Q2.1"); }
                auto end = std::chrono::high_resolution_clock::now();
                printf("ITER_RESULT,%s,%s,%s,%d,%f\n", device_name.c_str(), query_str.c_str(), mode, i, std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
    }
    return 0;
}

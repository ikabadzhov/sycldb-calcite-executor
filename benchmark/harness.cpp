#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <cstdio>
#include <map>

namespace fs = std::filesystem;

size_t get_rows(const std::string& path, size_t elem_size) {
    if (!fs::exists(path)) return 0;
    return fs::file_size(path) / elem_size;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    int dev_idx = (argc > 1) ? std::stoi(argv[1]) : 0;
    if (dev_idx >= devices.size()) return 1;

    sycl::queue q(devices[dev_idx]);
    std::string device_name = devices[dev_idx].get_info<sycl::info::device::name>();
    
    std::string data_root = std::getenv("SSB_PATH") ? std::getenv("SSB_PATH") : "/media/ssb/s40_columnar/";
    if (data_root.back() != '/') data_root += "/";

    size_t lineorder_rows = get_rows(data_root + "LINEORDER5", 4);
    
    // Allocate LINEORDER columns
    int* lo_orderdate = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_partkey = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_suppkey = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_quantity = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_extendedprice = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_discount = sycl::malloc_device<int>(lineorder_rows, q);
    int* lo_revenue = sycl::malloc_device<int>(lineorder_rows, q);
    bool* d_flags = sycl::malloc_device<bool>(lineorder_rows, q);
    uint64_t* d_revenue_out = sycl::malloc_device<uint64_t>(1, q);

    // Loading Helper
    std::vector<int> h_buf(125000000);
    auto load_chunked = [&](const std::string& name, int* d_ptr) {
        std::ifstream is(data_root + name, std::ios::binary);
        if(!is) return;
        for(size_t i=0; i<lineorder_rows; i+=h_buf.size()) {
            size_t chunk = std::min(h_buf.size(), lineorder_rows - i);
            is.read((char*)h_buf.data(), chunk*4);
            q.memcpy(d_ptr + i, h_buf.data(), chunk*4).wait();
        }
    };
    load_chunked("LINEORDER5", lo_orderdate);
    load_chunked("LINEORDER3", lo_partkey);
    load_chunked("LINEORDER4", lo_suppkey);
    load_chunked("LINEORDER8", lo_quantity);
    load_chunked("LINEORDER9", lo_extendedprice);
    load_chunked("LINEORDER11", lo_discount);
    load_chunked("LINEORDER12", lo_revenue);

    // Masks
    // We found that lo_partkey/suppkey can go up to 600M in some datasets.
    size_t mask_size = 700000000;
    bool* d_p_mask = sycl::malloc_device<bool>(mask_size, q);
    bool* d_s_mask = sycl::malloc_device<bool>(mask_size, q);
    {
        bool* h_m = new bool[mask_size]; std::fill(h_m, h_m+mask_size, false);
        for(size_t i=0; i<mask_size; ++i) if(i % 10 == 0) h_m[i] = true;
        q.memcpy(d_p_mask, h_m, mask_size).wait();
        std::fill(h_m, h_m+mask_size, false);
        for(size_t i=0; i<mask_size; ++i) if(i % 5 == 0) h_m[i] = true;
        q.memcpy(d_s_mask, h_m, mask_size).wait();
        delete[] h_m;
    }

    // Q1.1 Date Mask
    size_t ddate_rows = 2556;
    std::vector<int> h_d_key(ddate_rows), h_d_year(ddate_rows);
    {
        std::ifstream is(data_root + "DDATE0", std::ios::binary); is.read((char*)h_d_key.data(), ddate_rows*4);
        std::ifstream is2(data_root + "DDATE4", std::ios::binary); is2.read((char*)h_d_year.data(), ddate_rows*4);
    }
    int d_min = 99999999, d_max = 0;
    for(int i=0; i<ddate_rows; ++i) { if(h_d_key[i]<d_min)d_min=h_d_key[i]; if(h_d_key[i]>d_max)d_max=h_d_key[i]; }
    bool* d_q11_date_mask = sycl::malloc_device<bool>(d_max - d_min + 1, q);
    {
        bool* h_m1 = new bool[d_max-d_min+1]; std::fill(h_m1, h_m1+d_max-d_min+1, false);
        for(int i=0; i<ddate_rows; ++i) if(h_d_year[i]==1993) h_m1[h_d_key[i]-d_min]=true;
        q.memcpy(d_q11_date_mask, h_m1, d_max-d_min+1).wait(); delete[] h_m1;
    }

    int reps = (argc > 3) ? std::stoi(argv[3]) : 100;

    auto run_q11_unfused = [&]() {
        q.fill<uint64_t>(d_revenue_out, 0, 1).wait();
        auto e1 = q.submit([&](sycl::handler& cgh) { cgh.parallel_for(lineorder_rows, [=](sycl::id<1> i) { d_flags[i] = (lo_discount[i] >= 1 && lo_discount[i] <= 3); }); });
        auto e2 = q.submit([&](sycl::handler& cgh) { cgh.depends_on(e1); cgh.parallel_for(lineorder_rows, [=](sycl::id<1> i) { if (d_flags[i]) d_flags[i] = (lo_quantity[i] < 25); }); });
        auto e3 = q.submit([&](sycl::handler& cgh) { cgh.depends_on(e2); cgh.parallel_for(lineorder_rows, [=](sycl::id<1> i) { if (d_flags[i]) { int d = lo_orderdate[i]; d_flags[i] = (d>=d_min && d<=d_max) ? d_q11_date_mask[d-d_min] : false; } }); });
        q.submit([&](sycl::handler& cgh) { cgh.depends_on(e3); auto red = sycl::reduction(d_revenue_out, sycl::plus<uint64_t>()); cgh.parallel_for(sycl::range<1>(lineorder_rows), red, [=](sycl::id<1> i, auto& sum) { if (d_flags[i]) sum += (uint64_t)lo_extendedprice[i] * (uint64_t)lo_discount[i]; }); }).wait();
    };

    auto run_q11_fused = [&]() {
        q.fill<uint64_t>(d_revenue_out, 0, 1).wait();
        q.submit([&](sycl::handler& cgh) {
            auto red = sycl::reduction(d_revenue_out, sycl::plus<uint64_t>());
            cgh.parallel_for(sycl::range<1>(lineorder_rows), red, [=](sycl::id<1> i, auto& sum) {
                bool pass = (lo_discount[i] >= 1 && lo_discount[i] <= 3);
                if(pass) pass = (lo_quantity[i] < 25);
                if(pass) { int d = lo_orderdate[i]; pass = (d>=d_min && d<=d_max && d_q11_date_mask[d-d_min]); }
                if(pass) sum += (uint64_t)lo_extendedprice[i] * (uint64_t)lo_discount[i];
            });
        }).wait();
    };

    auto run_q21_unfused = [&]() {
        q.fill<uint64_t>(d_revenue_out, 0, 1).wait();
        auto e1 = q.submit([&](sycl::handler& cgh) { cgh.parallel_for(lineorder_rows, [=](sycl::id<1> i) { d_flags[i] = (lo_partkey[i] < mask_size) ? d_p_mask[lo_partkey[i]] : false; }); });
        auto e2 = q.submit([&](sycl::handler& cgh) { cgh.depends_on(e1); cgh.parallel_for(lineorder_rows, [=](sycl::id<1> i) { if (d_flags[i]) d_flags[i] = (lo_suppkey[i] < mask_size) ? d_s_mask[lo_suppkey[i]] : false; }); });
        q.submit([&](sycl::handler& cgh) { cgh.depends_on(e2); auto red = sycl::reduction(d_revenue_out, sycl::plus<uint64_t>()); cgh.parallel_for(sycl::range<1>(lineorder_rows), red, [=](sycl::id<1> i, auto& sum) { if (d_flags[i]) sum += (uint64_t)lo_revenue[i]; }); }).wait();
    };

    auto run_q21_fused = [&]() {
        q.fill<uint64_t>(d_revenue_out, 0, 1).wait();
        q.submit([&](sycl::handler& cgh) {
            auto red = sycl::reduction(d_revenue_out, sycl::plus<uint64_t>());
            cgh.parallel_for(sycl::range<1>(lineorder_rows), red, [=](sycl::id<1> i, auto& sum) {
                if (lo_partkey[i] < mask_size && lo_suppkey[i] < mask_size && d_p_mask[lo_partkey[i]] && d_s_mask[lo_suppkey[i]]) {
                    sum += (uint64_t)lo_revenue[i];
                }
            });
        }).wait();
    };

    for(auto query : {"Q1.1", "Q2.1"}) {
        for(auto mode : {"unfused", "fused", "hardcoded"}) {
            for(int i=0; i<2; ++i) { 
                if(std::string(query)=="Q1.1") { if(std::string(mode)=="unfused") run_q11_unfused(); else run_q11_fused(); }
                else { if(std::string(mode)=="unfused") run_q21_unfused(); else run_q21_fused(); }
            }
            q.wait();
            for(int i=0; i<reps; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                if(std::string(query)=="Q1.1") { if(std::string(mode)=="unfused") run_q11_unfused(); else run_q11_fused(); }
                else { if(std::string(mode)=="unfused") run_q21_unfused(); else run_q21_fused(); }
                auto end = std::chrono::high_resolution_clock::now();
                printf("ITER_RESULT,%s,%s,%s,%d,%f\n", device_name.c_str(), query, mode, i, std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
    }
    return 0;
}

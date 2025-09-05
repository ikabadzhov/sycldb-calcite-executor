#pragma once

#include <sycl/sycl.hpp>

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
void build_keys_ht(T col[], bool flags[], int col_len, bool ht[], T ht_len, T ht_min_value, sycl::queue &queue)
{
    // Build the hash table with flags
    //for (int i = 0; i < col_len; i++)
    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        ht[HASH(col[i], ht_len, ht_min_value)] = flags[i];
    });
}

void build_key_vals_ht(int col[], int agg_col[], bool flags[], int col_len, int ht[], int ht_len, int ht_min_value, sycl::queue &queue)
{
    //for (int i = 0; i < col_len; i++)
    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            int hash = HASH(col[i], ht_len, ht_min_value);
            ht[hash << 1] = 1;
            ht[(hash << 1) + 1] = agg_col[i];
        }
        else
        {
            ht[HASH(col[i], ht_len, ht_min_value) << 1] = 0;
        }
    });
}

template <typename T>
void filter_join(T build_col[],
                 bool build_flags[],
                 int build_col_len,
                 T build_max_value,
                 T build_min_value,
                 T probe_col[],
                 bool probe_col_flags[],
                 int probe_col_len, sycl::queue &queue)
{
    queue.wait();
    int ht_len = build_max_value - build_min_value + 1;
    //bool *ht = new bool[ht_len];
    //std::fill_n(ht, ht_len, false);
    bool *ht = sycl::malloc_device<bool>(ht_len, queue);
    queue.memset(ht, 0, sizeof(bool) * ht_len).wait();
    
    build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue);
    bool *h_bf = sycl::malloc_host<bool>(build_col_len, queue);
    queue.memcpy(h_bf, build_flags, sizeof(bool) * build_col_len).wait();
    int flags_true = std::count(h_bf, h_bf + build_col_len, true);
    std::cout << "build flags true: " << flags_true << std::endl;
    sycl::free(h_bf, queue);
    queue.wait();
    //for (int i = 0; i < probe_col_len; i++)
    int *flags_probe_h = sycl::malloc_host<int>(probe_col_len, queue);
    queue.memcpy(flags_probe_h, probe_col_flags, sizeof(bool) * probe_col_len).wait();
    int flags_probe_true = std::count(flags_probe_h, flags_probe_h + probe_col_len, true);
    std::cout << "probe flags true: " << flags_probe_true << std::endl;
    queue.parallel_for(probe_col_len, [=](sycl::id<1> i) {
        if (probe_col_flags[i])
            probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];
    });
    queue.memcpy(flags_probe_h, probe_col_flags, sizeof(bool) * probe_col_len).wait();
    flags_probe_true = std::count(flags_probe_h, flags_probe_h + probe_col_len, true);
    std::cout << "probe flags after filter: " << flags_probe_true << std::endl;
    sycl::free(flags_probe_h, queue);
    queue.wait();
    //delete[] ht;
    sycl::free(ht, queue);
}

void full_join(TableData<int> &probe_table,
               TableData<int> &build_table,
               int probe_col_index,
               int build_col_index, sycl::queue &queue)
{
    queue.wait();
    int ht_len = build_table.columns[build_table.column_indices.at(build_col_index)].max_value -
                build_table.columns[build_table.column_indices.at(build_col_index)].min_value + 1;
    //int *ht = new int[ht_len * 2];
    int *ht = sycl::malloc_device<int>(ht_len * 2, queue);
    queue.memset(ht, 0, sizeof(int) * ht_len * 2).wait();
    int build_column = build_table.column_indices.at(build_col_index);
    int probe_column = probe_table.column_indices.at(probe_col_index);

    //std::fill_n(ht, ht_len * 2, 0); // maybe it's not necessary to initialize

    bool *build_flags_h = sycl::malloc_host<bool>(build_table.col_len, queue);
    queue.memcpy(build_flags_h, build_table.flags, sizeof(bool) * build_table.col_len).wait();
    int flags_true = std::count(build_flags_h, build_flags_h + build_table.col_len, true);
    std::cout << "build flags true: " << flags_true << std::endl;
    sycl::free(build_flags_h, queue);
    
    build_key_vals_ht(
        build_table.columns[build_column].content,
        build_table.columns[build_table.column_indices.at(build_table.group_by_column)].content,
        build_table.flags, build_table.col_len, ht, ht_len,
        build_table.columns[build_column].min_value, queue);
    queue.wait();
    //for (int i = 0; i < probe_table.col_len; i++)
    auto& flags = probe_table.flags;
    auto& content = probe_table.columns[probe_column].content;
    auto& min_value = build_table.columns[build_column].min_value;

    int *ht_h = sycl::malloc_host<int>(ht_len * 2, queue);
    queue.memcpy(ht_h, ht, sizeof(int) * ht_len * 2).wait();
    int ht_true = std::count_if(ht_h, ht_h + ht_len * 2, [](int val) { return val % 2 == 0; });
    std::cout << "ht true K: " << ht_true << std::endl;
    int ht_true2 = std::count_if(ht_h, ht_h + ht_len * 2, [](int val) { return val % 2 == 1; });
    std::cout << "ht true V: " << ht_true2 << std::endl;
    sycl::free(ht_h, queue);

    bool *probe_flags_h = sycl::malloc_host<bool>(probe_table.col_len, queue);
    queue.memcpy(probe_flags_h, flags, sizeof(bool) * probe_table.col_len).wait();
    int probe_flags_true = std::count(probe_flags_h, probe_flags_h + probe_table.col_len, true);
    std::cout << "probe flags true before: " << probe_flags_true << std::endl;

    queue.parallel_for(probe_table.col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            int hash = HASH(content[i], ht_len, min_value);
            if (ht[hash << 1] == 1)
                content[i] = ht[(hash << 1) + 1]; // replace the probe column value with the value to group by on
            else
                flags[i] = false; // mark as not selected
        }
    });
    queue.wait();
    queue.memcpy(probe_flags_h, flags, sizeof(bool) * probe_table.col_len).wait();
    probe_flags_true = std::count(probe_flags_h, probe_flags_h + probe_table.col_len, true);
    std::cout << "probe flags true after: " << probe_flags_true << std::endl;
    sycl::free(probe_flags_h, queue);
    // the group by column index must refer to the old foreign key
    probe_table.column_indices.erase(probe_col_index);
    probe_table.column_indices[probe_table.col_number + build_table.group_by_column] = probe_column;

    //delete[] ht;
    sycl::free(ht, queue);
}
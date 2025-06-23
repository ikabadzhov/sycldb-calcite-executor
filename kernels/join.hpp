#pragma once

#include <sycl/sycl.hpp>
#include <chrono>

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
void build_keys_ht(T col[], bool flags[], int col_len, bool ht[], T ht_len, T ht_min_value, sycl::queue &queue)
{
    //for (int i = 0; i < col_len; i++)
    //    ht[HASH(col[i], ht_len, ht_min_value)] = flags[i];
    queue.parallel_for(col_len, [=](sycl::id<1> idx) {
        if (flags[idx])
            ht[HASH(col[idx], ht_len, ht_min_value)] = true;
    });
}

void build_key_vals_ht(int col[], int agg_col[], bool flags[], int col_len, int ht[], int ht_len, int ht_min_value, sycl::queue &queue)
{
    /*for (int i = 0; i < col_len; i++)
    {
        if (flags[i])
        {
            int hash = HASH(col[i], ht_len, ht_min_value);
            ht[hash << 1] = 1;
            ht[(hash << 1) + 1] = agg_col[i];
        }
    }*/
    queue.parallel_for(col_len, [=](sycl::id<1> idx) {
        if (flags[idx])
        {
            int hash = HASH(col[idx], ht_len, ht_min_value);
            ht[hash << 1] = 1;
            ht[(hash << 1) + 1] = agg_col[idx];
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
    int ht_len = build_max_value - build_min_value + 1;
    //bool *ht = new bool[ht_len];
    bool *ht = sycl::malloc_shared<bool>(ht_len, queue);
    queue.prefetch(ht, ht_len * sizeof(bool));
    std::fill_n(ht, ht_len, false);

    queue.wait();
    auto start = std::chrono::high_resolution_clock::now();
    build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue);
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << ">>> BUILD took: " << elapsed.count() * 1000 << " ms.\n";

    //for (int i = 0; i < probe_col_len; i++)
    //    if (probe_col_flags[i])
    //        probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];
    queue.prefetch(probe_col, probe_col_len * sizeof(T));
    queue.prefetch(probe_col_flags, probe_col_len * sizeof(bool));
    queue.wait();
    start = std::chrono::high_resolution_clock::now();
    queue.parallel_for(probe_col_len, [=](sycl::id<1> idx) {
        if (probe_col_flags[idx])
            probe_col_flags[idx] = ht[HASH(probe_col[idx], ht_len, build_min_value)];
    });
    queue.wait();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << ">>> PROBE took: " << elapsed.count() * 1000 << " ms.\n";

    //delete[] ht;
    sycl::free(ht, queue);
}

void full_join(TableData<int> &probe_table,
               TableData<int> &build_table,
               int probe_col_index,
               int build_col_index, sycl::queue &queue)
{
    int ht_len = build_table.columns[build_table.column_indices.at(build_col_index)].max_value -
                build_table.columns[build_table.column_indices.at(build_col_index)].min_value + 1;
    //int *ht = new int[ht_len * 2],
    int *ht = sycl::malloc_shared<int>(ht_len * 2, queue);
    queue.prefetch(ht, ht_len * 2 * sizeof(int));
    int build_column = build_table.column_indices.at(build_col_index);
    int probe_column = probe_table.column_indices.at(probe_col_index);

    std::fill_n(ht, ht_len * 2, 0); // maybe it's not necessary to initialize

    queue.prefetch(build_table.columns[build_column].content, build_table.col_len * sizeof(int));
    queue.prefetch(build_table.columns[build_table.column_indices.at(build_table.group_by_column)].
                    content, build_table.col_len * sizeof(int));
    queue.prefetch(probe_table.columns[probe_column].content, probe_table.col_len * sizeof(int));
    queue.prefetch(probe_table.flags, probe_table.col_len * sizeof(bool));

    queue.wait();
    auto start = std::chrono::high_resolution_clock::now();
    build_key_vals_ht(
        build_table.columns[build_column].content,
        build_table.columns[build_table.column_indices.at(build_table.group_by_column)].content,
        build_table.flags, build_table.col_len, ht, ht_len,
        build_table.columns[build_column].min_value, queue);
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << ">>> BUILD took: " << elapsed.count() * 1000 << " ms.\n";

    /*for (int i = 0; i < probe_table.col_len; i++)
    {
        if (probe_table.flags[i])
        {
            int hash = HASH(probe_table.columns[probe_column].content[i], ht_len,
                            build_table.columns[build_column].min_value);
            if (ht[hash << 1] == 1)
                probe_table.columns[probe_column].content[i] = ht[(hash << 1) + 1]; // replace the probe column value with the value to group by on
            else
                probe_table.flags[i] = false; // mark as not selected
        }
    }*/
    auto flags = probe_table.flags;
    auto content = probe_table.columns[probe_column].content;
    auto min_value = build_table.columns[build_column].min_value;
    queue.wait();
    start = std::chrono::high_resolution_clock::now();
    queue.parallel_for(probe_table.col_len, [=](sycl::id<1> idx) {
        if (flags[idx])
        {
            int hash = HASH(content[idx], ht_len, min_value);
            uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
            if (slot)
                content[idx] = (slot >> 32); // replace the probe column value with the value to group by on
            else
                flags[idx] = false; // mark as not selected*/
        }
    });
    queue.wait();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << ">>> PROBE took: " << elapsed.count() * 1000 << " ms.\n";

    // the group by column index must refer to the old foreign key
    probe_table.column_indices.erase(probe_col_index);
    probe_table.column_indices[probe_table.col_number + build_table.group_by_column] = probe_column;

    //delete[] ht;
    sycl::free(ht, queue);
}
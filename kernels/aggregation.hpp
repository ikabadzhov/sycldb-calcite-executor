#pragma once

#include <string>

enum class BinaryOp : uint8_t {
    Multiply,
    Divide,
    Add,
    Subtract
};

template <typename T>
inline T element_operation(T a, T b, BinaryOp op)
{
    switch (op) {
        case BinaryOp::Multiply: return a * b;
        case BinaryOp::Divide:   return a / b;
        case BinaryOp::Add:      return a + b;
        case BinaryOp::Subtract: return a - b;
        default:                 return 0;
    }
}

BinaryOp get_op_from_string(const std::string &op) {
    if (op == "*") return BinaryOp::Multiply;
    if (op == "/") return BinaryOp::Divide;
    if (op == "+") return BinaryOp::Add;
    if (op == "-") return BinaryOp::Subtract;
    throw std::invalid_argument("Unknown operation: " + op);
}

template <typename T>
void perform_operation(T result[], const T a[], const T b[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    //for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i) {
        if (flags[i])
            result[i] = element_operation(a[i], b[i], op_enum);
    });
}

template <typename T>
void perform_operation(T result[], T a, const T b[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    //for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i) {
        if (flags[i])
            result[i] = element_operation(a, b[i], op_enum);
    });
}

template <typename T>
void perform_operation(T result[], const T a[], T b, bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    //for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i) {
        if (flags[i])
            result[i] = element_operation(a[i], b, op_enum);
    });
}

unsigned long long aggregate_operation(const int a[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    unsigned long long result = 0;
    unsigned long long *d_result = sycl::malloc_device<unsigned long long>(1, queue);
    queue.memset(d_result, 0, sizeof(unsigned long long)).wait();
    queue.parallel_for(size, sycl::reduction(d_result, sycl::plus<>()), [=](sycl::id<1> idx, auto &sum) {
        if (flags[idx]) { sum.combine(a[idx]); }
    });
    queue.memcpy(&result, d_result, sizeof(unsigned long long)).wait();
    sycl::free(d_result, queue);
    return result;
}

std::tuple<int **, unsigned long long, bool *> group_by_aggregate2(int **group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op, sycl::queue &queue)
{
    //int *max_values = new int[col_num],
    //    *min_values = new int[col_num];
    int *max_values = sycl::malloc_host<int>(col_num, queue);
    int *min_values = sycl::malloc_host<int>(col_num, queue);
    //queue.fill(max_values, INT_MIN, sizeof(int) * col_num).wait();
    //queue.fill(min_values, INT_MAX, sizeof(int) * col_num).wait();
    unsigned long long prod_ranges = 1;

    // TODO: Pass min and max from the beginning, recalculating them is expensive
    // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:reduction
    // The proper way is seg faulting...
    for (int i = 0; i < col_num; i++)
    {
        int min = INT_MAX,
            max = INT_MIN;
        int * group_columns_i = sycl::malloc_host<int>(col_len, queue);
        queue.memcpy(group_columns_i, group_columns[i], sizeof(int) * col_len).wait();
        for (int j = 0; j < col_len; j++)
        {
            if (flags[j])
            {
                if (group_columns_i[j] < min)
                    min = group_columns_i[j];
                if (group_columns_i[j] > max)
                    max = group_columns_i[j];
            }
        }
        sycl::free(group_columns_i, queue);
        queue.memcpy(&min_values[i], &min, sizeof(int)).wait();
        queue.memcpy(&max_values[i], &max, sizeof(int)).wait();
        //min_values[i] = min;
        //max_values[i] = max;
        prod_ranges *= max - min + 1;
    }

    std::cout << "prod_ranges: " << prod_ranges << std::endl;

    int **results = new int *[col_num + 1];
    int *flat_results = sycl::malloc_device<int>(prod_ranges * (col_num + 1), queue);
    for (int i = 0; i < col_num+1; i++)
        //results[i] = new int[prod_ranges];
        results[i] = sycl::malloc_shared<int>(prod_ranges, queue);
    //results[col_num] = (int *)new uint64_t[prod_ranges];
    //results[col_num] = sycl::malloc_device<uint64_t>(prod_ranges, queue);
    bool *res_flags = sycl::malloc_shared<bool>(prod_ranges, queue);
    //bool *res_flags = new bool[prod_ranges]();
    //for (int i = 0; i < col_len; i++)
    int *h_hashes = sycl::malloc_host<int>(col_len, queue);
    int **all_groups = sycl::malloc_host<int *>(col_num, queue);
    for (int i = 0; i < col_num; i++) {
        all_groups[i] = sycl::malloc_host<int>(col_len, queue);
        queue.memcpy(all_groups[i], group_columns[i], sizeof(int) * col_len).wait();
    }
    for (int i = 0; i < col_len; i++)
        if (flags[i])
        {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++)
            {
                hash += ((all_groups[j][i] - min_values[j]) * mult); //% prod_ranges;
                mult *= (max_values[j] - min_values[j] + 1);
            }
            hash %= prod_ranges;
            h_hashes[i] = hash;
        }
    int *hashes = sycl::malloc_device<int>(col_len, queue);
    queue.memcpy(hashes, h_hashes, sizeof(int) * col_len).wait();
    sycl::free(h_hashes, queue);
    std::cout << "col num: " << col_num << ", col len: " << col_len << std::endl;
    // print groups
    bool *host_sflags = sycl::malloc_host<bool>(col_len, queue);
    queue.memcpy(host_sflags, flags, sizeof(bool) * col_len).wait();
    int count = 0;
    for (size_t i = 0; i < col_num; i++)
    {
        std::cout << "group " << i << ": ";
        for (size_t j = 0; j < col_len; j++)
            if (host_sflags[j])
                count++;
                //std::cout << all_groups[i][j] << " ";
        //std::cout << std::endl;
    }
    std::cout << "count: " << count << std::endl;
    
    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            int hash = hashes[i];
            res_flags[hash] = true;
            flat_results[0 + hash * (col_num+1)] = all_groups[0][i];
            flat_results[1 + hash * (col_num+1)] = all_groups[1][i];
            //for (int j = 0; j < col_num - 1; j++)
            //    flat_results[j + hash * (col_num+1)] = all_groups[j][i];
            //auto sum_obj =
            //    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
            //                    sycl::memory_scope::work_group,
            //                    sycl::access::address_space::global_space>(
            //        *reinterpret_cast<unsigned long long *>(&flat_results[col_num + hash * (col_num+1)]));
            //sum_obj.fetch_add((unsigned long long)(agg_column[i]));
            //((uint64_t *)results[col_num])[hash] += agg_column[i];

            //results[col_num][hash] = 0; // Initialize the aggregation column
            //res_flags[hash] = true;
            //for (int j = 0; j < col_num; j++)
            //    results[j][hash] = group_columns[j][i];
            //((uint64_t *)results[col_num])[hash] += agg_column[i];
        }
    });
    queue.wait();
    std::cout << "done with aggregation: " << prod_ranges << std::endl;
    queue.fill(res_flags, 1, prod_ranges).wait();
    for (int i = 0; i < col_num; i++)
    {
        queue.memcpy(results[i], &flat_results[i * prod_ranges], sizeof(int) * prod_ranges).wait();
        sycl::free(all_groups[i], queue);
    }
    //delete[] max_values;
    //delete[] min_values;
    sycl::free(max_values, queue);
    sycl::free(min_values, queue);
    return std::make_tuple(results, prod_ranges, res_flags);
}

std::tuple<int **, unsigned long long, bool *> group_by_aggregate(int **group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op, sycl::queue &queue)
{
    int *max_values = new int[col_num],
        *min_values = new int[col_num];
    unsigned long long prod_ranges = 1;

    int **h_group_columns = new int *[col_num];
    queue.wait();
    for (int i = 0; i < col_num; i++)
    {
        h_group_columns[i] = new int[col_len];
        queue.memcpy(h_group_columns[i], group_columns[i], sizeof(int) * col_len).wait();
        queue.memcpy(h_group_columns[i], group_columns[i], sizeof(int) * col_len).wait();
    }
    int *agg_column_copy = new int[col_len];
    queue.memcpy(agg_column_copy, agg_column, sizeof(int) * col_len).wait();
    bool *h_flags = new bool[col_len];
    queue.memcpy(h_flags, flags, sizeof(bool) * col_len).wait();
    queue.wait();
    queue.memcpy(agg_column_copy, agg_column, sizeof(int) * col_len).wait();
    queue.memcpy(h_flags, flags, sizeof(bool) * col_len).wait();

    std::cout << "col num: " << col_num << ", col len: " << col_len << std::endl;

    int flags_true = std::count(h_flags, h_flags + col_len, true);
    std::cout << "flags true: " << flags_true << std::endl;

    for (int i = 0; i < col_num; i++)
    {
        int min = INT_MAX,
            max = INT_MIN;
        for (int j = 0; j < col_len; j++)
        {
            if (h_flags[j])
            {
                if (h_group_columns[i][j] < min)
                    min = h_group_columns[i][j];
                if (h_group_columns[i][j] > max)
                    max = h_group_columns[i][j];
            }
        }
        min_values[i] = min;
        max_values[i] = max;
        prod_ranges *= max - min + 1;
    }

    int **results = new int *[col_num + 1];

    for (int i = 0; i < col_num; i++)
        results[i] = new int[prod_ranges];
    results[col_num] = (int *)new uint64_t[prod_ranges];
    bool *res_flags = new bool[prod_ranges]();

    // Initialize results
    for (int i = 0; i < col_num + 1; i++)
        for (int j = 0; j < prod_ranges; j++)
            results[i][j] = 0;

    std::cout << "col num: " << col_num << ", col len: " << col_len << std::endl;

    
    int *d_hashes = sycl::malloc_device<int>(col_len, queue);
    queue.wait();
    queue.memset(d_hashes, 0, sizeof(int) * col_len).wait();
    int *d_max_values = sycl::malloc_device<int>(col_num, queue);
    int *d_min_values = sycl::malloc_device<int>(col_num, queue);
    queue.memcpy(d_max_values, max_values, sizeof(int) * col_num).wait();
    queue.memcpy(d_min_values, min_values, sizeof(int) * col_num).wait();

    /*
    int *multipliers = sycl::malloc_shared<int>(col_num, queue);
    multipliers[0] = 1;
    for (int i = 1; i < col_num; i++)
        multipliers[i] = multipliers[i - 1] * (max_values[i - 1] - min_values[i - 1] + 1);

    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            int hash = 0, mult = 1;
            
            for (int j = 0; j < col_num; j++)
            {
                hash += ((group_columns[j][i] - d_min_values[j]) * multipliers[j]) % prod_ranges;
                //hash = (hash + (((group_columns[j][i] - d_min_values[j]) * multipliers[j]) % prod_ranges)) % prod_ranges;
                //mult *= d_max_values[j] - d_min_values[j] + 1;
            }
            d_hashes[i] = hash % prod_ranges;
        }
    }).wait();
    */


    int *hashes = new int[col_len];
    //queue.memcpy(hashes, d_hashes, sizeof(int) * col_len).wait();
    
    for (int i = 0; i < col_len; i++)
    {
        if (h_flags[i])
        {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++)
            {
                hash += (h_group_columns[j][i] - min_values[j]) * mult;
                mult *= max_values[j] - min_values[j] + 1;
            }
            hash %= prod_ranges;
            hashes[i] = hash;
        }
    }

    //int *d_flat_results = sycl::malloc_device<int>(prod_ranges * (col_num + 1), queue);
    //queue.memset(d_flat_results, 0, sizeof(int) * prod_ranges * (col_num + 1)).wait();

    //bool *d_res_flags = sycl::malloc_device<bool>(prod_ranges, queue);
    //queue.memset(d_res_flags, 0, sizeof(bool) * prod_ranges).wait();

    queue.memcpy(d_hashes, hashes, sizeof(int) * col_len).wait();

    int **d_group_columns = sycl::malloc_device<int *>(col_num - 1, queue);
    for (int i = 0; i < col_num - 1; i++)
    {
        d_group_columns[i] = sycl::malloc_device<int>(col_len, queue);
        //queue.memcpy(d_group_columns[i], group_columns[i], sizeof(int) * col_len).wait();
    }

    std::cout << "allocated group columns" << std::endl;

    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            //int hash = d_hashes[i];
            //d_res_flags[d_hashes[i]] = true;
            for (int j = 0; j < col_num - 1; j++)
                //d_flat_results[d_hashes[i]] = 1;
                d_group_columns[j][i] = 1;
        }
    }).wait();

    for (int i = 0; i < col_len; i++)
    {
        if (h_flags[i])
        {
            res_flags[hashes[i]] = true;
            for (int j = 0; j < col_num; j++)
                results[j][hashes[i]] = h_group_columns[j][i];

            if (agg_op == "SUM")
                ((uint64_t *)results[col_num])[hashes[i]] += agg_column_copy[i];
            else
            {
                // std::cout << "Unsupported aggregate operation: " << agg_op << std::endl;
            }
        }
    }

    delete[] max_values;
    delete[] min_values;
    return std::make_tuple(results, prod_ranges, res_flags);
}
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

std::tuple<int **, unsigned long long, bool *> group_by_aggregate(int **group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op, sycl::queue &queue)
{
    //int *max_values = new int[col_num],
    //    *min_values = new int[col_num];
    int *max_values = sycl::malloc_device<int>(col_num, queue);
    int *min_values = sycl::malloc_device<int>(col_num, queue);
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

    for (int i = 0; i < col_num+1; i++)
        //results[i] = new int[prod_ranges];
        results[i] = sycl::malloc_device<int>(prod_ranges, queue);
    //results[col_num] = (int *)new uint64_t[prod_ranges];
    //results[col_num] = sycl::malloc_device<uint64_t>(prod_ranges, queue);
    bool *res_flags = sycl::malloc_device<bool>(prod_ranges, queue);
    //bool *res_flags = new bool[prod_ranges]();
    //for (int i = 0; i < col_len; i++)
    queue.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i])
        {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++)
            {
                hash += (group_columns[j][i] - min_values[j]) * mult;
                mult *= max_values[j] - min_values[j] + 1;
            }
            hash %= prod_ranges;
            
            //res_flags[hash] = true;
            //for (int j = 0; j < col_num; j++)
            //    results[j][hash] = group_columns[j][i];
            //((uint64_t *)results[col_num])[hash] += agg_column[i];
        }
    });
    queue.wait();
    std::cout << "done with aggregation: " << prod_ranges << std::endl;
    //delete[] max_values;
    //delete[] min_values;
    sycl::free(max_values, queue);
    sycl::free(min_values, queue);
    return std::make_tuple(results, prod_ranges, res_flags);
}
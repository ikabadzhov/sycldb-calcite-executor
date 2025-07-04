#pragma once

#include <string>

#define HASH_SET_LEN 1000000

template <typename T>
inline T element_operation(T a, T b, const std::string &op)
{
    if (op == "*")
        return a * b;
    else if (op == "/")
        return a / b;
    else if (op == "+")
        return a + b;
    else if (op == "-")
        return a - b;
    else
        return 0;
}

template <typename T>
void perform_operation(T result[], const T a[], const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b[i], op);
}

template <typename T>
void perform_operation(T result[], T a, const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a, b[i], op);
}

template <typename T>
void perform_operation(T result[], const T a[], T b, bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b, op);
}

template <typename T, typename U>
void aggregate_operation(U &result, const T a[], bool flags[], int size, const std::string &op)
{
    if (op == "SUM")
    {
        result = 0;
        for (int i = 0; i < size; i++)
            if (flags[i])
                result += a[i];
    }
    else
    {
        // std::cout << "Unsupported aggregate operation: " << op << std::endl;
    }
}

bool is_in_set(int *set, int value, int set_len)
{
    for (int i = 0; i < set_len; i++)
        if (set[i] == value)
            return true;
    return false;
}

std::tuple<int **, unsigned long long, bool *> group_by_aggregate(int **group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op)
{
    int *max_values = new int[col_num],
        *min_values = new int[col_num];
    unsigned long long prod_ranges = 1;

    for (int i = 0; i < col_num; i++)
    {
        int min = INT_MAX,
            max = INT_MIN;
        for (int j = 0; j < col_len; j++)
        {
            if (flags[j])
            {
                if (group_columns[i][j] < min)
                    min = group_columns[i][j];
                if (group_columns[i][j] > max)
                    max = group_columns[i][j];
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

    for (int i = 0; i < col_len; i++)
    {
        if (flags[i])
        {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++)
            {
                hash += (group_columns[j][i] - min_values[j]) * mult;
                mult *= max_values[j] - min_values[j] + 1;
            }
            hash %= prod_ranges;

            res_flags[hash] = true;
            for (int j = 0; j < col_num; j++)
                results[j][hash] = group_columns[j][i];

            if (agg_op == "SUM")
                ((uint64_t *)results[col_num])[hash] += agg_column[i];
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
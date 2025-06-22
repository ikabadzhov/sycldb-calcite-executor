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

std::pair<int **, int> group_by_aggregate(int **group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op)
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

    int *ht = new int[prod_ranges * (col_num + 2)],
        result_size = 0,
        *hash_set = new int[HASH_SET_LEN],
        **results = new int *[col_num + 1];

    for (int i = 0; i < col_num; i++)
        results[i] = new int[HASH_SET_LEN];
    results[col_num] = (int *)new uint64_t[HASH_SET_LEN];

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

            bool insert = !is_in_set(hash_set, hash, result_size);

            if (insert)
            {
                hash_set[result_size] = hash;
                for (int j = 0; j < col_num; j++)
                {
                    results[j][result_size] = group_columns[j][i];
                    ht[hash * (col_num + 2) + j] = group_columns[j][i];
                }
                *((uint64_t *)&ht[hash * (col_num + 2) + col_num]) = 0;
                result_size++;
            }

            if (agg_op == "SUM")
                *((uint64_t *)&ht[hash * (col_num + 2) + col_num]) += agg_column[i]; // this may not work based on the machine (endianness)
            else
            {
                // std::cout << "Unsupported aggregate operation: " << agg_op << std::endl;
            }
        }
    }

    for (int i = 0; i < result_size; i++)
    {
        int hash = 0, mult = 1;
        for (int j = 0; j < col_num; j++)
        {
            hash += (results[j][i] - min_values[j]) * mult;
            mult *= max_values[j] - min_values[j] + 1;
        }
        hash %= prod_ranges;

        ((uint64_t *)results[col_num])[i] = *((uint64_t *)&ht[hash * (col_num + 2) + col_num]);
    }

    delete[] ht;
    delete[] hash_set;
    delete[] max_values;
    delete[] min_values;
    return std::make_pair(results, result_size);
}
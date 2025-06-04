#pragma once
#include <sycl/sycl.hpp>

template <typename T>
struct ColumnData
{
    T *content;
    bool has_ownership;
};

template <typename T>
struct TableData
{
    ColumnData<T> *columns;
    int col_len;
    int col_number;
    bool *flags;
};

TableData<int> generate_dummy(int col_len, int col_number, sycl::queue &queue)
{
    int i, j;

    TableData<int> res;

    res.col_len = col_len;
    res.col_number = col_number;

    //res.columns = new ColumnData<int>[col_number];
    res.columns = sycl::malloc_shared<ColumnData<int>>(col_number, queue);
    //res.flags = new bool[col_len];
    res.flags = sycl::malloc_shared<bool>(col_len, queue);
    for (i = 0; i < col_len; i++)
        res.flags[i] = true; // all rows are valid

    for (i = 0; i < col_number; i++)
    {
        //res.columns[i].content = new int[col_len];
        res.columns[i].content = sycl::malloc_shared<int>(col_len, queue);
        res.columns[i].has_ownership = true;
        for (j = 0; j < col_len; j++)
            res.columns[i].content[j] = (j + i) % 42; // why not
    }

    return res;
}
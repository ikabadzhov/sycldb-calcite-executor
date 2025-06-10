#pragma once

template <typename T>
struct ColumnData
{
    T *content;
    bool has_ownership;
    bool is_aggregate_result;
};

template <typename T>
struct TableData
{
    ColumnData<T> *columns;
    int col_len;
    int col_number;
    bool *flags;
};

TableData<int> generate_dummy(int col_len, int col_number)
{
    int i, j;

    TableData<int> res;

    res.col_len = col_len;
    res.col_number = col_number;

    res.columns = new ColumnData<int>[col_number];
    res.flags = new bool[col_len];

    for (i = 0; i < col_number; i++)
    {
        res.columns[i].content = new int[col_len];
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;
        for (j = 0; j < col_len; j++)
            res.columns[i].content[j] = (j + i) % 42; // why not
    }

    return res;
}
#pragma once

// template <typename T>
// struct ColumnData
// {
//     T *content;
// };

template <typename T>
struct TableData
{
    T **columns;
    unsigned int col_len;
    unsigned int col_number;
    bool *flags;
};

struct TableData<int> generate_dummy(int col_len, int col_number)
{
    int i, j;

    struct TableData<int> res;

    res.col_len = col_len;
    res.col_number = col_number;

    res.columns = new int *[col_number];
    res.flags = new bool[col_len];

    for (i = 0; i < col_number; i++)
    {
        res.columns[i] = new int[col_len];
        for (j = 0; j < col_len; j++)
            res.columns[i][j] = (j + i) % 42; // why not
    }

    return res;
}
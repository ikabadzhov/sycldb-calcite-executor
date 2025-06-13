#pragma once

template <typename T>
struct ColumnData
{
    T *content;
    bool has_ownership;       // Indicates if this column owns the memory for its content
    bool is_aggregate_result; // Indicates if this column is the result of an aggregate operation
    T min_value;              // Minimum value in the column
    T max_value;              // Maximum value in the column
};

template <typename T>
struct TableData
{
    ColumnData<T> *columns;
    int columns_size; // length of the columns array (number of columns loaded)
    int col_len;      // number of rows
    int col_number;   // total number of columns in the table
    bool *flags;      // selection flags
    std::string table_name;
    std::map<int, int> column_indices; // Maps column numbers from calcite to its index in the columns array
};

TableData<int> generate_dummy(int col_len, int col_number)
{
    int i, j;

    TableData<int> res;

    res.col_len = col_len;
    res.col_number = col_number;
    res.columns_size = col_number; // in dummy we load all columns

    res.columns = new ColumnData<int>[col_number];
    res.flags = new bool[col_len];

    for (i = 0; i < col_number; i++)
    {
        // in dummy all columns are loaded so indexes are themselves
        res.column_indices[i] = i;
        res.columns[i].content = new int[col_len];
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;
        res.columns[i].min_value = 0;
        res.columns[i].max_value = 42;
        for (j = 0; j < col_len; j++)
            res.columns[i].content[j] = (j + i) % 42; // why not
    }

    return res;
}
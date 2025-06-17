#pragma once

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
void build_keys_ht(T col[], bool flags[], int col_len, bool ht[], T ht_len, T ht_min_value)
{
    for (int i = 0; i < col_len; i++)
        ht[HASH(col[i], ht_len, ht_min_value)] = flags[i];
}

void build_key_vals_ht(int col[], bool flags[], int col_len, int ht[], int ht_max_value, int ht_min_value)
{
    for (int i = 0; i < col_len; i++)
    {
        if (flags[i])
        {
            int hash = HASH(col[i], ht_max_value, ht_min_value);
            ht[hash << 1] = 1;
            ht[(hash << 1) + 1] = i;
        }
        else
            ht[HASH(col[i], ht_max_value, ht_min_value) << 1] = 0;
    }
}

template <typename T>
void filter_join(T build_col[],
                 bool build_flags[],
                 int build_col_len,
                 T build_max_value,
                 T build_min_value,
                 T probe_col[],
                 bool probe_col_flags[],
                 int probe_col_len)
{
    int ht_len = build_max_value - build_min_value + 1;
    bool *ht = new bool[ht_len];
    std::fill_n(ht, ht_len, false);

    build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value);

    std::cout << "Filter join: build_col_len = " << build_col_len
              << ", probe_col_len = " << probe_col_len
              << ", ht_len = " << ht_len << std::endl;

    for (int i = 0; i < probe_col_len; i++)
        if (probe_col_flags[i])
            probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];

    delete[] ht;
}

void full_join(TableData<int> &probe_table,
               TableData<int> &build_table,
               int probe_col_index,
               int build_col_index)
{
    int *ht = new int[build_table.col_len * 2],
        build_column = build_table.column_indices.at(build_col_index),
        probe_column = probe_table.column_indices.at(probe_col_index);

    std::fill_n(ht, build_table.col_len * 2, 0); // maybe it's not necessary to initialize

    build_key_vals_ht(
        build_table.columns[build_column].content,
        build_table.flags, build_table.col_len, ht,
        build_table.columns[build_column].max_value,
        build_table.columns[build_column].min_value);

    // merge columns
    ColumnData<int> *new_columns = new ColumnData<int>[build_table.columns_size + probe_table.columns_size];
    for (int i = 0; i < probe_table.columns_size; i++)
    {
        new_columns[i].content = probe_table.columns[i].content;
        new_columns[i].has_ownership = probe_table.columns[i].has_ownership;
        probe_table.columns[i].has_ownership = false; // transfer ownership
        new_columns[i].is_aggregate_result = probe_table.columns[i].is_aggregate_result;
        new_columns[i].min_value = probe_table.columns[i].min_value;
        new_columns[i].max_value = probe_table.columns[i].max_value;
    }
    for (auto const &pair : build_table.column_indices)
    {
        int new_index = pair.second + probe_table.columns_size;
        probe_table.column_indices[pair.first + probe_table.col_number] = new_index;
        new_columns[new_index].content = new int[probe_table.col_len];
        new_columns[new_index].has_ownership = true;
        new_columns[new_index].is_aggregate_result = false;
        new_columns[new_index].min_value = build_table.columns[pair.second].min_value;
        new_columns[new_index].max_value = build_table.columns[pair.second].max_value;
    }

    for (int i = 0; i < probe_table.col_len; i++)
    {
        if (probe_table.flags[i])
        {
            int hash = HASH(new_columns[probe_column].content[i],
                            build_table.columns[build_column].max_value,
                            build_table.columns[build_column].min_value);
            if (ht[hash << 1] == 1)
            {
                int build_row_index = ht[(hash << 1) + 1];
                // copy build row to the new columns
                for (int j = 0; j < build_table.columns_size; j++)
                {
                    // assumed same order of columns bewteen indexed and real
                    new_columns[j + probe_table.columns_size].content[i] = build_table.columns[j].content[build_row_index];
                }
            }
            else
                probe_table.flags[i] = false; // mark as not selected
        }
    }

    probe_table.columns_size += build_table.columns_size;
    probe_table.col_number += build_table.col_number;
    delete[] probe_table.columns;
    probe_table.columns = new_columns;
    delete[] ht;
}
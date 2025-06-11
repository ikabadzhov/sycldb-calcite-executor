#pragma once

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
void build_keys_ht(T col[], bool flags[], int col_len, bool ht[], T ht_max_value, T ht_min_value)
{
    for (int i = 0; i < col_len; i++)
        ht[HASH(col[i], ht_max_value, ht_min_value)] = flags[i];
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

    build_keys_ht(build_col, build_flags, build_col_len, ht, build_max_value, build_min_value);

    for (int i = 0; i < probe_col_len; i++)
        if (probe_col_flags[i])
            probe_col_flags[i] = ht[HASH(probe_col[i], build_max_value, build_min_value)];

    delete[] ht;
}
#include "types.hpp"

const std::map<std::string, std::set<int>> table_column_indices(
    {
        {"lineorder", {2, 3, 4, 5, 8, 9, 11, 12, 13}},
        {"part", {0, 2, 3, 4}},
        {"supplier", {0, 3, 4, 5}},
        {"customer", {0, 3, 4, 5}},
        {"ddate", {0, 4, 5}}
    }
);

const std::map<std::string, int> table_column_numbers(
    {
        {"lineorder", 17},
        {"part", 9},
        {"supplier", 7},
        {"customer", 8},
        {"ddate", 17}
    }
);

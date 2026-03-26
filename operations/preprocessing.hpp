#pragma once

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <deque>
#include <string>

#include "../gen-cpp/CalciteServer.h"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/types.hpp"


struct ExecutionInfo
{
    std::map<std::string, std::set<int>> loaded_columns;
    std::map<std::string, int> table_last_used, group_by_columns;
    std::map<int, std::tuple<int, int>> prepare_join_id;
    std::map<std::string, std::tuple<int, int>> prepare_join;
    std::vector<int> dag_order;
};

void parse_expression_columns(const ExprType &expr, std::set<int> &columns);
std::vector<int> dag_topological_sort(const PlanResult &result);
ExecutionInfo parse_execution_info(const PlanResult &result);

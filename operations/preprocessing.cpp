#include "preprocessing.hpp"

#include <algorithm>

void parse_expression_columns(const ExprType &expr, std::set<int> &columns)
{
    if (expr.exprType == ExprOption::COLUMN)
        columns.insert(expr.input);
    else if (expr.exprType == ExprOption::EXPR)
        for (const ExprType &operand : expr.operands)
            parse_expression_columns(operand, columns);
}

std::vector<int> dag_topological_sort(const PlanResult &result)
{
    int **dag = new int *[result.rels.size()];
    std::deque<int> S;
    for (int i = 0; i < result.rels.size(); i++)
    {
        dag[i] = new int[result.rels.size()];
        std::fill_n(dag[i], result.rels.size(), 0);
    }

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            S.push_front(rel.id);
            break;
        case RelNodeType::FILTER:
        case RelNodeType::PROJECT:
        case RelNodeType::AGGREGATE:
        case RelNodeType::SORT:
            dag[rel.id - 1][rel.id] = 1;
            break;
        case RelNodeType::JOIN:
            if (rel.inputs.size() != 2)
            {
                std::cerr << "Join operation: Invalid number of inputs." << std::endl;
                return std::vector<int>();
            }
            dag[rel.inputs[0]][rel.id] = 1;
            dag[rel.inputs[1]][rel.id] = 1;
            break;
        default:
            std::cerr << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            return std::vector<int>();
        }
    }

    std::vector<int> sorted;
    sorted.reserve(result.rels.size());

    while (!S.empty())
    {
        int node = S.back();
        S.pop_back();

        if (S.size() != 0 &&
            result.rels[node].relOp == RelNodeType::TABLE_SCAN &&
            result.rels[node].tables[1] == "lineorder")
        {
            S.push_front(node);
            continue;
        }
        sorted.push_back(node);

        for (int i = 0; i < result.rels.size(); i++)
        {
            if (dag[node][i] == 1)
            {
                dag[node][i] = 0;

                bool has_inbound_edges = false;
                for (int j = 0; j < result.rels.size(); j++)
                {
                    if (dag[j][i] == 1)
                    {
                        has_inbound_edges = true;
                        break;
                    }
                }

                if (!has_inbound_edges)
                    S.push_front(i);
            }
        }
    }

    for (int i = 0; i < result.rels.size(); i++)
        delete[] dag[i];
    delete[] dag;

    if (sorted.size() != result.rels.size())
    {
        std::cerr << "DAG topological sort failed: not all nodes were sorted." << std::endl;
        return std::vector<int>();
    }

    return sorted;
}

ExecutionInfo parse_execution_info(const PlanResult &result)
{
    ExecutionInfo info;
    std::vector<std::vector<std::tuple<std::string, int>>> ops_info;
    ops_info.reserve(result.rels.size());

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
        {
            int num_columns = table_column_numbers.at(rel.tables[1]);
            std::vector<std::tuple<std::string, int>> table;
            table.reserve(num_columns);

            for (int i = 0; i < num_columns; i++)
                table.push_back(std::make_tuple(rel.tables[1], i));

            ops_info.push_back(table);
            info.loaded_columns[rel.tables[1]] = std::set<int>();
            info.table_last_used[rel.tables[1]] = rel.id;
            break;
        }
        case RelNodeType::FILTER:
        {
            std::vector<std::tuple<std::string, int>> op_info(ops_info.back());
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            for (int col : columns)
            {
                const std::string &table_name = std::get<0>(op_info[col]);
                info.loaded_columns[table_name].insert(std::get<1>(op_info[col]));
                info.table_last_used[table_name] = rel.id;
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::PROJECT:
        {
            std::vector<std::tuple<std::string, int>> op_info;
            std::vector<std::tuple<std::string, int>> last_op_info = ops_info.back();

            for (const ExprType &expr : rel.exprs)
            {
                std::set<int> columns;
                parse_expression_columns(expr, columns);

                for (int col : columns)
                {
                    const std::string &table_name = std::get<0>(last_op_info[col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[col]));
                    info.table_last_used[table_name] = rel.id;
                }

                if (!columns.empty())
                {
                    op_info.push_back(std::make_tuple(
                        std::get<0>(last_op_info[*columns.begin()]),
                        std::get<1>(last_op_info[*columns.begin()])
                    ));
                }
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::AGGREGATE:
        {
            std::vector<std::tuple<std::string, int>> op_info;
            std::vector<std::tuple<std::string, int>> last_op_info = ops_info.back();

            for (int agg_col : rel.group)
            {
                const std::string &table_name = std::get<0>(last_op_info[agg_col]);
                int col_index = std::get<1>(last_op_info[agg_col]);
                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
                op_info.push_back(std::make_tuple(table_name, col_index));
                info.group_by_columns[table_name] = col_index;
            }

            for (const AggType &agg : rel.aggs)
            {
                for (int agg_col : agg.operands)
                {
                    const std::string &table_name = std::get<0>(last_op_info[agg_col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[agg_col]));
                    info.table_last_used[table_name] = rel.id;
                }

                int agg_col = agg.operands[0];
                op_info.push_back(std::make_tuple(
                    std::get<0>(last_op_info[agg_col]),
                    std::get<1>(last_op_info[agg_col])
                ));
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::JOIN:
        {
            int left_id = rel.inputs[0];
            int right_id = rel.inputs[1];
            std::vector<std::tuple<std::string, int>> left_info = ops_info[left_id];
            std::vector<std::tuple<std::string, int>> right_info = ops_info[right_id];
            std::vector<std::tuple<std::string, int>> op_info;
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            op_info.reserve(left_info.size() + right_info.size());

            for (int col : columns)
            {
                const bool is_left = col < left_info.size();
                const auto &column_info = is_left ? left_info[col] : right_info[col - left_info.size()];
                const std::string &table_name = std::get<0>(column_info);
                int col_index = std::get<1>(column_info);

                if (table_name != "lineorder")
                {
                    int previous_op_id = info.table_last_used[table_name];
                    if (previous_op_id != rel.id)
                    {
                        int right_column = rel.condition.operands[1].input - left_info.size();
                        info.prepare_join[table_name] = std::make_tuple(rel.id, right_column);
                        info.prepare_join_id[previous_op_id] = std::make_tuple(rel.id, right_column);
                    }
                }

                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
            }

            op_info.insert(op_info.begin(), left_info.begin(), left_info.end());
            op_info.insert(op_info.end(), right_info.begin(), right_info.end());
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::SORT:
        {
            std::vector<std::tuple<std::string, int>> op_info(ops_info.back());
            std::set<int64_t> columns;

            for (const CollationType &col : rel.collation)
                columns.insert(col.field);

            for (int64_t col : columns)
            {
                if (col < op_info.size())
                {
                    const std::string &table_name = std::get<0>(op_info[col]);
                    int col_index = std::get<1>(op_info[col]);
                    info.loaded_columns[table_name].insert(col_index);
                    info.table_last_used[table_name] = rel.id;
                }
            }

            ops_info.push_back(op_info);
            break;
        }
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    info.dag_order = dag_topological_sort(result);
    return info;
}

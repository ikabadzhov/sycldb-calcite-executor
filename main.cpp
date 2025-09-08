#include <iostream>
#include <fstream>
#include <deque>
#include <sycl/sycl.hpp>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/calciteserver_types.h"

#include "kernels/selection.hpp"
#include "kernels/types.hpp"
#include "kernels/aggregation.hpp"
#include "kernels/join.hpp"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

#define MAX_NTABLES 5

std::map<std::string, int> table_column_numbers({{"lineorder", 17},
                                                 {"part", 9},
                                                 {"supplier", 7},
                                                 {"customer", 8},
                                                 {"ddate", 17}});

struct ExecutionInfo
{
    std::map<std::string, std::set<int>> loaded_columns;
    std::map<std::string, int> table_last_used, group_by_columns;
    std::vector<int> dag_order;
};

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
            S.push_front(rel.id); // push the table scan operation to the list of nodes without inbound edges
            break;
        case RelNodeType::FILTER:
        case RelNodeType::PROJECT:
        case RelNodeType::AGGREGATE:
            dag[rel.id - 1][rel.id] = 1; // connect to the previous operation
            break;
        case RelNodeType::JOIN:
            if (rel.inputs.size() != 2)
            {
                std::cerr << "Join operation: Invalid number of inputs." << std::endl;
                return std::vector<int>();
            }
            dag[rel.inputs[0]][rel.id] = 1; // connect left input to the join
            dag[rel.inputs[1]][rel.id] = 1; // connect right input to the join
            break;
        default:
            std::cerr << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            return std::vector<int>();
        }
    }

    // Topological sort of the DAG
    std::vector<int> sorted;
    sorted.reserve(result.rels.size());
    int delayed = 0;

    while (!S.empty())
    {
        int node = S.back();
        S.pop_back();

        if (delayed < 2 &&
            result.rels[node].relOp == RelNodeType::TABLE_SCAN &&
            result.rels[node].tables[0] == "lineorder")
        {
            S.push_front(node);
            delayed++;
            continue;
        }
        sorted.push_back(node);

        for (int i = 0; i < result.rels.size(); i++)
        {
            if (dag[node][i] == 1) // if there is an edge from node to i
            {
                dag[node][i] = 0; // remove the edge

                // check if there are any inbound edges to i
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
                    S.push_front(i); // push to the list of nodes without inbound edges
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

// initially parse the data structure in order to find all columns used and the last time each table was used
ExecutionInfo parse_execution_info(const PlanResult &result)
{
    ExecutionInfo info;

    // ops_info stores information about every operation.
    // index is the operation id, content is a vector of tuples
    // where the first element is the table name and the second is the column number.
    std::vector<std::vector<std::tuple<std::string, int>>> ops_info;
    ops_info.reserve(result.rels.size());

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
        {
            int num_columns = table_column_numbers[rel.tables[1]];
            std::vector<std::tuple<std::string, int>> table;
            table.reserve(num_columns);

            // all the columns of the table
            for (int i = 0; i < num_columns; i++)
                table.push_back(std::make_tuple(rel.tables[1], i));

            ops_info.push_back(table);

            // init info for the table in the result
            info.loaded_columns[rel.tables[1]] = std::set<int>();
            info.table_last_used[rel.tables[1]] = rel.id;
            break;
        }
        case RelNodeType::FILTER:
        {
            // a filter list of columns is the same as the previous operation
            std::vector<std::tuple<std::string, int>> op_info(ops_info.back());
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            // mark columns in the filter as used
            // mark the table as last used at current id
            for (int col : columns)
            {
                std::string table_name = std::get<0>(op_info[col]);
                info.loaded_columns[table_name].insert(std::get<1>(op_info[col]));
                info.table_last_used[table_name] = rel.id;
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::PROJECT:
        {
            std::vector<std::tuple<std::string, int>> op_info, last_op_info = ops_info.back();

            for (const ExprType &expr : rel.exprs)
            {
                std::set<int> columns;
                parse_expression_columns(expr, columns);

                // mark columns in the project as used (if any)
                // mark the table as last used at current id
                for (int col : columns)
                {
                    std::string table_name = std::get<0>(last_op_info[col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[col]));
                    info.table_last_used[table_name] = rel.id;
                }

                // if the expression contains at least one column, use the first one as a reference
                // TODO: improve this by considering all columns
                if (!columns.empty())
                    op_info.push_back(std::make_tuple(
                        std::get<0>(last_op_info[*columns.begin()]),
                        std::get<1>(last_op_info[*columns.begin()])));
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::AGGREGATE:
        {
            std::vector<std::tuple<std::string, int>> op_info, last_op_info = ops_info.back();

            // mark all columns in the group by as used (if any)
            // mark the table as last used at current id
            // save the info about the columns since they form the new table
            for (int agg_col : rel.group)
            {
                std::string table_name = std::get<0>(last_op_info[agg_col]);
                int col_index = std::get<1>(last_op_info[agg_col]);
                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
                op_info.push_back(std::make_tuple(table_name, col_index));
                info.group_by_columns[table_name] = col_index;
            }

            for (const AggType &agg : rel.aggs)
            {
                // save columns and table for every aggregate operation
                for (int agg_col : agg.operands)
                {
                    std::string table_name = std::get<0>(last_op_info[agg_col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[agg_col]));
                    info.table_last_used[table_name] = rel.id;
                }

                // use the first column of the aggregate as a reference
                // TODO: improve this by considering all columns
                int agg_col = agg.operands[0];
                op_info.push_back(std::make_tuple(
                    std::get<0>(last_op_info[agg_col]),
                    std::get<1>(last_op_info[agg_col])));
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::JOIN:
        {
            int left_id = rel.inputs[0], right_id = rel.inputs[1];
            std::vector<std::tuple<std::string, int>> left_info = ops_info[left_id],
                                                      right_info = ops_info[right_id],
                                                      op_info;
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            op_info.reserve(left_info.size() + right_info.size());

            // mark columns in the join condition as used
            // mark the tables as last used at current id
            for (int col : columns)
            {
                std::string table_name = std::get<0>(
                    (col < left_info.size())
                        ? left_info[col]
                        : right_info[col - left_info.size()]);

                int col_index = std::get<1>(
                    (col < left_info.size())
                        ? left_info[col]
                        : right_info[col - left_info.size()]);

                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
            }

            // insert left and right info into the operation info
            op_info.insert(op_info.begin(), left_info.begin(), left_info.end());
            op_info.insert(op_info.end(), right_info.begin(), right_info.end());

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

// logicals are AND, OR etc. while comparisons are ==, <= etc.
// So checking alpha characters is enough to determine if the operation is logical.
bool is_filter_logical(const std::string &op)
{
    for (int i = 0; i < op.length(); i++)
        if (!isalpha(op[i]))
            return false;
    return true;
}

void parse_filter(const ExprType &expr,
                  const TableData<int> table_data,
                  std::string parent_op, sycl::queue &queue)
{
    // Recursive parsing of EXRP types. LITERAL and COLUMN are handled in parent EXPR type.
    if (expr.exprType != ExprOption::EXPR)
    {
        std::cout << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
        return;
    }

    if (expr.op == "SEARCH")
    {
        int col_index = table_data.column_indices.at(expr.operands[0].input);
        //bool *local_flags = new bool[table_data.col_len];
        bool *local_flags = sycl::malloc_shared<bool>(table_data.col_len, queue);
        queue.prefetch(local_flags, table_data.col_len * sizeof(bool));

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            selection(local_flags,
                      table_data.columns[col_index].content,
                      ">=", lower, "NONE", table_data.col_len, queue);
            selection(local_flags,
                      table_data.columns[col_index].content,
                      "<=", upper, "AND", table_data.col_len, queue);

            for (int i = 0; i < table_data.col_len; i++)
                table_data.flags[i] = logical(
                    get_logical_op(parent_op),
                    table_data.flags[i], local_flags[i]);
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            selection(local_flags,
                      table_data.columns[col_index].content,
                      "==", first, "NONE", table_data.col_len, queue);
            selection(local_flags,
                      table_data.columns[col_index].content,
                      "==", second, "OR", table_data.col_len, queue);

            for (int i = 0; i < table_data.col_len; i++)
                table_data.flags[i] = logical(
                    get_logical_op(parent_op),
                    table_data.flags[i], local_flags[i]);
        }
        //delete[] local_flags;
        sycl::free(local_flags, queue);
    }
    else if (is_filter_logical(expr.op))
    {
        // Logical operation between other expressions. Pass parent op to the first then use the current op.
        // TODO: check if passing parent logic is correct in general
        bool parent_op_used = false;
        for (const ExprType &operand : expr.operands)
        {
            parse_filter(operand, table_data, parent_op_used ? expr.op : parent_op, queue);
            parent_op_used = true;
        }
    }
    else
    {
        // Comparison between two operands
        int **cols = new int *[2];
        bool literal = false;
        if (expr.operands.size() != 2)
        {
            std::cout << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
            return;
        }

        // Get the pointer to the two columns or make a new column with the literal value as first cell
        for (int i = 0; i < 2; i++)
        {
            switch (expr.operands[i].exprType)
            {
            case ExprOption::COLUMN:
                cols[i] = table_data.columns[table_data.column_indices.at(expr.operands[i].input)].content;
                break;
            case ExprOption::LITERAL:
                //cols[i] = new int[1];
                cols[i] = sycl::malloc_shared<int>(1, queue);
                queue.prefetch(cols[i], sizeof(int));
                literal = true;
                cols[i][0] = expr.operands[i].literal.value;
                break;
            default:
                std::cout << "Filter condition: Unsupported parsing ExprType "
                          << expr.operands[i].exprType
                          << " for comparison operand"
                          << std::endl;
                break;
            }
        }

        // Assumed literal is always the second operand.
        if (literal)
        {
            selection(table_data.flags, cols[0], expr.op, cols[1][0], parent_op, table_data.col_len, queue);
            //delete[] cols[1];
            sycl::free(cols[1], queue);
        }
        else
            selection(table_data.flags, cols[0], expr.op, cols[1], parent_op, table_data.col_len, queue);

        delete[] cols;
    }
}

void parse_project(const std::vector<ExprType> &exprs, TableData<int> &table_data, sycl::queue &queue)
{
    ColumnData<int> *new_columns = new ColumnData<int>[exprs.size()];
    for (size_t i = 0; i < exprs.size(); i++)
    {
        switch (exprs[i].exprType)
        {
        case ExprOption::COLUMN:
            // Copy the column data from the original table and pass ownership
            new_columns[i].content = table_data.columns[table_data.column_indices.at(exprs[i].input)].content;
            new_columns[i].min_value = table_data.columns[table_data.column_indices.at(exprs[i].input)].min_value;
            new_columns[i].max_value = table_data.columns[table_data.column_indices.at(exprs[i].input)].max_value;
            new_columns[i].has_ownership = true;
            new_columns[i].is_aggregate_result = table_data.columns[table_data.column_indices.at(exprs[i].input)].is_aggregate_result;
            table_data.columns[table_data.column_indices.at(exprs[i].input)].has_ownership = false;
            if (exprs[i].input == table_data.group_by_column)
                table_data.group_by_column = i; // update group by column index
            break;
        case ExprOption::LITERAL:
            // create a new column with the literal value
            //new_columns[i].content = new int[table_data.col_len];
            new_columns[i].content = sycl::malloc_device<int>(table_data.col_len, queue);
            queue.fill(new_columns[i].content, exprs[i].literal.value, table_data.col_len);
            //std::fill_n(new_columns[i].content, table_data.col_len, exprs[i].literal.value);
            new_columns[i].min_value = exprs[i].literal.value;
            new_columns[i].max_value = exprs[i].literal.value;
            new_columns[i].has_ownership = true;
            new_columns[i].is_aggregate_result = false;
            break;
        case ExprOption::EXPR:
            // Assumed only 2 operands which are either COLUMN or LITERAL
            if (exprs[i].operands.size() != 2)
            {
                std::cout << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                return;
            }
            //new_columns[i].content = new int[table_data.col_len];
            new_columns[i].content = sycl::malloc_device<int>(table_data.col_len, queue);
            //queue.prefetch(new_columns[i].content, table_data.col_len * sizeof(int));
            new_columns[i].has_ownership = true;
            new_columns[i].is_aggregate_result = false;

            // call the perform_operation overloaded function based on the operands types
            // and set min and max values of the column
            if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                exprs[i].operands[1].exprType == ExprOption::COLUMN)
            {
                perform_operation(new_columns[i].content,
                                  table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].content,
                                  table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].content,
                                  table_data.flags, table_data.col_len, exprs[i].op, queue);
                //new_columns[i].min_value =
                //    std::min(table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].min_value,
                //             table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].min_value);
                //new_columns[i].max_value =
                //    std::max(table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].max_value,
                //             table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].max_value);
            }
            else if (exprs[i].operands[0].exprType == ExprOption::LITERAL &&
                     exprs[i].operands[1].exprType == ExprOption::COLUMN)
            {
                perform_operation(new_columns[i].content,
                                  (int)exprs[i].operands[0].literal.value,
                                  table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].content,
                                  table_data.flags, table_data.col_len, exprs[i].op, queue);
                //new_columns[i].min_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].min_value;
                //new_columns[i].max_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].max_value;
            }
            else if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                     exprs[i].operands[1].exprType == ExprOption::LITERAL)
            {
                perform_operation(new_columns[i].content,
                                  table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].content,
                                  (int)exprs[i].operands[1].literal.value,
                                  table_data.flags, table_data.col_len, exprs[i].op, queue);
                //new_columns[i].min_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].min_value;
                //new_columns[i].max_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].max_value;
            }
            else
            {
                std::cout << "Project operation: Unsupported parsing ExprType "
                          << exprs[i].operands[0].exprType << " and "
                          << exprs[i].operands[1].exprType
                          << " for EXPR" << std::endl;
                return;
            }
            break;
        }
    }

    // Free old columns and replace with new ones
    // for (int i = 0; i < table_data.columns_size; i++)
    //     if (table_data.columns[i].has_ownership)
    //         delete[] table_data.columns[i].content;
    // delete[] table_data.columns;

    table_data.columns = new_columns;
    table_data.col_number = exprs.size();
    table_data.columns_size = exprs.size();

    // update column indices (they are now just themselves)
    table_data.column_indices.clear();
    for (int i = 0; i < table_data.col_number; i++)
        table_data.column_indices[i] = i;
}

void parse_aggregate(TableData<int> &table_data, const AggType &agg, const std::vector<long> &group, sycl::queue &queue)
{
    if (group.size() == 0)
    {
        unsigned long long result = aggregate_operation(table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
                            table_data.flags, table_data.col_len, agg.agg, queue);

        // Free old columns and replace with the result column
        /*for (int i = 0; i < table_data.columns_size; i++)
            if (table_data.columns[i].has_ownership)
                delete[] table_data.columns[i].content;
        delete[] table_data.columns;
        delete[] table_data.flags;
        table_data.column_indices.clear();*/

        table_data.columns = new ColumnData<int>[1];
        table_data.columns[0].content = new int[sizeof(unsigned long long) / sizeof(int)];
        ((unsigned long long *)table_data.columns[0].content)[0] = result;
        table_data.columns[0].has_ownership = true;
        table_data.columns[0].is_aggregate_result = true;
        table_data.columns[0].min_value = 0; // TODO: set real min value
        table_data.columns[0].max_value = 0; // TODO: set real max value
        table_data.col_number = 1;
        table_data.columns_size = 1;
        table_data.col_len = 1;
        //table_data.flags = new bool[1];
        table_data.flags = sycl::malloc_shared<bool>(1, queue);
        queue.prefetch(table_data.flags, sizeof(bool));
        table_data.flags[0] = true;
        table_data.column_indices[0] = 0;
    }
    else
    {
        //int **group_columns = new int *[group.size()];
        int **group_columns = sycl::malloc_shared<int*>(group.size(), queue);
        for (int i = 0; i < group.size(); i++)
            group_columns[i] = table_data.columns[table_data.column_indices.at(group[i])].content;
        std::tuple<int **, unsigned long long, bool *> agg_res = group_by_aggregate(
            group_columns,
            table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
            table_data.flags, group.size(), table_data.col_len, agg.agg, queue);
        //delete[] group_columns;
        sycl::free(group_columns, queue);

        // Free old columns and replace with the result columns
        /*for (int i = 0; i < table_data.columns_size; i++)
            if (table_data.columns[i].has_ownership)
                delete[] table_data.columns[i].content;
        delete[] table_data.columns;
        delete[] table_data.flags;
        */
        table_data.column_indices.clear();

        table_data.columns = new ColumnData<int>[group.size() + 1];
        for (int i = 0; i < group.size(); i++)
        {
            table_data.columns[i].content = std::get<0>(agg_res)[i];
            table_data.columns[i].has_ownership = true;
            table_data.columns[i].is_aggregate_result = false;
            table_data.columns[i].min_value = 0; // TODO: set real min value
            table_data.columns[i].max_value = 0; //  TODO: set real max value
            table_data.column_indices[i] = i;
        }

        table_data.columns[group.size()].content = std::get<0>(agg_res)[group.size()];
        table_data.columns[group.size()].has_ownership = true;
        table_data.columns[group.size()].is_aggregate_result = true;
        table_data.columns[group.size()].min_value = 0; // TODO: set real min value
        table_data.columns[group.size()].max_value = 0; // TODO: set real max value
        table_data.col_number = group.size() + 1;
        table_data.columns_size = group.size() + 1;
        table_data.col_len = std::get<1>(agg_res);
        table_data.flags = std::get<2>(agg_res);
        table_data.column_indices[group.size()] = group.size();
    }
}

void parse_join(const RelNode &rel, TableData<int> &left_table, TableData<int> &right_table, const std::map<std::string, int> &table_last_used, sycl::queue &queue)
{
    int left_column = rel.condition.operands[0].input,
        right_column = rel.condition.operands[1].input - left_table.col_number;

    if (left_column < 0 ||
        left_column >= left_table.col_number ||
        right_column < 0 ||
        right_column >= right_table.col_number)
    {
        std::cout << "Join operation: Invalid column indices in join condition." << std::endl;
        return;
    }

    // filter joins if the right table is last accessed at this operation
    if (right_table.table_name != "" && table_last_used.at(right_table.table_name) == rel.id)
    {
        filter_join(right_table.columns[right_table.column_indices.at(right_column)].content,
                    right_table.flags, right_table.col_len,
                    right_table.columns[right_table.column_indices.at(right_column)].max_value,
                    right_table.columns[right_table.column_indices.at(right_column)].min_value,
                    left_table.columns[left_table.column_indices.at(left_column)].content,
                    left_table.flags, left_table.col_len, queue);
    }
    else if (left_table.table_name == "lineorder")
    {
        full_join(left_table, right_table, left_column, right_column, queue);
    }
    else
    {
        std::cout << "Join operation Unsupported" << std::endl;
    }
    left_table.col_number += right_table.col_number;
}

void print_result(const TableData<int> &table_data)
{
    int res_count = 0;
    std::cout << "Result table:" << std::endl;
    for (int i = 0; i < table_data.col_len; i++)
    {
        if (table_data.flags[i])
        {
            for (int j = 0; j < table_data.columns_size; j++) // at this point column_size should match col_number
                std::cout << ((table_data.columns[j].is_aggregate_result) ? ((unsigned long long *)table_data.columns[j].content)[i] : table_data.columns[j].content[i]) << " ";
            std::cout << std::endl;
            res_count++;
        }
    }

    std::cout << "Total rows in result: " << res_count << std::endl;
}

void execute_result(const PlanResult &result)
{
    sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    TableData<int> tables[MAX_NTABLES];
    int current_table = 0;
    int *output_table = sycl::malloc_shared<int>(result.rels.size(), queue);
    queue.prefetch(output_table, result.rels.size() * sizeof(int));
    //*output_table = new int[result.rels.size()];
    // used to track the output table of each operation, in order to be referenced in the joins. other operation types just use the previous output table
    ExecutionInfo exec_info = parse_execution_info(result);

    for (const RelNode &rel : result.rels)
    {
        if (rel.relOp != RelNodeType::TABLE_SCAN)
            continue;
        std::cout << "Table Scan on: " << rel.tables[1] << std::endl;
        if (exec_info.loaded_columns.find(rel.tables[1]) == exec_info.loaded_columns.end())
        {
            std::cout << "Table " << rel.tables[1] << " was never loaded." << std::endl;
            return;
        }
        std::set<int> &column_idxs = exec_info.loaded_columns[rel.tables[1]];
        // tables[current_table] = generate_dummy(100 * (current_table + 1), table_column_numbers[rel.tables[1]]);
        tables[current_table] = loadTable(rel.tables[1], table_column_numbers[rel.tables[1]], column_idxs, queue);
        tables[current_table].table_name = rel.tables[1];
        if (exec_info.group_by_columns.find(rel.tables[1]) != exec_info.group_by_columns.end())
            tables[current_table].group_by_column = exec_info.group_by_columns[rel.tables[1]];
        output_table[rel.id] = current_table;
        current_table++;
    }

    for (int id : exec_info.dag_order)
    {
        const RelNode &rel = result.rels[id];
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            break;
        case RelNodeType::FILTER:
            // std::cout << "Filter condition: " << rel.condition << std::endl;
            parse_filter(rel.condition, tables[output_table[rel.id - 1]], "", queue);
            output_table[rel.id] = output_table[rel.id - 1];
            break;
        case RelNodeType::PROJECT:
            std::cout << "Project operation" << std::endl;
            parse_project(rel.exprs, tables[output_table[rel.id - 1]], queue);
            output_table[rel.id] = output_table[rel.id - 1];
            break;
        case RelNodeType::AGGREGATE:
            std::cout << "Aggregate operation: " << rel.aggs[0].agg << std::endl;
            parse_aggregate(tables[output_table[rel.id - 1]], rel.aggs[0], rel.group, queue);
            output_table[rel.id] = output_table[rel.id - 1];
            break;
        case RelNodeType::JOIN:
            std::cout << "Join operation" << std::endl;
            parse_join(rel, tables[output_table[rel.inputs[0]]], tables[output_table[rel.inputs[1]]], exec_info.table_last_used, queue);
            output_table[rel.id] = output_table[rel.inputs[0]];
            break;
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    print_result(tables[output_table[result.rels.size() - 1]]);

    /*for (int i = 0; i < current_table; i++)
    {
        delete[] tables[i].flags;
        for (int j = 0; j < tables[i].columns_size; j++)
            if (tables[i].columns[j].has_ownership)
                delete[] tables[i].columns[j].content;
        delete[] tables[i].columns;
    }
    delete[] output_table;*/
}

int main(int argc, char **argv)
{
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    CalciteServerClient client(protocol);
    std::string sql;

    if (argc == 2)
    {
        std::ifstream file(argv[1]);
        if (!file.is_open())
        {
            std::cerr << "Could not open file: " << argv[1] << std::endl;
            return 1;
        }

        sql.assign((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());

        file.close();
    }
    else
    {
        sql = "select sum(lo_revenue)\
        from lineorder, ddate, part, supplier\
        where lo_orderdate = d_datekey\
        and lo_partkey = p_partkey\
        and lo_suppkey = s_suppkey;";
    }

    try
    {
        std::cout << "SQL Query: " << sql << std::endl;
        transport->open();
        std::cout << "Transport opened successfully." << std::endl;
        PlanResult result;
        client.parse(result, sql);

        std::cout << "Result: " << result << std::endl;

        execute_result(result);

        client.shutdown();

        transport->close();
    }
    catch (TTransportException &e)
    {
        std::cerr << "Transport exception: " << e.what() << std::endl;
    }
    catch (TException &e)
    {
        std::cerr << "Thrift exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}
#include <iostream>
#include <fstream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/calciteserver_types.h"
#include "kernels/selection.hpp"
#include "kernels/types.hpp"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

#define MAX_NTABLES 5

std::map<std::string, int> table_column_numbers({{"lineorder", 17},
                                                 {"part", 9},
                                                 {"supplier", 7},
                                                 {"customer", 8},
                                                 {"ddate", 17}});

// logicals are AND, OR etc. while comparisons are ==, <= etc.
// So checking alpha characters is enough to determine if the operation is logical.
bool is_filter_logical(const std::string &op)
{
    for (int i = 0; i < op.length(); i++)
        if (!isalpha(op[i]))
            return false;
    return true;
}

int perform_operation(int a, int b, const std::string &op)
{
    if (op == "*")
        return a * b;
    else if (op == "/")
        return a / b;
    else if (op == "+")
        return a + b;
    else if (op == "-")
        return a - b;
    else
    {
        std::cout << "Unsupported operation: " << op << std::endl;
        return 0;
    }
}

void parse_filter(const ExprType &expr,
                  const TableData<int> table_data,
                  std::string parent_op = "")
{
    // Recursive parsing of EXRP types. LITERAL and COLUMN are handled in parent EXPR type.
    if (expr.exprType != ExprOption::EXPR)
    {
        std::cout << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
        return;
    }

    if (is_filter_logical(expr.op))
    {
        // Logical operation between other expressions. Pass parent op to the first then use the current op.
        // TODO: check if passing parent logic is correct in general
        bool parent_op_used = false;
        for (const ExprType &operand : expr.operands)
        {
            parse_filter(operand, table_data, parent_op_used ? expr.op : parent_op);
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
                cols[i] = table_data.columns[expr.operands[i].input].content;
                break;
            case ExprOption::LITERAL:
                cols[i] = new int[1];
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
            selection(table_data.flags, cols[0], expr.op, cols[1][0], parent_op, table_data.col_len);
            delete[] cols[1];
        }
        else
            selection(table_data.flags, cols[0], expr.op, cols[1], parent_op, table_data.col_len);

        delete[] cols;
    }
}

void parse_project(const std::vector<ExprType> &exprs, TableData<int> &table_data)
{
    ColumnData<int> *new_columns = new ColumnData<int>[exprs.size()];

    for (size_t i = 0; i < exprs.size(); i++)
    {
        switch (exprs[i].exprType)
        {
        case ExprOption::COLUMN:
            new_columns[i].content = table_data.columns[exprs[i].input].content;
            new_columns[i].has_ownership = true;
            table_data.columns[exprs[i].input].has_ownership = false;
            break;
        case ExprOption::LITERAL:
            new_columns[i].content = new int[table_data.col_len];
            std::fill_n(new_columns[i].content, table_data.col_len, exprs[i].literal.value);
            new_columns[i].has_ownership = true;
            break;
        case ExprOption::EXPR:
            // Assumed only 2 operands which are either COLUMN or LITERAL
            if (exprs[i].operands.size() != 2)
            {
                std::cout << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                return;
            }
            new_columns[i].content = new int[table_data.col_len];
            new_columns[i].has_ownership = true;

            for (int j = 0; j < table_data.col_len; j++)
            {
                if (table_data.flags[j])
                {
                    int val1, val2;
                    switch (exprs[i].operands[0].exprType)
                    {
                    case ExprOption::COLUMN:
                        val1 = table_data.columns[exprs[i].operands[0].input].content[j];
                        break;
                    case ExprOption::LITERAL:
                        val1 = exprs[i].operands[0].literal.value;
                        break;
                    default:
                        std::cout << "Project operation: Unsupported parsing ExprType "
                                  << exprs[i].operands[0].exprType
                                  << " for first operand" << std::endl;
                        return;
                    }

                    switch (exprs[i].operands[1].exprType)
                    {
                    case ExprOption::COLUMN:
                        val2 = table_data.columns[exprs[i].operands[1].input].content[j];
                        break;
                    case ExprOption::LITERAL:
                        val2 = exprs[i].operands[1].literal.value;
                        break;
                    default:
                        std::cout << "Project operation: Unsupported parsing ExprType "
                                  << exprs[i].operands[1].exprType
                                  << " for second operand" << std::endl;
                        return;
                    }

                    new_columns[i].content[j] = perform_operation(val1, val2, exprs[i].op);
                }
                else
                    new_columns[i].content[j] = 0;
            }
            break;
        }
    }

    // Free old columns and replace with new ones
    for (int i = 0; i < table_data.col_number; i++)
        if (table_data.columns[i].has_ownership)
            delete[] table_data.columns[i].content;
    delete[] table_data.columns;

    table_data.columns = new_columns;
    table_data.col_number = exprs.size();
}

unsigned long long parse_aggregate(const TableData<int> &table_data, const AggType &agg)
{
    unsigned long long result = 0;

    if (agg.agg == "SUM")
    {
        for (int i = 0; i < table_data.col_len; i++)
        {
            if (table_data.flags[i])
            {
                result += table_data.columns[agg.operands[0]].content[i]; // Assumed single operand for SUM
            }
        }
    }
    else
    {
        std::cout << "Unsupported aggregate function: " << agg.agg << std::endl;
        return 0;
    }

    return result;
}

void execute_result(const PlanResult &result)
{
    TableData<int> tables[MAX_NTABLES];
    int current_table = 0;

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            std::cout << "Table Scan on: " << rel.tables[0] << std::endl;
            // TODO load real data
            tables[current_table] = generate_dummy(100 * (current_table + 1), table_column_numbers[rel.tables[0]]);
            current_table++;
            break;
        case RelNodeType::FILTER:
            // std::cout << "Filter condition: " << rel.condition << std::endl;
            parse_filter(rel.condition, tables[0]); // TODO: proper table detection for queries > q13
            break;
        case RelNodeType::PROJECT:
            std::cout << "Project operation" << std::endl;
            parse_project(rel.exprs, tables[0]); // TODO: proper table detection for queries > q13
            break;
        case RelNodeType::AGGREGATE:
            std::cout << "Aggregate operation: " << rel.aggs[0].agg << std::endl;
            std::cout << "Aggregate result: " << parse_aggregate(tables[0], rel.aggs[0]) << std::endl;
            break;
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    for (int i = 0; i < current_table; i++)
    {
        delete[] tables[i].flags;
        for (int j = 0; j < tables[i].col_number; j++)
            if (tables[i].columns[j].has_ownership)
                delete[] tables[i].columns[j].content;
        delete[] tables[i].columns;
    }
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
        sql = "select sum(lo_extendedprice * lo_discount) as revenue\
                            from lineorder\
                            where lo_orderdate >= 19930101 and\
                            lo_orderdate <= 19940101 and lo_discount >= 1 and lo_discount <= 3 and lo_quantity < 25;";
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
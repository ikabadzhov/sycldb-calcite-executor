#include <iostream>
#include <fstream>
#include <sycl/sycl.hpp>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/calciteserver_types.h"

#include "kernels/selection.hpp"
#include "kernels/types.hpp"
#include "kernels/aggregation.hpp"

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

void parse_filter(const ExprType &expr,
                  const TableData<int> table_data, sycl::queue &queue,
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
            parse_filter(operand, table_data, queue, parent_op_used ? expr.op : parent_op);
            parent_op_used = true;
        }
    }
    else
    {
        if (expr.operands.size() != 2)
        {
            std::cout << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
            return;
        }
        //std::cout << "Column: " << expr.operands[0].input << std::endl;
        int *column_data = table_data.columns[expr.operands[0].input].content;
        int literal_value = expr.operands[1].literal.value;

        selection(table_data.flags, column_data, expr.op, literal_value,
                  parent_op, table_data.col_len, queue);
    }
}

void parse_project(const std::vector<ExprType> &exprs, TableData<int> &table_data, sycl::queue &queue)
{
    //ColumnData<int> *new_columns = new ColumnData<int>[exprs.size()];
    ColumnData<int> *new_columns = sycl::malloc_shared<ColumnData<int>>(exprs.size(), queue);

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
            //new_columns[i].content = new int[table_data.col_len];
            new_columns[i].content = sycl::malloc_shared<int>(table_data.col_len, queue);
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
            //new_columns[i].content = new int[table_data.col_len];
            new_columns[i].content = sycl::malloc_shared<int>(table_data.col_len, queue);
            new_columns[i].has_ownership = true;

            if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                exprs[i].operands[1].exprType == ExprOption::COLUMN)
                perform_operation(new_columns[i].content,
                                  table_data.columns[exprs[i].operands[0].input].content,
                                  table_data.columns[exprs[i].operands[1].input].content,
                                  table_data.flags, table_data.col_len, exprs[i].op);
            else if (exprs[i].operands[0].exprType == ExprOption::LITERAL &&
                     exprs[i].operands[1].exprType == ExprOption::COLUMN)
                perform_operation(new_columns[i].content,
                                  (int)exprs[i].operands[0].literal.value,
                                  table_data.columns[exprs[i].operands[1].input].content,
                                  table_data.flags, table_data.col_len, exprs[i].op);
            else if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                     exprs[i].operands[1].exprType == ExprOption::LITERAL)
                perform_operation(new_columns[i].content,
                                  table_data.columns[exprs[i].operands[0].input].content,
                                  (int)exprs[i].operands[1].literal.value,
                                  table_data.flags, table_data.col_len, exprs[i].op);
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
    for (int i = 0; i < table_data.col_number; i++)
    {
        if (table_data.columns[i].has_ownership)
            sycl::free(table_data.columns[i].content, queue);
    }
    sycl::free(table_data.columns, queue);

    table_data.columns = new_columns;
    table_data.col_number = exprs.size();
}

unsigned long long parse_aggregate(const TableData<int> &table_data, const AggType &agg, sycl::queue &queue)
{
    unsigned long long result;

    aggregate_operation(result, table_data.columns[agg.operands[0]].content,
                        table_data.flags, table_data.col_len, agg.agg, queue);

    return result;
}

void execute_result(const PlanResult &result)
{
    TableData<int> tables[MAX_NTABLES];
    int current_table = 0;

    sycl::queue queue{sycl::gpu_selector_v};

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            std::cout << "Table Scan on: " << rel.tables[0] << std::endl;
            // TODO load real data
            tables[current_table] = generate_dummy(100 * (current_table + 1), table_column_numbers[rel.tables[0]], queue);
            current_table++;
            break;
        case RelNodeType::FILTER:
            std::cout << "Filter condition: " << rel.condition << std::endl;
            parse_filter(rel.condition, tables[0], queue); // TODO: proper table detection for queries > q13
            break;
        case RelNodeType::PROJECT:
            std::cout << "Project operation" << std::endl;
            parse_project(rel.exprs, tables[0], queue); // TODO: proper table detection for queries > q13
            break;
        case RelNodeType::AGGREGATE:
            std::cout << "Aggregate operation: " << rel.aggs[0].agg << std::endl;
            std::cout << "Aggregate result: " << parse_aggregate(tables[0], rel.aggs[0], queue) << std::endl;
            break;
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    for (int i = 0; i < current_table; i++)
    {
        sycl::free(tables[i].flags, queue);
        for (int j = 0; j < tables[i].col_number; j++)
        {
            if (tables[i].columns[j].has_ownership)
                sycl::free(tables[i].columns[j].content, queue);
        }
        sycl::free(tables[i].columns, queue);
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
                            where lo_orderdate >= 1\
                            and lo_orderdate <= 41 and lo_discount >= 1 and lo_discount <= 40 and lo_quantity < 25;";
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
#include "ddor_executor.hpp"

#include <iostream>

#include "common.hpp"
#include "executor/result_writer.hpp"
#include "models/transient_table.hpp"
#include "operations/preprocessing.hpp"
#include "runtime/runtime.hpp"

namespace executor
{

namespace
{

class EndTimer2;
class EndTimer3;

} // namespace

std::chrono::duration<double, std::milli> execute_ddor_plan(
    const PlanResult &result,
    const std::string &query_path,
    std::vector<Table> &tables,
    runtime_setup::RuntimeQueues &queues,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators,
    int primary_device_index,
    std::ostream &perf_out)
{
    auto load_start = std::chrono::high_resolution_clock::now();

    const ExecutionInfo exec_info = parse_execution_info(result);
    std::vector<int> output_table(result.rels.size(), -1);
    std::vector<TransientTable> transient_tables;

    for (const RelNode &rel : result.rels)
    {
        if (rel.relOp != RelNodeType::TABLE_SCAN)
            continue;

        #if not PERFORMANCE_MEASUREMENT_ACTIVE
        std::cout << "Table Scan on: " << rel.tables[1] << std::endl;
        #endif

        const auto loaded_columns_it = exec_info.loaded_columns.find(rel.tables[1]);
        if (loaded_columns_it == exec_info.loaded_columns.end())
        {
            std::cerr << "Table " << rel.tables[1] << " was never loaded." << std::endl;
            return std::chrono::duration<double, std::milli>::zero();
        }

        Table *table_ptr = runtime_setup::find_table_by_name(tables, rel.tables[1]);
        if (table_ptr == nullptr)
        {
            std::cerr << "Table " << rel.tables[1] << " not found among loaded tables." << std::endl;
            return std::chrono::duration<double, std::milli>::zero();
        }

        if (primary_device_index >= 0)
        {
            for (int col_idx : loaded_columns_it->second)
            {
                table_ptr->move_column_to_device_background(
                    col_idx,
                    primary_device_index,
                    queues.copy_device_queues[primary_device_index]
                );
                table_ptr->activate_column_background_device_buffers(
                    col_idx,
                    primary_device_index
                );
            }
        }

        TransientTable &table = transient_tables.emplace_back(
            table_ptr,
            queues.cpu_queue,
            queues.device_queues,
            queues.copy_device_queues,
            cpu_allocator,
            device_allocators
        );

        const auto group_by_it = exec_info.group_by_columns.find(rel.tables[1]);
        if (group_by_it != exec_info.group_by_columns.end())
            table.set_group_by_column(group_by_it->second);

        output_table[rel.id] = transient_tables.size() - 1;
    }

    const auto load_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> load_time = load_end - load_start;
    const auto kernel_start = std::chrono::high_resolution_clock::now();

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Execution order: ";
    for (int id : exec_info.dag_order)
        std::cout << id << " -> ";
    std::cout << std::endl;
    #endif

    for (int id : exec_info.dag_order)
    {
        const RelNode &rel = result.rels[id];
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            break;
        case RelNodeType::FILTER:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Starting Filter operation." << std::endl;
            const auto start_filter = std::chrono::high_resolution_clock::now();
            #endif

            const int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_filter(
                rel.condition,
                "",
                cpu_allocator,
                device_allocators
            );
            output_table[id] = prev_table_idx;

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            const auto end_filter = std::chrono::high_resolution_clock::now();
            std::cout << "Filter operation ("
                << std::chrono::duration<double, std::milli>(end_filter - start_filter).count()
                << " ms)" << std::endl;
            #endif
            break;
        }
        case RelNodeType::PROJECT:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Starting Project operation." << std::endl;
            const auto start_project = std::chrono::high_resolution_clock::now();
            #endif

            const int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_project(
                rel.exprs,
                cpu_allocator,
                device_allocators
            );
            output_table[id] = prev_table_idx;

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            const auto end_project = std::chrono::high_resolution_clock::now();
            std::cout << "Project operation ("
                << std::chrono::duration<double, std::milli>(end_project - start_project).count()
                << " ms)" << std::endl;
            #endif
            break;
        }
        case RelNodeType::AGGREGATE:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Starting Aggregate operation." << std::endl;
            const auto start_aggregate = std::chrono::high_resolution_clock::now();
            #endif

            const int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_aggregate(
                rel.aggs[0],
                rel.group,
                cpu_allocator,
                device_allocators
            );
            output_table[id] = prev_table_idx;

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            const auto end_aggregate = std::chrono::high_resolution_clock::now();
            std::cout << "Aggregate operation ("
                << std::chrono::duration<double, std::milli>(end_aggregate - start_aggregate).count()
                << " ms)" << std::endl;
            #endif
            break;
        }
        case RelNodeType::JOIN:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Starting Join operation." << std::endl;
            const auto start_join = std::chrono::high_resolution_clock::now();
            #endif

            const int left_table_idx = output_table[rel.inputs[0]];
            const int right_table_idx = output_table[rel.inputs[1]];

            transient_tables[left_table_idx].apply_join(
                transient_tables[right_table_idx],
                rel,
                cpu_allocator,
                device_allocators
            );
            output_table[id] = left_table_idx;

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            const auto end_join = std::chrono::high_resolution_clock::now();
            std::cout << "Join operation ("
                << std::chrono::duration<double, std::milli>(end_join - start_join).count()
                << " ms)" << std::endl;
            #endif
            break;
        }
        case RelNodeType::SORT:
            output_table[id] = output_table[id - 1];
            break;
        default:
            std::cerr << "RelNodeType " << rel.relOp << " not yet supported in DDOR." << std::endl;
            break;
        }
    }

    auto final_exec_dependencies =
        transient_tables[output_table[result.rels.size() - 1]].execute_pending_kernels();
    runtime_setup::wait_for_dependencies_and_throw(
        final_exec_dependencies.first,
        final_exec_dependencies.second
    );

    const auto kernel_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> kernel_duration = kernel_end - kernel_start;
    const std::chrono::duration<double, std::milli> duration = kernel_end - load_start;

    std::cout << "Engine Breakdown: Load " << load_time.count()
        << " ms, Kernel " << kernel_duration.count() << " ms - " << std::flush;

    queues.cpu_queue.single_task<EndTimer2>([=]() {}).wait();
    for (sycl::queue &q : queues.device_queues)
        q.single_task<EndTimer3>([=]() {}).wait();

    #if PERFORMANCE_MEASUREMENT_ACTIVE
    perf_out << duration.count() << '\n';
    #else
    TransientTable &final_table = transient_tables[output_table[result.rels.size() - 1]];
    save_result(final_table, query_path, cpu_allocator, device_allocators);
    #endif

    return duration;
}

} // namespace executor

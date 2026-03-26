#include "result_writer.hpp"

#include <fstream>
#include <iostream>

#include "app/query_utils.hpp"
#include "runtime/runtime.hpp"

namespace executor
{

void save_result(
    TransientTable &table,
    const std::string &query_path,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators)
{
    const std::string query_name = app::query_stem_from_path(query_path);
    std::cout << "Saving result to " << query_name << ".res" << std::endl;

    std::ofstream outfile(query_name + ".res");
    if (!outfile.is_open())
    {
        std::cerr << "Could not open result file for writing." << std::endl;
        return;
    }

    auto materialize_dependencies = table.materialize_host_view(
        cpu_allocator,
        device_allocators
    );
    runtime_setup::wait_for_dependencies_and_throw(
        materialize_dependencies.first,
        materialize_dependencies.second
    );
    table.assert_flags_to_cpu();

    const std::vector<Column *> columns = table.get_columns();
    const bool *flags = table.get_flags_host();

    for (uint64_t row = 0; row < table.get_nrows(); ++row)
    {
        if (!flags[row])
            continue;

        const uint64_t segment_index = row / SEGMENT_SIZE;
        const uint64_t offset = row % SEGMENT_SIZE;
        bool first = true;

        for (const Column *col : columns)
        {
            if (col == nullptr)
                continue;

            if (!first)
                outfile << ' ';

            const Segment &segment = col->get_segments()[segment_index];
            if (col->get_is_aggregate_result())
                outfile << segment.get_aggregate_data(false, -1)[offset];
            else
                outfile << segment.get_data(false, -1)[offset];

            first = false;
        }
        outfile << '\n';
    }
}

} // namespace executor

#pragma once

#include <string>
#include <vector>

#include "models/transient_table.hpp"
#include "operations/memory_manager.hpp"

namespace executor
{

void save_result(
    TransientTable &table,
    const std::string &query_path,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators);

} // namespace executor

#pragma once

#include <chrono>
#include <ostream>
#include <string>
#include <vector>

#include "gen-cpp/calciteserver_types.h"
#include "models/models.hpp"
#include "operations/memory_manager.hpp"
#include "runtime/runtime.hpp"

namespace executor
{

std::chrono::duration<double, std::milli> execute_ddor_plan(
    const PlanResult &result,
    const std::string &query_path,
    std::vector<Table> &tables,
    runtime_setup::RuntimeQueues &queues,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators,
    int primary_device_index,
    std::ostream &perf_out = std::cout);

} // namespace executor

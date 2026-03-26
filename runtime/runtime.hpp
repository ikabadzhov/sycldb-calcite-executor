#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "common.hpp"
#include "models/models.hpp"
#include "operations/memory_manager.hpp"

namespace runtime_setup
{

struct RuntimeQueues
{
    sycl::queue cpu_queue;
    std::vector<sycl::queue> device_queues;
    std::vector<sycl::queue> copy_device_queues;
};

struct RuntimeConfig
{
    bool many_device_mode = false;
    bool reuse_allocators_across_repetitions = false;
    int primary_device_index = -1;
    uint64_t cpu_allocator_size = SIZE_TEMP_MEMORY_CPU;
    uint64_t cpu_allocator_region_size = SIZE_TEMP_MEMORY_CPU;
};

struct RuntimeEnvironment
{
    RuntimeQueues queues;
    RuntimeConfig config;
    std::vector<Table> tables;
};

RuntimeEnvironment build_runtime_environment();
std::vector<int> build_preferred_device_order(const std::vector<sycl::queue> &device_queues);
void print_runtime_summary(const RuntimeEnvironment &runtime);

memory_manager build_cpu_allocator(const RuntimeEnvironment &runtime);
std::vector<memory_manager> build_device_allocators(const RuntimeEnvironment &runtime);
std::vector<sycl::event> reset_allocators(
    const RuntimeEnvironment &runtime,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators);
void wait_for_reset_events(const std::vector<sycl::event> &events);

void wait_for_all_queues_and_throw(const RuntimeQueues &queues);
void wait_for_dependencies_and_throw(
    const std::vector<sycl::event> &cpu_events,
    const std::vector<std::vector<sycl::event>> &device_events);

Table *find_table_by_name(std::vector<Table> &tables, const std::string &name);

} // namespace runtime_setup

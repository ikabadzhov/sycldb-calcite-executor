#include "runtime.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace runtime_setup
{

namespace
{

int device_preload_priority(const sycl::device &device)
{
    const auto backend = device.get_backend();
    if (backend == sycl::backend::ext_oneapi_cuda)
        return 400;
    if (backend == sycl::backend::ext_oneapi_hip)
        return 300;
    if (backend == sycl::backend::opencl && device.is_cpu())
        return 200;
    if (backend == sycl::backend::ext_oneapi_level_zero)
        return 100;
    return 0;
}

std::vector<sycl::device> discover_execution_devices()
{
    std::vector<sycl::device> devices = sycl::device::get_devices();
    std::stable_sort(
        devices.begin(),
        devices.end(),
        [](const sycl::device &lhs, const sycl::device &rhs)
        {
            if (lhs.is_gpu() != rhs.is_gpu())
                return lhs.is_gpu();
            return lhs.get_info<sycl::info::device::name>() <
                rhs.get_info<sycl::info::device::name>();
        }
    );
    return devices;
}

std::vector<Table> load_base_tables(RuntimeQueues &queues)
{
    std::vector<Table> tables;
    tables.reserve(MAX_NTABLES);

    tables.emplace_back("part", queues.cpu_queue, queues.device_queues);
    tables.emplace_back("supplier", queues.cpu_queue, queues.device_queues);
    tables.emplace_back("customer", queues.cpu_queue, queues.device_queues);
    tables.emplace_back("ddate", queues.cpu_queue, queues.device_queues);
    tables.emplace_back("lineorder", queues.cpu_queue, queues.device_queues);

    for (Table &table : tables)
        table.set_copy_device_queues(queues.copy_device_queues);

    return tables;
}

} // namespace

std::vector<int> build_preferred_device_order(const std::vector<sycl::queue> &device_queues)
{
    std::vector<int> order(device_queues.size());
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(
        order.begin(),
        order.end(),
        [&](int lhs, int rhs)
        {
            const sycl::device lhs_device = device_queues[lhs].get_device();
            const sycl::device rhs_device = device_queues[rhs].get_device();

            const int lhs_priority = device_preload_priority(lhs_device);
            const int rhs_priority = device_preload_priority(rhs_device);
            if (lhs_priority != rhs_priority)
                return lhs_priority > rhs_priority;

            return lhs_device.get_info<sycl::info::device::global_mem_size>() >
                rhs_device.get_info<sycl::info::device::global_mem_size>();
        }
    );

    return order;
}

RuntimeEnvironment build_runtime_environment()
{
    RuntimeEnvironment runtime{
        RuntimeQueues{ sycl::queue{ sycl::default_selector_v }, {}, {} },
        RuntimeConfig{},
        {}
    };

    const std::vector<sycl::device> execution_devices = discover_execution_devices();
    runtime.queues.device_queues.reserve(execution_devices.size());
    runtime.queues.copy_device_queues.reserve(execution_devices.size());

    std::cout << "Found " << execution_devices.size() << " execution device(s):" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    for (const sycl::device &device : execution_devices)
    {
        const auto name = device.get_info<sycl::info::device::name>();
        const auto vendor = device.get_info<sycl::info::device::vendor>();
        const auto memsize = device.get_info<sycl::info::device::global_mem_size>();
        const auto platform =
            device.get_info<sycl::info::device::platform>()
                .get_info<sycl::info::platform::name>();
        const auto max_compute_units =
            device.get_info<sycl::info::device::max_compute_units>();

        std::cout << "Name:    " << name
            << "\nVendor:  " << vendor
            << "\nMemSize (MB): " << (memsize >> 20)
            << "\nMax Compute Units: " << max_compute_units
            << "\nPlatform: " << platform
            << "\nBackend: " << device.get_backend()
            << "\n---------------------------------" << std::endl;

        runtime.queues.device_queues.emplace_back(device);
        runtime.queues.copy_device_queues.emplace_back(
            device,
            sycl::property::queue::in_order{}
        );

        if (runtime.queues.device_queues.size() == 4)
            break;
    }

    runtime.config.many_device_mode = runtime.queues.device_queues.size() > 2;
    runtime.config.reuse_allocators_across_repetitions =
        !runtime.queues.device_queues.empty();
    runtime.config.cpu_allocator_region_size = runtime.config.many_device_mode ?
        (((uint64_t)1) << 30) :
        SIZE_TEMP_MEMORY_CPU;

    const std::vector<int> preferred_order =
        build_preferred_device_order(runtime.queues.device_queues);
    runtime.config.primary_device_index =
        preferred_order.empty() ? -1 : preferred_order.front();

    runtime.tables = load_base_tables(runtime.queues);

    wait_for_all_queues_and_throw(runtime.queues);
    print_runtime_summary(runtime);

    return runtime;
}

void print_runtime_summary(const RuntimeEnvironment &runtime)
{
    std::cout << "Running on CPU: "
        << runtime.queues.cpu_queue.get_device().get_info<sycl::info::device::name>()
        << std::endl;
    for (const sycl::queue &q : runtime.queues.device_queues)
    {
        std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
    }

    for (const Table &table : runtime.tables)
    {
        std::cout << table.get_name() << " num segments: "
            << table.num_segments() << std::endl;
    }

    uint64_t total_host_mem = 0;
    std::vector<uint64_t> total_device_mem_per_queue(
        runtime.queues.device_queues.size(),
        0
    );
    for (const Table &table : runtime.tables)
    {
        total_host_mem += table.get_data_size(false, -1);
        for (size_t d = 0; d < runtime.queues.device_queues.size(); ++d)
            total_device_mem_per_queue[d] += table.get_data_size(true, d);
    }

    std::cout << "Total memory used by tables:\nCPU: "
        << (total_host_mem >> 20) << " MB" << std::endl;
    for (size_t d = 0; d < runtime.queues.device_queues.size(); ++d)
    {
        std::cout << "Device " << d
            << " (" << runtime.queues.device_queues[d].get_device().get_info<sycl::info::device::name>() << "): "
            << (total_device_mem_per_queue[d] >> 20) << " MB" << std::endl;
    }
}

memory_manager build_cpu_allocator(const RuntimeEnvironment &runtime)
{
    return memory_manager(
        const_cast<sycl::queue &>(runtime.queues.cpu_queue),
        runtime.config.cpu_allocator_size,
        runtime.config.cpu_allocator_region_size
    );
}

std::vector<memory_manager> build_device_allocators(const RuntimeEnvironment &runtime)
{
    std::vector<memory_manager> device_allocators;
    device_allocators.reserve(runtime.queues.device_queues.size());

    for (const sycl::queue &queue_ref : runtime.queues.device_queues)
    {
        sycl::queue &queue = const_cast<sycl::queue &>(queue_ref);
        const sycl::device device = queue.get_device();
        const auto backend = device.get_backend();
        const uint64_t mem_size =
            device.get_info<sycl::info::device::global_mem_size>();
        const bool is_opencl_cpu_device =
            backend == sycl::backend::opencl && device.is_cpu();
        const uint64_t max_allocator_size = is_opencl_cpu_device ?
            SIZE_TEMP_MEMORY_CPU :
            ((((uint64_t)10) << 30) + (((uint64_t)512) << 20));

        uint64_t allocator_size = std::min<uint64_t>(
            std::max<uint64_t>(SIZE_TEMP_MEMORY_GPU, mem_size / 2),
            max_allocator_size
        );
        uint64_t allocator_region_size = ((uint64_t)2) << 30;

        if (!device.is_gpu())
        {
            if (!is_opencl_cpu_device)
                allocator_size = std::min<uint64_t>(allocator_size, (((uint64_t)1) << 30));
            allocator_region_size = is_opencl_cpu_device ?
                (((uint64_t)1) << 30) :
                (((uint64_t)256) << 20);
        }

        if (runtime.config.many_device_mode)
        {
            const uint64_t min_budget = is_opencl_cpu_device ?
                (((uint64_t)4) << 30) :
                (((uint64_t)1) << 30);

            if (!device.is_gpu())
            {
                allocator_size = std::min<uint64_t>(
                    allocator_size,
                    std::max<uint64_t>(
                        min_budget,
                        mem_size / (runtime.queues.device_queues.size() + 2)
                    )
                );
            }

            allocator_region_size = std::min<uint64_t>(
                allocator_region_size,
                std::max<uint64_t>(
                    is_opencl_cpu_device ?
                        (((uint64_t)1) << 30) :
                        (((uint64_t)256) << 20),
                    allocator_size / 2
                )
            );
        }

        if (backend == sycl::backend::ext_oneapi_level_zero)
        {
            allocator_size = std::min<uint64_t>(allocator_size, ((uint64_t)10) << 30);
            allocator_region_size = std::min<uint64_t>(allocator_region_size, ((uint64_t)1) << 30);
        }

        device_allocators.emplace_back(queue, allocator_size, allocator_region_size);
    }

    return device_allocators;
}

std::vector<sycl::event> reset_allocators(
    const RuntimeEnvironment &runtime,
    memory_manager &cpu_allocator,
    std::vector<memory_manager> &device_allocators)
{
    std::vector<sycl::event> events = cpu_allocator.reset();

    for (size_t d = 0; d < device_allocators.size(); ++d)
    {
        auto device_events = device_allocators[d].reset();
        events.insert(events.end(), device_events.begin(), device_events.end());
    }

    return events;
}

void wait_for_reset_events(const std::vector<sycl::event> &events)
{
    if (!events.empty())
        sycl::event::wait_and_throw(events);
}

void wait_for_all_queues_and_throw(const RuntimeQueues &queues)
{
    const_cast<sycl::queue &>(queues.cpu_queue).wait_and_throw();
    for (const sycl::queue &q : queues.device_queues)
        const_cast<sycl::queue &>(q).wait_and_throw();
    for (const sycl::queue &q : queues.copy_device_queues)
        const_cast<sycl::queue &>(q).wait_and_throw();
}

void wait_for_dependencies_and_throw(
    const std::vector<sycl::event> &cpu_events,
    const std::vector<std::vector<sycl::event>> &device_events)
{
    if (!cpu_events.empty())
        sycl::event::wait_and_throw(cpu_events);

    for (const auto &events : device_events)
    {
        if (!events.empty())
            sycl::event::wait_and_throw(events);
    }
}

Table *find_table_by_name(std::vector<Table> &tables, const std::string &name)
{
    for (Table &table : tables)
    {
        if (table.get_name() == name)
            return &table;
    }
    return nullptr;
}

} // namespace runtime_setup

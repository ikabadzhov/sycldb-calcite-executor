#pragma once

#include <sycl/sycl.hpp>

#include <optional>
#include <deque>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "../common.hpp"

#include "models.hpp"
#include "execution.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/common.hpp"

class TransientTable
{
private:
    struct AsyncRowIds
    {
        int *row_ids_device;
        int *row_ids_host;
        uint32_t *selected_count_device;
        uint32_t *selected_count_host;
        sycl::event selected_count_copy_event;
        sycl::event row_ids_copy_event;
    };

    bool *flags_host;
    bool *flags_host_secondary;
    std::vector<bool *> flags_devices;
    std::vector<bool *> flags_devices_secondary;
    std::vector<bool> flags_modified_host;
    std::vector<std::vector<bool>> flags_modified_devices;
    sycl::queue &cpu_queue;
    std::vector<sycl::queue> &device_queues;
    std::vector<sycl::queue> &copy_device_queues;

    std::vector<Column *> current_columns;
    std::deque<Column> materialized_columns;
    uint64_t nrows;
    Column *group_by_column;
    uint64_t group_by_column_index;
    std::vector<std::vector<KernelBundle>> pending_kernels;
    std::vector<sycl::event> pending_kernels_dependencies_cpu;
    std::vector<std::vector<sycl::event>> pending_kernels_dependencies_devices;
public:
    TransientTable(Table *base_table,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        std::vector<sycl::queue> &copy_device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators
    )
        :
        flags_modified_devices(device_queues.size()),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        copy_device_queues(copy_device_queues),

        nrows(base_table->get_nrows()),
        group_by_column(nullptr),
        group_by_column_index(0),
        pending_kernels_dependencies_devices(device_queues.size())
    {
        // std::cout << "Creating transient table with " << nrows << " rows." << std::endl;

        const std::vector<Column> &base_columns = base_table->get_columns();
        std::vector<bool> base_device_usage(device_queues.size(), false);
        for (const Column &col : base_columns)
        {
            for (const Segment &segment : col.get_segments())
            {
                const std::vector<bool> &segment_devices = segment.get_on_device_vec();
                for (int d = 0; d < device_queues.size() && d < segment_devices.size(); d++)
                    base_device_usage[d] = base_device_usage[d] || segment_devices[d];
            }
        }

        flags_host = cpu_allocator.alloc<bool>(nrows, false);
        flags_host_secondary = nullptr;
        auto e1 = cpu_queue.fill<bool>(flags_host, true, nrows);

        std::vector<sycl::event> gpu_events;
        gpu_events.reserve(device_queues.size());
        flags_devices.reserve(device_queues.size());
        flags_devices_secondary.reserve(device_queues.size());
        for (int d = 0; d < device_queues.size(); d++)
        {
            if (!base_device_usage[d])
            {
                flags_devices.push_back(nullptr);
                flags_devices_secondary.push_back(nullptr);
                gpu_events.push_back(sycl::event());
                continue;
            }

            flags_devices.push_back(device_allocators[d].alloc<bool>(nrows, true));
            flags_devices_secondary.push_back(nullptr);
            gpu_events.push_back(
                device_queues[d].fill<bool>(flags_devices[d], true, nrows)
            );
        }

        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

        flags_modified_host.resize(segment_num, false);
        for (int d = 0; d < device_queues.size(); d++)
            flags_modified_devices[d].resize(segment_num, false);

        current_columns.reserve(base_columns.size() + 100);
        for (const Column &col : base_columns)
            current_columns.push_back(const_cast<Column *>(&col));

        pending_kernels_dependencies_cpu.push_back(e1);
        for (int d = 0; d < device_queues.size(); d++)
        {
            if (!base_device_usage[d])
                continue;

            pending_kernels_dependencies_devices[d].reserve(segment_num);
            for (uint64_t segment_index = 0; segment_index < segment_num; segment_index++)
            {
                std::vector<sycl::event> deps = { gpu_events[d] };

                for (const Column *col : current_columns)
                {
                    if (col == nullptr)
                        continue;

                    const std::vector<Segment> &segments = col->get_segments();
                    if (segment_index >= segments.size())
                        continue;

                    const Segment &seg = segments[segment_index];
                    if (seg.is_on_device(d) && seg.has_background_copy(d))
                        deps.push_back(seg.get_background_copy_event(d));
                }

                pending_kernels_dependencies_devices[d].push_back(
                    device_queues[d].submit(
                        [&](sycl::handler &cgh)
                        {
                            cgh.depends_on(deps);
                            cgh.single_task([=]() {});
                        }
                    )
                );
            }
        }
        // std::cout << "Transient table created." << std::endl;
    }

    ~TransientTable()
    {
    }

    std::vector<Column *> get_columns() const { return current_columns; }
    uint64_t get_nrows() const { return nrows; }
    const Column *get_group_by_column() const { return group_by_column; }
    const bool *get_flags_host() const { return flags_host; }
    std::vector<Column *> get_active_columns() const
    {
        std::vector<Column *> cols;
        cols.reserve(current_columns.size());
        for (Column *col : current_columns)
        {
            if (col != nullptr)
                cols.push_back(col);
        }
        return cols;
    }

    std::pair<std::vector<sycl::event>, std::vector<std::vector<sycl::event>>> materialize_host_view(
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        const std::vector<Column *> &columns_to_sync = {})
    {
        std::vector<Column *> target_columns =
            columns_to_sync.empty() ? get_active_columns() : columns_to_sync;
        ensure_host_state_available(target_columns, cpu_allocator, device_allocators);
        return execute_pending_kernels();
    }

    void set_group_by_column(uint64_t col)
    {
        group_by_column = current_columns[col];
        group_by_column_index = col;
    }

    Column &register_materialized_column(Column &col)
    {
        col.set_copy_device_queues(copy_device_queues);
        return col;
    }

    void ensure_secondary_host_flags(memory_manager &cpu_allocator)
    {
        if (flags_host_secondary != nullptr)
            return;

        flags_host_secondary = cpu_allocator.alloc<bool>(nrows, false);
    }

    void ensure_primary_host_flags(memory_manager &cpu_allocator)
    {
        if (flags_host != nullptr)
            return;

        flags_host = cpu_allocator.alloc<bool>(nrows, false);
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        flags_modified_host.resize(segment_num, false);
    }

    void ensure_secondary_device_flags(int device_index, std::vector<memory_manager> &device_allocators)
    {
        if (flags_devices_secondary[device_index] != nullptr)
            return;

        flags_devices_secondary[device_index] = device_allocators[device_index].alloc<bool>(nrows, true);
    }

    void ensure_secondary_flag_buffers(
        bool host_needed,
        const std::vector<bool> &devices_needed,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        if (host_needed)
            ensure_secondary_host_flags(cpu_allocator);

        for (int d = 0; d < device_queues.size(); d++)
        {
            if (devices_needed[d])
                ensure_secondary_device_flags(d, device_allocators);
        }
    }

    bool column_needs_host_sync_on_device(const Column *column, int device_index) const
    {
        if (column == nullptr)
            return false;

        for (const Segment &segment : column->get_segments())
        {
            if (segment.is_on_device(device_index) && segment.needs_copy_on(false, -1))
                return true;
        }

        return false;
    }

    void ensure_host_state_available(
        const std::vector<Column *> &columns_to_sync,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        ensure_primary_host_flags(cpu_allocator);

        for (int d = 0; d < device_queues.size(); d++)
        {
            bool need_sync = false;
            for (bool modified : flags_modified_devices[d])
            {
                if (modified)
                {
                    need_sync = true;
                    break;
                }
            }

            if (!need_sync)
            {
                for (const Column *column : columns_to_sync)
                {
                    if (column_needs_host_sync_on_device(column, d))
                    {
                        need_sync = true;
                        break;
                    }
                }
            }

            if (need_sync)
                compress_and_sync(cpu_allocator, device_allocators[d], d, columns_to_sync);
        }
    }

    void ensure_host_flags_available(
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        ensure_host_state_available({}, cpu_allocator, device_allocators);
    }

    void swap_flag_buffers(bool host_touched, const std::vector<bool> &device_touched)
    {
        if (host_touched && flags_host_secondary != nullptr)
            std::swap(flags_host, flags_host_secondary);

        for (int d = 0; d < device_queues.size(); d++)
        {
            if (device_touched[d] && flags_devices_secondary[d] != nullptr)
                std::swap(flags_devices[d], flags_devices_secondary[d]);
        }
    }

    void swap_flag_buffers(const std::vector<KernelBundle> &ops)
    {
        bool host_touched = false;
        std::vector<bool> device_touched(device_queues.size(), false);

        for (const auto &bundle : ops)
        {
            if (bundle.is_on_device())
                device_touched[bundle.get_device_index()] = true;
            else
                host_touched = true;
        }

        swap_flag_buffers(host_touched, device_touched);
    }

    friend std::ostream &operator<<(std::ostream &out, const TransientTable &table)
    {
        for (uint64_t i = 0; i < table.nrows; i++)
        {
            if (table.flags_host[i])
            {
                for (uint64_t j = 0; j < table.current_columns.size(); j++)
                {
                    const Column *col = table.current_columns[j];
                    if (col != nullptr && col->get_segments().size() > 0)
                        out << (col->get_is_aggregate_result() ? col->get_aggregate_value(i) : col->operator[](i)) << ((j < table.current_columns.size() - 1) ? " " : "");
                }
                out << "\n";
            }
        }

        return out;
    }

    void add_pending_kernels(const std::vector<KernelBundle> &ops)
    {
        if (pending_kernels.empty())
        {
            pending_kernels.push_back(ops);
        }
        else
        {
            auto &last_ops = pending_kernels.back();
            bool can_merge = (last_ops.size() == ops.size());
            for (size_t i = 0; i < ops.size() && can_merge; ++i)
            {
                if (last_ops[i].is_on_device() != ops[i].is_on_device() ||
                    last_ops[i].get_device_index() != ops[i].get_device_index())
                {
                    can_merge = false;
                }
            }

            if (can_merge)
            {
                for (size_t i = 0; i < ops.size(); ++i)
                {
                    for (const auto &k : ops[i].get_kernels())
                    {
                        last_ops[i].add_kernel(k);
                    }
                }
            }
            else
            {
                pending_kernels.push_back(ops);
            }
        }
    }

    std::pair<std::vector<sycl::event>, std::vector<std::vector<sycl::event>>> execute_pending_kernels()
    {
        // std::cout << "start execute" << std::endl;
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        std::vector<sycl::event> events_cpu;
        std::vector<std::vector<sycl::event>> events_devices(device_queues.size());
        bool executed_cpu = false;
        std::vector<bool> executed_devices(device_queues.size(), false);

        if (pending_kernels.size() == 0)
        {
            events_cpu = pending_kernels_dependencies_cpu;
            for (int d = 0; d < device_queues.size(); d++)
            {
                events_devices[d] = pending_kernels_dependencies_devices[d];
                pending_kernels_dependencies_devices[d].clear();
            }
            pending_kernels_dependencies_cpu.clear();

            // std::cout << "No pending kernels to execute." << std::endl;

            return { events_cpu, events_devices };
        }

        for (const auto &phases : pending_kernels)
        {
            if (phases.size() != segment_num)
            {
                std::cerr << "Pending kernels segment number mismatch: expected " << segment_num << ", got " << phases.size() << std::endl;
                throw std::runtime_error("Pending kernels segment number mismatch.");
            }
        }

        for (int d = 0; d < device_queues.size(); d++)
            events_devices[d].reserve(segment_num);
        events_cpu.reserve(segment_num);

        for (uint64_t segment_index = 0; segment_index < segment_num; segment_index++)
        {
            std::vector<sycl::event> deps_cpu, tmp;
            std::vector<std::vector<sycl::event>> deps_devices(device_queues.size());
            bool kernel_present_cpu = false;
            std::vector<bool> kernel_present_devices(device_queues.size(), false);

            for (int d = 0; d < device_queues.size(); d++)
            {
                if (pending_kernels_dependencies_devices[d].size() == segment_num)
                    deps_devices[d].push_back(pending_kernels_dependencies_devices[d][segment_index]);
                else
                    deps_devices[d] = pending_kernels_dependencies_devices[d];
            }

            if (pending_kernels_dependencies_cpu.size() == segment_num)
                deps_cpu.push_back(pending_kernels_dependencies_cpu[segment_index]);
            else
                deps_cpu = pending_kernels_dependencies_cpu;

            for (const auto &phases : pending_kernels)
            {
                const KernelBundle &bundle = phases[segment_index];
                tmp = bundle.execute(
                    cpu_queue,
                    device_queues,
                    deps_cpu,
                    deps_devices
                );

                if (bundle.is_on_device())
                {
                    int device_index = bundle.get_device_index();
                    deps_devices[device_index] = std::move(tmp);
                    kernel_present_devices[device_index] = true;
                }
                else
                {
                    deps_cpu = std::move(tmp);
                    kernel_present_cpu = true;
                }
            }

            if (kernel_present_cpu)
            {
                events_cpu.insert(
                    events_cpu.end(),
                    deps_cpu.begin(),
                    deps_cpu.end()
                );
                executed_cpu = true;
            }

            for (int d = 0; d < device_queues.size(); d++)
            {
                if (!kernel_present_devices[d])
                    continue;

                events_devices[d].insert(
                    events_devices[d].end(),
                    deps_devices[d].begin(),
                    deps_devices[d].end()
                );
                executed_devices[d] = true;
            }
        }

        // std::cout << "All segments executed." << std::endl;

        for (int d = 0; d < device_queues.size(); d++)
        {
            if (!executed_devices[d])
                events_devices[d] = pending_kernels_dependencies_devices[d];

            pending_kernels_dependencies_devices[d].clear();
        }

        if (!executed_cpu)
            events_cpu = pending_kernels_dependencies_cpu;

        pending_kernels.clear();
        pending_kernels_dependencies_cpu.clear();

        // std::cout << "end execute" << std::endl;

        return { events_cpu, events_devices };
    }

    void assert_flags_to_cpu()
    {
        for (auto &flags_modified_gpu : flags_modified_devices)
        {
            for (bool modified : flags_modified_gpu)
            {
                if (modified)
                {
                    std::cerr << "Flags on GPU modified but expected to be on CPU." << std::endl;
                    throw std::runtime_error("Flags on GPU modified but expected to be on CPU.");
                }
            }
        }
    }

    AsyncCountResult count_flags_true(bool on_device, int device_index)
    {
        auto dependencies = execute_pending_kernels();

        return count_true_flags_async(
            on_device ? flags_devices[device_index] : flags_host,
            nrows,
            on_device ? device_queues[device_index] : cpu_queue,
            on_device ? dependencies.second[device_index] : dependencies.first
        );
    }

    AsyncRowIds build_row_ids(
        int segment_n,
        int segment_size,
        memory_manager &gpu_allocator,
        int device_index,
        const std::vector<sycl::event> &dependencies = {}
    )
    {
        bool *flags = flags_devices[device_index] + segment_n * SEGMENT_SIZE;
        int *row_ids_gpu = gpu_allocator.alloc<int>(segment_size, true);
        int *row_ids_host = gpu_allocator.alloc<int>(segment_size, false);
        uint32_t *selected_count_device = gpu_allocator.alloc<uint32_t>(1, true);
        uint32_t *selected_count_host = gpu_allocator.alloc<uint32_t>(1, false);

        auto e_init = device_queues[device_index].fill<uint32_t>(selected_count_device, 0u, 1);
        auto e_build = device_queues[device_index].submit(
            [&](sycl::handler &cgh)
            {
                if (!dependencies.empty())
                    cgh.depends_on(dependencies);
                cgh.depends_on(e_init);
                cgh.parallel_for(
                    sycl::range<1>(segment_size),
                    [=](sycl::id<1> idx)
                    {
                        auto i = idx[0];
                        if (!flags[i])
                            return;

                        sycl::atomic_ref<
                            uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space
                        > count_atomic(*selected_count_device);
                        uint32_t pos = count_atomic.fetch_add(1);
                        row_ids_gpu[pos] = i;
                    }
                );
            }
        );

        auto e_count_copy = copy_device_queues[device_index].memcpy(
            selected_count_host,
            selected_count_device,
            sizeof(uint32_t),
            e_build
        );

        auto e_row_ids_copy = copy_device_queues[device_index].memcpy(
            row_ids_host,
            row_ids_gpu,
            segment_size * sizeof(int),
            e_build
        );

        return {
            row_ids_gpu,
            row_ids_host,
            selected_count_device,
            selected_count_host,
            e_count_copy,
            e_row_ids_copy
        };
    }

    // This is a sync point
    void compress_and_sync(memory_manager &cpu_allocator, memory_manager &device_allocator, int device_index, const std::vector<Column *> &columns_to_sync = {})
    {
        if (device_index < 0 || device_index >= device_queues.size())
        {
            std::cerr << "compress_and_sync: invalid device index " << device_index << std::endl;
            throw std::runtime_error("compress_and_sync: invalid device index.");
        }

        int num_segments = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        ensure_primary_host_flags(cpu_allocator);
        auto dependencies = execute_pending_kernels();
        std::vector<sycl::event> cpu_sync_events;
        cpu_sync_events.reserve(num_segments * 2);

        for (int i = 0; i < num_segments; i++)
        {
            std::optional<AsyncRowIds> row_ids;
            uint64_t segment_size = (i == num_segments - 1) ? (nrows - i * SEGMENT_SIZE) : SEGMENT_SIZE;
            sycl::event e_last_cpu_sync;

            if (flags_modified_devices[device_index][i])
            {
                row_ids = build_row_ids(
                    i,
                    segment_size,
                    device_allocator,
                    device_index,
                    dependencies.second[device_index]
                );
            }

            for (const Column *col : columns_to_sync)
            {
                Segment &seg = const_cast<Segment &>(col->get_segments()[i]);
                if (seg.is_on_device(device_index) && seg.needs_copy_on(false, -1))
                {
                    if (!row_ids.has_value())
                        row_ids = build_row_ids(i, segment_size, device_allocator, device_index, dependencies.second[device_index]);

                    cpu_sync_events.push_back(
                        seg.compress_sync(
                            row_ids->row_ids_device,
                            row_ids->row_ids_host,
                            row_ids->selected_count_device,
                            row_ids->selected_count_host,
                            row_ids->selected_count_copy_event,
                            row_ids->row_ids_copy_event,
                            segment_size,
                            device_allocator,
                            device_index,
                            copy_device_queues[device_index]
                        )
                    );
                }
            }

            if (flags_modified_devices[device_index][i])
            {
                bool *flags = flags_host + i * SEGMENT_SIZE;
                int *row_ids_host = row_ids->row_ids_host;
                uint32_t *selected_count_host = row_ids->selected_count_host;
                sycl::event selected_count_copy_event = row_ids->selected_count_copy_event;
                sycl::event row_ids_copy_event = row_ids->row_ids_copy_event;
                auto e_flags_reset = cpu_queue.fill(
                    flags,
                    false,
                    segment_size,
                    dependencies.first
                );

                e_last_cpu_sync = cpu_queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        cgh.depends_on(e_flags_reset);
                        cgh.depends_on(selected_count_copy_event);
                        cgh.depends_on(row_ids_copy_event);
                        cgh.parallel_for(
                            segment_size,
                            [=](sycl::id<1> idx)
                            {
                                auto selected = static_cast<uint64_t>(idx[0]);
                                if (selected < *selected_count_host)
                                    flags[row_ids_host[selected]] = true;
                            }
                        );
                    }
                );

                flags_modified_devices[device_index][i] = false;
            }

            if (e_last_cpu_sync != sycl::event())
                cpu_sync_events.push_back(e_last_cpu_sync);
        }

        pending_kernels_dependencies_cpu = std::move(cpu_sync_events);
        for (int d = 0; d < device_queues.size(); d++)
        {
            if (d == device_index)
                pending_kernels_dependencies_devices[d].clear();
            else
                pending_kernels_dependencies_devices[d] = std::move(dependencies.second[d]);
        }
    }

    std::tuple<bool *, int, int> build_keys_hash_table(int column, memory_manager &cpu_allocator, memory_manager &device_allocator, bool on_device, int device_index)
    {
        // Assumption: flags do not need to be synced before building hash table

        auto ht_res = current_columns[column]->build_keys_hash_table(
            (on_device ? flags_devices[device_index] : flags_host),
            (on_device ? device_allocator : cpu_allocator),
            on_device,
            device_index
        );

        std::vector<KernelBundle> ht_kernels = std::get<3>(ht_res);
        add_pending_kernels(ht_kernels);

        return std::make_tuple(
            std::get<0>(ht_res),
            std::get<1>(ht_res),
            std::get<2>(ht_res)
        );
    }

    std::tuple<int *, int, int> build_key_vals_hash_table(int column, bool on_device, int device_index, memory_manager &cpu_allocator, memory_manager &device_allocator)
    {
        // Assumption: flags do not need to be synced before building hash table

        auto ht_res = current_columns[column]->build_key_vals_hash_table(
            group_by_column,
            (on_device ? flags_devices[device_index] : flags_host),
            (on_device ? device_allocator : cpu_allocator),
            on_device,
            device_index
        );

        std::vector<KernelBundle> ht_kernels = std::get<3>(ht_res);
        add_pending_kernels(ht_kernels);

        return std::make_tuple(
            std::get<0>(ht_res),
            std::get<1>(ht_res),
            std::get<2>(ht_res)
        );
    }

    void apply_filter(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        // Recursive parsing of EXPR types. LITERAL and COLUMN are handled in parent EXPR type.
        if (expr.exprType != ExprOption::EXPR)
        {
            std::cerr << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
            return;
        }

        std::vector<KernelBundle> ops;

        if (expr.op == "SEARCH")
        {
            const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();
            bool host_needed = false;
            std::vector<bool> devices_needed(device_queues.size(), false);

            for (const Segment &segment : segments)
            {
                if (segment.is_on_device())
                    devices_needed[segment.get_device_index()] = true;
                else
                    host_needed = true;
            }

            ensure_secondary_flag_buffers(
                host_needed,
                devices_needed,
                cpu_allocator,
                device_allocators
            );

            if (host_needed)
                ensure_host_state_available(
                    { current_columns[expr.operands[0].input] },
                    cpu_allocator,
                    device_allocators
                );

            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                std::vector<bool *> segments_flags_devices_input;
                std::vector<bool *> segments_flags_devices_output;
                segments_flags_devices_input.reserve(device_queues.size());
                segments_flags_devices_output.reserve(device_queues.size());
                std::transform(
                    flags_devices.begin(),
                    flags_devices.end(),
                    std::back_inserter(segments_flags_devices_input),
                    [segment_number](bool *flags_device)
                    {
                        return flags_device == nullptr ? nullptr : flags_device + segment_number * SEGMENT_SIZE;
                    }
                );
                std::transform(
                    flags_devices_secondary.begin(),
                    flags_devices_secondary.end(),
                    std::back_inserter(segments_flags_devices_output),
                    [segment_number](bool *flags_device)
                    {
                        return flags_device == nullptr ? nullptr : flags_device + segment_number * SEGMENT_SIZE;
                    }
                );
                KernelBundle bundle = segment.search_operator(
                    expr,
                    parent_op,
                    cpu_allocator,
                    device_allocators,
                    flags_host + segment_number * SEGMENT_SIZE,
                    flags_host_secondary == nullptr ? nullptr : flags_host_secondary + segment_number * SEGMENT_SIZE,
                    segments_flags_devices_input,
                    segments_flags_devices_output
                );

                if (bundle.is_on_device())
                {
                    int device_index = bundle.get_device_index();
                    flags_modified_devices[device_index][segment_number] = true;
                }
                else
                    flags_modified_host[segment_number] = true;

                ops.push_back(bundle);
            }

            add_pending_kernels(ops);
            swap_flag_buffers(ops);
        }
        else if (is_filter_logical(expr.op))
        {
            // Logical operation between other expressions. Pass parent op to the first then use the current op.
            // TODO: check if passing parent logic is correct in general
            bool parent_op_used = false;
            for (const ExprType &operand : expr.operands)
            {
                apply_filter(
                    operand,
                    parent_op_used ? expr.op : parent_op,
                    cpu_allocator,
                    device_allocators
                );
                parent_op_used = true;
            }
        }
        else
        {
            if (expr.operands.size() != 2)
            {
                std::cerr << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
                return;
            }

            Column *cols[2];
            bool literal = false;
            int literal_value;

            for (int i = 0; i < 2; i++)
            {
                switch (expr.operands[i].exprType)
                {
                case ExprOption::COLUMN:
                    cols[i] = current_columns[expr.operands[i].input];
                    break;
                case ExprOption::LITERAL:
                    literal = true;
                    literal_value = expr.operands[i].literal.value;
                    break;
                default:
                    std::cerr << "Filter condition: Unsupported parsing ExprType "
                        << expr.operands[i].exprType
                        << " for comparison operand"
                        << std::endl;
                    return;
                }
            }

            const std::vector<Segment> &segments = cols[0]->get_segments();
            bool host_needed = false;
            std::vector<bool> devices_needed(device_queues.size(), false);
            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                bool on_device =
                    segment.is_on_device() &&
                    (literal ||
                        (cols[1]->get_segments()[segment_number].is_on_device()
                            && cols[1]->get_segments()[segment_number].get_device_index() == segment.get_device_index()
                            ));
                int device_index = segment.get_device_index();
                if (on_device)
                    devices_needed[device_index] = true;
                else
                    host_needed = true;
            }

            ensure_secondary_flag_buffers(
                host_needed,
                devices_needed,
                cpu_allocator,
                device_allocators
            );

            if (host_needed)
            {
                std::vector<Column *> columns_to_sync = { cols[0] };
                if (!literal)
                    columns_to_sync.push_back(cols[1]);
                ensure_host_state_available(columns_to_sync, cpu_allocator, device_allocators);
            }

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                bool on_device =
                    segment.is_on_device() &&
                    (literal ||
                        (cols[1]->get_segments()[segment_number].is_on_device()
                            && cols[1]->get_segments()[segment_number].get_device_index() == segment.get_device_index()
                            ));
                int device_index = segment.get_device_index();
                KernelBundle bundle(on_device, device_index);

                bundle.add_kernel(
                    KernelData(
                        literal ? KernelType::SelectionKernelLiteral : KernelType::SelectionKernelColumns,
                        literal ?
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                literal_value,
                                on_device,
                                (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                (on_device ? flags_devices_secondary[device_index] : flags_host_secondary) + segment_number * SEGMENT_SIZE,
                                device_index
                            )
                            ) :
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                cols[1]->get_segments()[segment_number],
                                on_device,
                                (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                (on_device ? flags_devices_secondary[device_index] : flags_host_secondary) + segment_number * SEGMENT_SIZE,
                                device_index
                            )
                            )
                    )
                );

                if (on_device)
                    flags_modified_devices[device_index][segment_number] = true;
                else
                    flags_modified_host[segment_number] = true;

                ops.push_back(bundle);
            }

            add_pending_kernels(ops);
            swap_flag_buffers(ops);
        }
    }

    void apply_project(
        const std::vector<ExprType> &exprs,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        std::vector<Column *> new_columns;
        new_columns.reserve(exprs.size() + 50);

        for (size_t i = 0; i < exprs.size(); i++)
        {
            const ExprType &expr = exprs[i];
            switch (expr.exprType)
            {
            case ExprOption::COLUMN:
            {
                new_columns.push_back(current_columns[expr.input]);
                if (expr.input == group_by_column_index)
                    group_by_column_index = i;
                break;
            }
            case ExprOption::LITERAL:
            {
                int literal_value = (int)expr.literal.value;
                Column &new_col = register_materialized_column(materialized_columns.emplace_back(
                    nrows,
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators
                ));

                // TODO better way

                std::vector<KernelBundle> fill_bundles_cpu = new_col.fill_with_literal(literal_value, false, -1, cpu_allocator);
                add_pending_kernels(fill_bundles_cpu);

                // std::vector<KernelBundle> fill_bundles_gpu = new_col.fill_with_literal(literal_value, true, 0, device_allocators[0]);
                // pending_kernels.push_back(fill_bundles_gpu);

                new_columns.push_back(&new_col);
                break;
            }
            case ExprOption::EXPR:
            {
                if (expr.operands.size() != 2)
                {
                    std::cerr << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                    return;
                }

                // TODO: pass the correct allocator in order to do host malloc, to speed up transfers.
                Column &new_col = register_materialized_column(materialized_columns.emplace_back(
                    nrows,
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators
                ));

                std::vector<KernelBundle> ops;

                std::vector<Segment> &segments_result = new_col.get_segments();
                ops.reserve(segments_result.size());

                if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments_a = current_columns[expr.operands[0].input]->get_segments();
                    const std::vector<Segment> &segments_b = current_columns[expr.operands[1].input]->get_segments();

                    if (segments_a.size() != segments_b.size())
                    {
                        std::cerr << "Project operation: Mismatched segment sizes between columns" << std::endl;
                        return;
                    }

                    bool host_needed = false;
                    for (size_t segment_number = 0; segment_number < segments_a.size(); segment_number++)
                    {
                        const Segment &segment_a = segments_a[segment_number];
                        const Segment &segment_b = segments_b[segment_number];
                        bool on_device = segment_a.is_on_device() && segment_b.is_on_device()
                            && segment_a.get_device_index() == segment_b.get_device_index();
                        if (!on_device)
                        {
                            host_needed = true;
                            break;
                        }
                    }

                    if (host_needed)
                        ensure_host_state_available(
                            { current_columns[expr.operands[0].input], current_columns[expr.operands[1].input] },
                            cpu_allocator,
                            device_allocators
                        );

                    for (size_t segment_number = 0; segment_number < segments_a.size(); segment_number++)
                    {
                        const Segment &segment_a = segments_a[segment_number];
                        const Segment &segment_b = segments_b[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        bool on_device = segment_a.is_on_device() && segment_b.is_on_device()
                            && segment_a.get_device_index() == segment_b.get_device_index();
                        int device_index = segment_a.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelColumns,
                                segment_result.perform_operator(
                                    segment_a,
                                    segment_b,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::LITERAL &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[1].input]->get_segments();

                    bool host_needed = false;
                    for (const Segment &segment : segments)
                    {
                        if (!segment.is_on_device())
                        {
                            host_needed = true;
                            break;
                        }
                    }

                    if (host_needed)
                        ensure_host_state_available(
                            { current_columns[expr.operands[1].input] },
                            cpu_allocator,
                            device_allocators
                        );

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        bool on_device = segment.is_on_device();
                        int device_index = segment.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralFirst,
                                segment_result.perform_operator(
                                    (int)expr.operands[0].literal.value,
                                    segment,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::LITERAL)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();

                    bool host_needed = false;
                    for (const Segment &segment : segments)
                    {
                        if (!segment.is_on_device())
                        {
                            host_needed = true;
                            break;
                        }
                    }

                    if (host_needed)
                        ensure_host_state_available(
                            { current_columns[expr.operands[0].input] },
                            cpu_allocator,
                            device_allocators
                        );

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        bool on_device = segment.is_on_device();
                        int device_index = segment.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralSecond,
                                segment_result.perform_operator(
                                    segment,
                                    (int)expr.operands[1].literal.value,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else
                {
                    std::cerr << "Project operation: Unsupported parsing ExprType "
                        << expr.operands[0].exprType << " and "
                        << expr.operands[1].exprType
                        << " for EXPR" << std::endl;
                    return;
                }

                add_pending_kernels(ops);
                new_columns.push_back(&new_col);
                break;
            }
            }
        }

        current_columns = new_columns;
    }

    void apply_aggregate(
        const AggType &agg,
        const std::vector<long> &group,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        std::vector<KernelBundle> agg_bundles;

        if (group.size() == 0)
        {
            const std::vector<Segment> &input_segments = current_columns[agg.operands[0]]->get_segments();

            bool on_device = current_columns[agg.operands[0]]->is_all_on_same_device(),
                need_sync = false;
            int device_index = input_segments[0].get_device_index();

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying aggregate on "
                << (on_device ? "GPU" : "CPU")
                << " with " << input_segments.size() << " segments." << std::endl;
            #endif

            if (on_device)
            {
                for (bool modified : flags_modified_host)
                {
                    if (modified)
                    {
                        // need_sync = true;
                        // break;
                        std::cerr << "Error: aggregation flags on CPU to GPU modified not implemented yet" << std::endl;
                        throw std::runtime_error("Flags on CPU to GPU modified not implemented yet.");
                    }
                }
                for (int d = 0; d < device_queues.size(); d++)
                {
                    if (need_sync)
                        break;
                    if (d == device_index)
                        continue;
                    for (bool modified : flags_modified_devices[d])
                    {
                        if (modified)
                        {
                            // need_sync = true;
                            // break;
                            std::cerr << "Error: aggregation flags on other GPU modified not implemented yet" << std::endl;
                            throw std::runtime_error("Flags on other GPU modified not implemented yet.");
                        }
                    }
                }
            }
            else
            {
                for (const auto &flags_modified_gpu : flags_modified_devices)
                {
                    for (bool modified : flags_modified_gpu)
                    {
                        if (modified)
                        {
                            need_sync = true;
                            break;
                        }
                    }
                    if (need_sync)
                        break;
                }
            }

            if (need_sync)
            {
                for (int d = 0; d < device_queues.size(); d++)
                    compress_and_sync(
                        cpu_allocator,
                        device_allocators[d],
                        d
                    );
            }
            uint64_t *final_result = (on_device ?
                device_allocators[device_index].alloc_zero<uint64_t>(1) :
                cpu_allocator.alloc_zero<uint64_t>(1)
                );

            agg_bundles.reserve(input_segments.size());

            for (int i = 0; i < input_segments.size(); i++)
            {
                const Segment &input_segment = input_segments[i];
                KernelBundle bundle(on_device, device_index);
                bundle.add_kernel(
                    KernelData(
                        KernelType::AggregateOperationKernel,
                        input_segment.aggregate_operator(
                            (on_device ? flags_devices[device_index] : flags_host) + i * SEGMENT_SIZE,
                            on_device,
                            device_index,
                            final_result
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            add_pending_kernels(agg_bundles);

            auto dependencies = execute_pending_kernels();
            pending_kernels_dependencies_cpu = dependencies.first;
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d] = dependencies.second[d];
                flags_devices_secondary[d] = nullptr;

                if (on_device && d == device_index)
                {
                    bool *new_flags = device_allocators[d].alloc<bool>(1, true);
                    pending_kernels_dependencies_devices[d].push_back(
                        device_queues[d].fill<bool>(new_flags, true, 1)
                    );
                    flags_devices[d] = new_flags;
                }
                else
                    flags_devices[d] = nullptr;

                flags_modified_devices[d] = { false };
            }

            if (on_device)
                flags_host = nullptr;
            else
            {
                bool *new_cpu_flags = cpu_allocator.alloc<bool>(1, false);
                new_cpu_flags[0] = true;
                flags_host = new_cpu_flags;
            }
            flags_host_secondary = nullptr;
            flags_modified_host = { false };
            if (on_device)
                flags_modified_devices[device_index] = { true };

            nrows = 1;

            Column &result_column = register_materialized_column(materialized_columns.emplace_back(
                final_result,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                nrows
            ));

            current_columns.clear();
            current_columns.push_back(&result_column);
        }
        else
        {
            const Column *agg_column = current_columns[agg.operands[0]];
            bool on_device = agg_column->is_all_on_same_device(),
                need_sync = false;
            int device_index = agg_column->get_segments()[0].get_device_index();
            for (int i = 0; i < group.size() && on_device; i++)
            {
                on_device = current_columns[group[i]]->is_all_on_same_device() &&
                    current_columns[group[i]]->get_segments()[0].get_device_index() == device_index;
            }

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying group-by aggregate on "
                << (on_device ? "GPU" : "CPU") << std::endl;
            #endif

            if (on_device)
            {
                for (bool modified : flags_modified_host)
                {
                    if (modified)
                    {
                        // need_sync = true;
                        // break;
                        std::cerr << "Error: aggregation flags on CPU to GPU modified not implemented yet" << std::endl;
                        throw std::runtime_error("Flags on CPU to GPU modified not implemented yet.");
                    }
                }
                for (int d = 0; d < device_queues.size(); d++)
                {
                    if (need_sync)
                        break;
                    if (d == device_index)
                        continue;
                    for (bool modified : flags_modified_devices[d])
                    {
                        if (modified)
                        {
                            // need_sync = true;
                            // break;
                            std::cerr << "Error: aggregation flags on other GPU modified not implemented yet" << std::endl;
                            throw std::runtime_error("Flags on other GPU modified not implemented yet.");
                        }
                    }
                }
            }
            else
            {
                for (const auto &flags_modified_gpu : flags_modified_devices)
                {
                    for (bool modified : flags_modified_gpu)
                    {
                        if (modified)
                        {
                            need_sync = true;
                            break;
                        }
                    }
                    if (need_sync)
                        break;
                }
            }

            need_sync = need_sync || agg_column->needs_copy_on(on_device, device_index);
            for (int i = 0; i < group.size() && !need_sync; i++)
                need_sync = current_columns[group[i]]->needs_copy_on(on_device, device_index);


            if (need_sync)
            {
                std::vector<Column *> columns_to_sync;
                columns_to_sync.reserve(group.size() + 1);
                columns_to_sync.push_back(const_cast<Column *>(agg_column));
                for (int i = 0; i < group.size(); i++)
                    columns_to_sync.push_back(current_columns[group[i]]);
                for (int d = 0; d < device_queues.size(); d++)
                    compress_and_sync(
                        cpu_allocator,
                        device_allocators[d],
                        d,
                        columns_to_sync
                    );
            }

            memory_manager &allocator = on_device ? device_allocators[device_index] : cpu_allocator;

            uint64_t prod_ranges = 1;
            int *min = allocator.alloc<int>(group.size(), !on_device),
                *max = allocator.alloc<int>(group.size(), !on_device);

            for (int i = 0; i < group.size(); i++)
            {
                auto min_max = current_columns[group[i]]->get_min_max();
                min[i] = min_max.first;
                max[i] = min_max.second;
                prod_ranges *= max[i] - min[i] + 1;
            }

            uint64_t *aggregate_result = allocator.alloc_zero<uint64_t>(prod_ranges);
            unsigned *temp_flags = allocator.alloc_zero<unsigned>(prod_ranges);
            int **results = allocator.alloc<int *>(group.size(), !on_device);

            for (int i = 0; i < group.size(); i++)
                results[i] = allocator.alloc<int>(prod_ranges, true);

            const std::vector<Segment> &agg_segments = agg_column->get_segments();

            agg_bundles.reserve(agg_segments.size());

            for (int i = 0; i < agg_segments.size(); i++)
            {
                const int **contents = allocator.alloc<const int *>(group.size(), !on_device);
                for (int j = 0; j < group.size(); j++)
                {
                    const Segment &segment = current_columns[group[j]]->get_segments()[i];
                    contents[j] = segment.get_data(on_device, device_index);
                }
                const Segment &agg_segment = agg_segments[i];

                KernelBundle bundle(on_device, device_index);
                bundle.add_kernel(
                    KernelData(
                        KernelType::GroupByAggregateKernel,
                        agg_segment.group_by_aggregate_operator(
                            contents,
                            max,
                            min,
                            (on_device ? flags_devices[device_index] : flags_host) + i * SEGMENT_SIZE,
                            aggregate_result,
                            group.size(),
                            results,
                            temp_flags,
                            on_device,
                            device_index,
                            prod_ranges
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            add_pending_kernels(agg_bundles);

            // std::cout << "Executing aggregate kernels" << std::endl;
            auto dependencies = execute_pending_kernels();

            bool *new_cpu_flags = nullptr;
            if (on_device)
                flags_host = nullptr;
            else
            {
                new_cpu_flags = cpu_allocator.alloc<bool>(prod_ranges, false);
                flags_host = new_cpu_flags;
            }
            flags_host_secondary = nullptr;

            nrows = prod_ranges;

            uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

            if (on_device)
            {
                bool *new_gpu_flags = device_allocators[device_index].alloc<bool>(prod_ranges, true);
                auto e1 = device_queues[device_index].submit(
                    [&](sycl::handler &cgh)
                    {
                        if (!dependencies.second[device_index].empty())
                            cgh.depends_on(dependencies.second[device_index]);
                        cgh.parallel_for(
                            prod_ranges,
                            [=](sycl::id<1> idx)
                            {
                                new_gpu_flags[idx[0]] = temp_flags[idx[0]] != 0;
                            }
                        );
                    }
                );

                flags_devices[device_index] = new_gpu_flags;
                for (int d = 0; d < device_queues.size(); d++)
                    flags_devices_secondary[d] = nullptr;
                pending_kernels_dependencies_devices[device_index].push_back(e1);
                for (int d = 0; d < device_queues.size(); d++)
                {
                    // if (d != device_index)
                    // {
                    //     bool *new_flags = device_allocators[d].alloc<bool>(prod_ranges, true);
                    //     pending_kernels_dependencies_devices[d].push_back(
                    //         device_queues[d].memcpy(
                    //             new_flags,
                    //             new_gpu_flags,
                    //             sizeof(bool) * prod_ranges,
                    //             e1
                    //         )
                    //     );
                    //     flags_devices[d] = new_flags;
                    // }
                    flags_modified_devices[d].resize(segment_num, false);
                    std::fill(
                        flags_modified_devices[d].begin(),
                        flags_modified_devices[d].end(),
                        d == device_index
                    );
                }
            }
            else
            {
                for (int d = 0; d < device_queues.size(); d++)
                    flags_devices_secondary[d] = nullptr;
                pending_kernels_dependencies_cpu.push_back(
                    cpu_queue.submit(
                        [&](sycl::handler &cgh)
                        {
                            if (!dependencies.first.empty())
                                cgh.depends_on(dependencies.first);
                            cgh.parallel_for(
                                prod_ranges,
                                [=](sycl::id<1> idx)
                                {
                                    new_cpu_flags[idx[0]] = temp_flags[idx[0]] != 0;
                                }
                            );
                        }
                    )
                );

                // gpu update skipped since after aggregation on cpu, nothing is run on gpu
            }

            current_columns.clear();

            for (int i = 0; i < group.size(); i++)
            {
                Column &new_col = register_materialized_column(materialized_columns.emplace_back(
                    results[i],
                    on_device,
                    device_index,
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators,
                    prod_ranges
                ));
                current_columns.push_back(&new_col);
            }

            Column &agg_col = register_materialized_column(materialized_columns.emplace_back(
                aggregate_result,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                prod_ranges
            ));
            current_columns.push_back(&agg_col);

            flags_modified_host.resize(segment_num, false);
            std::fill(
                flags_modified_host.begin(),
                flags_modified_host.end(),
                false
            );
        }
    }

    void apply_join(
        TransientTable &right_table,
        const RelNode &rel,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        int left_column = rel.condition.operands[0].input,
            right_column = rel.condition.operands[1].input - current_columns.size();

        if (left_column < 0 ||
            left_column >= current_columns.size() ||
            right_column < 0 ||
            right_column >= right_table.current_columns.size())
        {
            std::cerr << "Join operation: Invalid column indices in join condition: " << left_column << "/" << current_columns.size() << " and " << right_column << "/" << right_table.current_columns.size() << " ( " << rel.condition.operands[1].input << " )" << std::endl;
            throw std::invalid_argument("Invalid column indices in join condition.");
        }

        if (rel.joinType == "semi")
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying semi-join" << std::endl;
            #endif

            bool *ht_cpu = nullptr;
            std::vector<bool *> ht_devices(device_queues.size(), nullptr);
            int build_min_value, build_max_value;

            for (const Segment &seg : current_columns[left_column]->get_segments())
            {
                int device_index = seg.get_device_index();
                if (seg.is_on_device() && ht_devices[device_index] == nullptr)
                {
                    auto ht_data = right_table.build_keys_hash_table(
                        right_column,
                        cpu_allocator,
                        device_allocators[device_index],
                        true,
                        device_index
                    );
                    ht_devices[device_index] = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
                else if (!seg.is_on_device() && ht_cpu == nullptr)
                {
                    right_table.ensure_host_state_available(
                        { right_table.current_columns[right_column] },
                        cpu_allocator,
                        device_allocators
                    );
                    auto ht_data = right_table.build_keys_hash_table(
                        right_column,
                        cpu_allocator,
                        device_allocators[0],
                        false,
                        -1
                    );
                    ht_cpu = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
            }

            bool host_needed = false;
            std::vector<bool> devices_needed(device_queues.size(), false);
            for (const Segment &seg : current_columns[left_column]->get_segments())
            {
                int device_index = seg.get_device_index();
                if (seg.is_on_device() && ht_devices[device_index] != nullptr)
                    devices_needed[device_index] = true;
                else
                    host_needed = true;
            }

            ensure_secondary_flag_buffers(
                host_needed,
                devices_needed,
                cpu_allocator,
                device_allocators
            );

            if (host_needed)
                ensure_host_flags_available(cpu_allocator, device_allocators);

            auto ht_dependencies = right_table.execute_pending_kernels();

            pending_kernels_dependencies_cpu.insert(
                pending_kernels_dependencies_cpu.end(),
                ht_dependencies.first.begin(),
                ht_dependencies.first.end()
            );
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d].insert(
                    pending_kernels_dependencies_devices[d].end(),
                    ht_dependencies.second[d].begin(),
                    ht_dependencies.second[d].end()
                );
            }

            auto join_ops = current_columns[left_column]->semi_join(
                flags_host,
                flags_host_secondary,
                flags_devices,
                flags_devices_secondary,
                build_min_value,
                build_max_value,
                ht_cpu,
                ht_devices,
                device_allocators,
                flags_modified_host,
                flags_modified_devices
            );
            add_pending_kernels(join_ops);
            swap_flag_buffers(join_ops);

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);
        }
        else
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying full join" << std::endl;
            #endif

            auto probe_col_locations = current_columns[left_column]->get_positions();

            bool can_probe_on_device = false, can_build_on_device = false;
            for (bool on_device : probe_col_locations.second)
                can_probe_on_device = can_probe_on_device || on_device;

            auto col_devices = right_table.current_columns[right_column]->get_full_col_on_device(),
                group_by_col_devices = right_table.group_by_column->get_full_col_on_device();

            for (int d = 0; d < device_queues.size(); d++)
            {
                col_devices[d] = col_devices[d] && group_by_col_devices[d];
                can_build_on_device = can_build_on_device || col_devices[d];
            }

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Join hash table will be built on "
                << (can_build_on_device && can_probe_on_device ? "GPU" : "CPU")
                << (can_build_on_device && can_probe_on_device && probe_col_locations.first ? " and CPU" : "")
                << std::endl;
            #endif

            Column &new_column = register_materialized_column(materialized_columns.emplace_back(
                nrows,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                true
            ));

            auto min_max_gb = right_table.group_by_column->get_min_max();
            uint64_t group_by_col_index = current_columns.size() + right_table.group_by_column_index;

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);

            current_columns[group_by_col_index] = &materialized_columns[materialized_columns.size() - 1];

            int build_min_value, build_max_value, *ht_host = nullptr;
            std::vector<int *> ht_devices(device_queues.size(), nullptr);

            for (int d = 0; d < device_queues.size(); d++)
            {
                if (probe_col_locations.second[d] && col_devices[d])
                {
                    auto ht_data = right_table.build_key_vals_hash_table(
                        right_column,
                        true,
                        d,
                        cpu_allocator,
                        device_allocators[d]
                    );
                    ht_devices[d] = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
            }

            // Here we don't check if all probe segments have compatible device locations with build
            // Assumed all probe segments on a device have the full build columns on device too
            if (probe_col_locations.first)
            {
                right_table.ensure_host_state_available(
                    { right_table.current_columns[right_column], right_table.group_by_column },
                    cpu_allocator,
                    device_allocators
                );
                auto ht_data = right_table.build_key_vals_hash_table(
                    right_column,
                    false,
                    -1,
                    cpu_allocator,
                    device_allocators[0]
                );
                ht_host = std::get<0>(ht_data);
                build_min_value = std::get<1>(ht_data);
                build_max_value = std::get<2>(ht_data);
            }

            bool host_needed = false;
            std::vector<bool> devices_needed(device_queues.size(), false);
            for (const Segment &seg : current_columns[left_column]->get_segments())
            {
                bool built = false;
                if (seg.is_on_device())
                {
                    const std::vector<bool> &on_device_vec = seg.get_on_device_vec();
                    for (int d = 0; d < device_queues.size(); d++)
                    {
                        if (on_device_vec[d] && ht_devices[d] != nullptr)
                        {
                            devices_needed[d] = true;
                            built = true;
                            break;
                        }
                    }
                }

                if (!built)
                    host_needed = true;
            }

            ensure_secondary_flag_buffers(
                host_needed,
                devices_needed,
                cpu_allocator,
                device_allocators
            );

            if (host_needed)
                ensure_host_flags_available(cpu_allocator, device_allocators);

            auto ht_dependencies = right_table.execute_pending_kernels();

            pending_kernels_dependencies_cpu.insert(
                pending_kernels_dependencies_cpu.end(),
                ht_dependencies.first.begin(),
                ht_dependencies.first.end()
            );
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d].insert(
                    pending_kernels_dependencies_devices[d].end(),
                    ht_dependencies.second[d].begin(),
                    ht_dependencies.second[d].end()
                );
            }

            auto join_ops = current_columns[left_column]->full_join_operation(
                flags_host,
                flags_host_secondary,
                flags_devices,
                flags_devices_secondary,
                build_min_value,
                build_max_value,
                min_max_gb.first,
                min_max_gb.second,
                ht_host,
                ht_devices,
                new_column,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                flags_modified_host,
                flags_modified_devices
            );

            add_pending_kernels(join_ops);
            swap_flag_buffers(join_ops);
        }
    }
};

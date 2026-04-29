#include "application.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <thrift/Thrift.h>
#include <thrift/transport/TTransportException.h>

#include "benchmark/ssb_suite.hpp"
#include "common.hpp"
#include "executor/ddor_executor.hpp"
#include "options.hpp"
#include "planner_client.hpp"
#include "query_utils.hpp"
#include "runtime/runtime.hpp"

namespace app
{

namespace
{

struct QueryRunSummary
{
    std::vector<double> engine_ms;
    std::vector<double> total_ms;
    std::vector<double> jit_ms;
    std::vector<double> kernel_ms;
    std::vector<double> load_ms;
    std::vector<double> parse_ms;

    double warm_average_total_ms() const
    {
        if (total_ms.empty())
            return 0.0;
        if (total_ms.size() == 1)
            return total_ms.front();

        double sum = 0.0;
        for (size_t i = 1; i < total_ms.size(); ++i)
            sum += total_ms[i];
        return sum / static_cast<double>(total_ms.size() - 1);
    }
};

std::string resolve_benchmark_device_name(
    const AppOptions &options,
    const runtime_setup::RuntimeEnvironment &runtime)
{
    if (!options.benchmark_device_name.empty())
        return options.benchmark_device_name;

    if (const char *selector = std::getenv("ONEAPI_DEVICE_SELECTOR"))
        return selector;

    if (!runtime.queues.device_queues.empty() && runtime.config.primary_device_index != -1)
        return runtime.queues.device_queues[runtime.config.primary_device_index]
            .get_device()
            .get_info<sycl::info::device::name>();

    return runtime.queues.cpu_queue.get_device().get_info<sycl::info::device::name>();
}

std::ofstream open_perf_log(const std::string &query_path)
{
    if (query_path.empty())
        return std::ofstream();

    return std::ofstream(
        query_stem_from_path(query_path) + "-performance-xpu.log",
        std::ios::out | std::ios::trunc
    );
}

QueryRunSummary run_query_repetitions(
    PlannerClient &planner,
    runtime_setup::RuntimeEnvironment &runtime,
    const std::string &query_path,
    const std::string &sql,
    std::ostream &perf_out)
{
    QueryRunSummary summary;
    summary.engine_ms.reserve(PERFORMANCE_REPETITIONS);
    summary.total_ms.reserve(PERFORMANCE_REPETITIONS);

    if (runtime.config.reuse_allocators_across_repetitions)
    {
        memory_manager cpu_allocator = runtime_setup::build_cpu_allocator(runtime);
        auto device_allocators = runtime_setup::build_device_allocators(runtime);

        for (int i = 0; i < PERFORMANCE_REPETITIONS; ++i)
        {
            PlanResult result;
            const auto parse_start = std::chrono::high_resolution_clock::now();
            planner.parse(result, sql);
            const auto parse_end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> parse_time = parse_end - parse_start;

            reset_jit_timers();
            const auto exec_time = executor::execute_ddor_plan(
                result,
                query_path,
                runtime.tables,
                runtime.queues,
                cpu_allocator,
                device_allocators,
                runtime.config.primary_device_index,
                perf_out
            );
            auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> total_time = end - parse_start;

            runtime_setup::wait_for_all_queues_and_throw(runtime.queues);
            auto reset_events = runtime_setup::reset_allocators(
                runtime,
                cpu_allocator,
                device_allocators
            );
            runtime_setup::wait_for_reset_events(reset_events);
            runtime_setup::wait_for_all_queues_and_throw(runtime.queues);

            summary.engine_ms.push_back(exec_time.count());
            summary.total_ms.push_back(total_time.count());
            summary.jit_ms.push_back(g_total_jit_ms);
            summary.kernel_ms.push_back(g_total_kernel_ms);
            summary.load_ms.push_back(g_total_load_ms);
            summary.parse_ms.push_back(parse_time.count());

            std::cout << "Repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS
                << " - " << exec_time.count() << " ms - "
                << total_time.count() << " ms - "
                << "JIT: " << g_total_jit_ms << " ms, Kernel: " << g_total_kernel_ms << " ms, Parse: " << parse_time.count() << " ms" << std::endl;
        }
    }
    else
    {
        memory_manager cpu_allocator = runtime_setup::build_cpu_allocator(runtime);
        auto device_allocators = runtime_setup::build_device_allocators(runtime);

        for (int i = 0; i < PERFORMANCE_REPETITIONS; ++i)
        {
            PlanResult result;
            const auto parse_start = std::chrono::high_resolution_clock::now();
            planner.parse(result, sql);
            const auto parse_end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> parse_time = parse_end - parse_start;

            reset_jit_timers();
            const auto exec_time = executor::execute_ddor_plan(
                result,
                query_path,
                runtime.tables,
                runtime.queues,
                cpu_allocator,
                device_allocators,
                runtime.config.primary_device_index,
                perf_out
            );
            auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> total_time = end - parse_start;

            summary.engine_ms.push_back(exec_time.count());
            summary.total_ms.push_back(total_time.count());
            summary.jit_ms.push_back(g_total_jit_ms);
            summary.kernel_ms.push_back(g_total_kernel_ms);
            summary.load_ms.push_back(g_total_load_ms);
            summary.parse_ms.push_back(parse_time.count());

            std::cout << "Repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS
                << " - " << exec_time.count() << " ms - "
                << total_time.count() << " ms - "
                << "JIT: " << g_total_jit_ms << " ms, Kernel: " << g_total_kernel_ms << " ms, Parse: " << parse_time.count() << " ms" << std::endl;

            // Reset allocators for next repetition
            runtime_setup::wait_for_all_queues_and_throw(runtime.queues);
            auto reset_events = runtime_setup::reset_allocators(runtime, cpu_allocator, device_allocators);
            runtime_setup::wait_for_reset_events(reset_events);
            runtime_setup::wait_for_all_queues_and_throw(runtime.queues);
        }
    }

    return summary;
}

int run_single_query(
    PlannerClient &planner,
    runtime_setup::RuntimeEnvironment &runtime,
    const AppOptions &options)
{
    const std::string sql = load_sql_text(options.query_path);
    std::ofstream perf_file = open_perf_log(options.query_path);
    std::ostringstream null_stream;
    std::ostream &perf_out = perf_file.is_open() ?
        static_cast<std::ostream &>(perf_file) :
        static_cast<std::ostream &>(null_stream);

    run_query_repetitions(
        planner,
        runtime,
        options.query_path,
        sql,
        perf_out
    );

    return 0;
}

int run_ssb_benchmark(
    PlannerClient &planner,
    runtime_setup::RuntimeEnvironment &runtime,
    const AppOptions &options)
{
    const std::vector<std::string> queries =
        bench::load_ssb_suite(options.benchmark_suite_path);
    const std::string device_name =
        resolve_benchmark_device_name(options, runtime);

    std::ofstream results(
        options.benchmark_results_path,
        options.append_results ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc)
    );
    if (!results.is_open())
        throw std::runtime_error("could not open benchmark results file: " + options.benchmark_results_path);

        results << "Query,Repetition,Device,TotalTime_ms,JIT_ms,Kernel_ms,Transfer_ms,Parse_ms,Other_ms,Disk_to_RAM_ms,RAM_to_GPU_ms\n";

    for (const std::string &query_path : queries)
    {
        std::cout << "Benchmarking " << query_path
            << " on " << device_name << std::endl;

        const std::string sql = load_sql_text(query_path);
        std::ostringstream perf_sink;
        const QueryRunSummary summary = run_query_repetitions(
            planner,
            runtime,
            query_path,
            sql,
            perf_sink
        );

        for (int i = 0; i < PERFORMANCE_REPETITIONS; ++i)
        {
            double jit = summary.jit_ms[i];
            double kernel = summary.kernel_ms[i];
            double load = summary.load_ms[i];
            double parse = summary.parse_ms[i];
            double total = summary.total_ms[i];
            double other = total - jit - kernel - load - parse;

            results << query_filename_from_path(query_path)
                << "," << i + 1
                << "," << device_name
                << "," << total
                << "," << jit
                << "," << kernel
                << "," << load
                << "," << parse
                << "," << other
                << "," << (i == 0 ? 6394.48 : 0.0)
                << "," << load
                << "\n";
        }
        results.flush();
    }
    results.close();

    return 0;
}

} // namespace

int run_app(int argc, char **argv)
{
    try
    {
        const AppOptions options = parse_app_options(argc, argv);
        if (options.mode == AppMode::Help)
        {
            print_help(std::cout, argv[0]);
            return 0;
        }

        runtime_setup::RuntimeEnvironment runtime =
            runtime_setup::build_runtime_environment();
        PlannerClient planner(options.planner_host, options.planner_port);
        planner.open();
        std::cout << "Transport opened successfully." << std::endl;
        planner.ping();

        if (options.mode == AppMode::BenchmarkSsb)
            return run_ssb_benchmark(planner, runtime, options);

        return run_single_query(planner, runtime, options);
    }
    catch (const apache::thrift::transport::TTransportException &e)
    {
        std::cerr << "Transport exception: " << e.what() << std::endl;
    }
    catch (const apache::thrift::TException &e)
    {
        std::cerr << "Thrift exception: " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 1;
}

} // namespace app

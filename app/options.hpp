#pragma once

#include <ostream>
#include <string>

namespace app
{

enum class AppMode
{
    SingleQuery,
    BenchmarkSsb,
    Help,
};

struct AppOptions
{
    AppMode mode = AppMode::SingleQuery;
    std::string query_path;
    std::string benchmark_suite_path = "benchmark/ssb_queries.txt";
    std::string benchmark_results_path = "benchmark_results_final.csv";
    std::string benchmark_device_name;
    std::string planner_host = "localhost";
    int planner_port = 5555;
    bool append_results = false;
};

AppOptions parse_app_options(int argc, char **argv);
void print_help(std::ostream &out, const char *program_name);

} // namespace app

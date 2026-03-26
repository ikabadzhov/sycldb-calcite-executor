#include "options.hpp"

#include <stdexcept>

namespace app
{

namespace
{

bool requires_value(int index, int argc)
{
    return index + 1 < argc;
}

} // namespace

AppOptions parse_app_options(int argc, char **argv)
{
    AppOptions options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            options.mode = AppMode::Help;
            return options;
        }

        if (arg == "--benchmark-ssb")
        {
            options.mode = AppMode::BenchmarkSsb;
            continue;
        }

        if (arg == "--suite")
        {
            if (!requires_value(i, argc))
                throw std::invalid_argument("--suite requires a value");
            options.benchmark_suite_path = argv[++i];
            continue;
        }

        if (arg == "--results")
        {
            if (!requires_value(i, argc))
                throw std::invalid_argument("--results requires a value");
            options.benchmark_results_path = argv[++i];
            continue;
        }

        if (arg == "--device-name")
        {
            if (!requires_value(i, argc))
                throw std::invalid_argument("--device-name requires a value");
            options.benchmark_device_name = argv[++i];
            continue;
        }

        if (arg == "--planner-host")
        {
            if (!requires_value(i, argc))
                throw std::invalid_argument("--planner-host requires a value");
            options.planner_host = argv[++i];
            continue;
        }

        if (arg == "--planner-port")
        {
            if (!requires_value(i, argc))
                throw std::invalid_argument("--planner-port requires a value");
            options.planner_port = std::stoi(argv[++i]);
            continue;
        }

        if (arg == "--append-results")
        {
            options.append_results = true;
            continue;
        }

        if (!arg.empty() && arg[0] == '-')
            throw std::invalid_argument("unknown option: " + arg);

        if (!options.query_path.empty())
            throw std::invalid_argument("only one query path may be provided");

        options.query_path = arg;
    }

    return options;
}

void print_help(std::ostream &out, const char *program_name)
{
    out
        << "Usage:\n"
        << "  " << program_name << " <query.sql>\n"
        << "  " << program_name << " --benchmark-ssb [options]\n"
        << "\nOptions:\n"
        << "  --benchmark-ssb          Run the SSB benchmark suite\n"
        << "  --suite <path>           Benchmark suite file (default: benchmark/ssb_queries.txt)\n"
        << "  --results <path>         Benchmark CSV output path\n"
        << "  --device-name <name>     Device label written into benchmark CSV rows\n"
        << "  --append-results         Append to the benchmark CSV instead of truncating it\n"
        << "  --planner-host <host>    Calcite planner host (default: localhost)\n"
        << "  --planner-port <port>    Calcite planner port (default: 5555)\n"
        << "  --help                   Show this help text\n";
}

} // namespace app

#include "ssb_suite.hpp"

#include <fstream>
#include <stdexcept>

namespace bench
{

std::vector<std::string> load_ssb_suite(const std::string &suite_path)
{
    std::ifstream input(suite_path);
    if (!input.is_open())
        throw std::runtime_error("could not open benchmark suite: " + suite_path);

    std::vector<std::string> queries;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        queries.push_back(line);
    }

    if (queries.empty())
        throw std::runtime_error("benchmark suite is empty: " + suite_path);

    return queries;
}

} // namespace bench

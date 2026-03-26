#include "query_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace app
{

std::string default_query_sql()
{
    return "select sum(lo_revenue)"
           " from lineorder, ddate, part, supplier"
           " where lo_orderdate = d_datekey"
           " and lo_partkey = p_partkey"
           " and lo_suppkey = s_suppkey;";
}

std::string load_sql_text(const std::string &path)
{
    if (path.empty())
        return default_query_sql();

    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("could not open SQL file: " + path);

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

std::string query_filename_from_path(const std::string &path)
{
    if (path.empty())
        return "ad_hoc.sql";

    return std::filesystem::path(path).filename().string();
}

std::string query_stem_from_path(const std::string &path)
{
    if (path.empty())
        return "ad_hoc";

    return std::filesystem::path(path).stem().string();
}

} // namespace app

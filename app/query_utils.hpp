#pragma once

#include <string>

namespace app
{

std::string default_query_sql();
std::string load_sql_text(const std::string &path);
std::string query_filename_from_path(const std::string &path);
std::string query_stem_from_path(const std::string &path);

} // namespace app

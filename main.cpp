#include "app/application.hpp"
#include "common.hpp"

double g_total_jit_ms = 0;
double g_total_kernel_ms = 0;
double g_total_load_ms = 0;
double g_total_disk_to_ram_ms = 0;
double g_total_sql_parse_ms = 0;

int main(int argc, char **argv)
{
    return app::run_app(argc, argv);
}

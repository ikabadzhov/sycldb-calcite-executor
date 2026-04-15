#pragma once

#define PERFORMANCE_MEASUREMENT_ACTIVE 1
#define PERFORMANCE_REPETITIONS 5

#define SIZE_TEMP_MEMORY_GPU (((uint64_t)32) << 30) // 32GB
#define SIZE_TEMP_MEMORY_CPU (((uint64_t)64) << 30) // 64GB

#define DATA_DIR "/media/ssb/s100_columnar/"

#define SEGMENT_SIZE (((uint64_t)1) << 28) // 256M elements (~1GB)

extern double g_total_jit_ms;
extern double g_total_kernel_ms;
extern double g_total_load_ms;
extern double g_total_sql_parse_ms;
inline void reset_jit_timers() { 
    g_total_jit_ms = 0; 
    g_total_kernel_ms = 0; 
    g_total_load_ms = 0;
    g_total_sql_parse_ms = 0;
}

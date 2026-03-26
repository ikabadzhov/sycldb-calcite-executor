#pragma once

#define PERFORMANCE_MEASUREMENT_ACTIVE 1
#define PERFORMANCE_REPETITIONS 5

#define SIZE_TEMP_MEMORY_GPU (((uint64_t)8) << 30) // 8GB
#define SIZE_TEMP_MEMORY_CPU (((uint64_t)20) << 30) // 20GB

#define DATA_DIR "/media/ssb/s100_columnar/"

#define SEGMENT_SIZE (((uint64_t)1) << 28) // 256M elements (~1GB)

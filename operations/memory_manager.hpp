#pragma once

#include <sycl/sycl.hpp>

#define MEMORY_MANAGER_DEBUG_INFO 0

class memory_region
{
private:
    void *memory_ptr, *current_free;
    uint64_t size, allocated;
    sycl::queue &queue;
    bool zero;
public:
    memory_region(sycl::queue &queue, uint64_t size, bool on_device, bool zero);
    ~memory_region();

    template <typename T>
    bool can_alloc(uint64_t count) const;

    template <typename T>
    T *alloc(uint64_t count);

    sycl::event reset();
};

template <typename T>
inline bool memory_region::can_alloc(uint64_t count) const
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes
    return (allocated + bytes <= size);
}

template <typename T>
T *memory_region::alloc(uint64_t count)
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory manager allocating " << bytes << " bytes. "
        << size << " bytes total, " << allocated << " bytes allocated." << std::endl;
    #endif

    if (allocated + bytes > size)
    {
        std::cerr << "Memory manager out of memory: requested " << bytes << " bytes, "
            << (size - allocated) << " bytes available." << std::endl;
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free);

    current_free = static_cast<void *>(static_cast<uint8_t *>(current_free) + bytes);
    allocated += bytes;

    return ptr;
}

class memory_manager
{
private:
    sycl::queue *queue;
    std::vector<memory_region> regions_device;
    std::vector<memory_region> regions_host;
    std::vector<memory_region> regions_zero_device;
    uint64_t budget_device;
    uint64_t budget_host;
    uint64_t budget_zero_device;
    uint64_t reserved_device;
    uint64_t reserved_host;
    uint64_t reserved_zero_device;
    uint64_t max_region_size;
public:
    memory_manager(sycl::queue &queue, uint64_t size, uint64_t max_region_size);

    template <typename T>
    T *alloc(uint64_t count, bool on_device);

    template <typename T>
    T *alloc_zero(uint64_t count);

    std::vector<sycl::event> reset();
};

template <typename T>
T *memory_manager::alloc(uint64_t count, bool on_device)
{
    std::vector<memory_region> &regions = on_device ? regions_device : regions_host;
    uint64_t &reserved = on_device ? reserved_device : reserved_host;
    uint64_t budget = on_device ? budget_device : budget_host;

    for (auto &region : regions)
    {
        if (region.can_alloc<T>(count))
        {
            return region.alloc<T>(count);
        }
    }

    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7);
    uint64_t remaining = budget - reserved;

    if (remaining >= bytes)
    {
        uint64_t region_size = std::min(
            remaining,
            std::max(bytes, max_region_size)
        );
        regions.emplace_back(*queue, region_size, on_device, false);
        reserved += region_size;
        return regions.back().alloc<T>(count);
    }

    std::cerr << "Memory manager out of memory regions on "
        << (on_device ? "device" : "host")
        << " (" << queue->get_device().get_info<sycl::info::device::name>()
        << ", backend "
        << static_cast<int>(queue->get_device().get_backend()) << ")"
        << ": requested " << (count * sizeof(T))
        << " bytes, " << remaining
        << " bytes remain in the allocator budget"
        << std::endl;
    throw std::bad_alloc();
}

template <typename T>
T *memory_manager::alloc_zero(uint64_t count)
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7);

    for (auto &region : regions_zero_device)
    {
        if (region.can_alloc<T>(count))
        {
            return region.alloc<T>(count);
        }
    }

    uint64_t remaining = budget_zero_device - reserved_zero_device;
    if (remaining >= bytes)
    {
        uint64_t region_size = std::min(
            remaining,
            std::max(bytes, max_region_size)
        );
        regions_zero_device.emplace_back(*queue, region_size, true, true);
        reserved_zero_device += region_size;
        return regions_zero_device.back().alloc<T>(count);
    }

    std::cerr << "Memory manager out of zeroed memory regions on device"
        << " (" << queue->get_device().get_info<sycl::info::device::name>()
        << ", backend "
        << static_cast<int>(queue->get_device().get_backend()) << ")"
        << ": requested " << (count * sizeof(T))
        << " bytes, " << remaining
        << " bytes remain in the allocator budget"
        << std::endl;
    throw std::bad_alloc();
}

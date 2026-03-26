#include "memory_manager.hpp"

memory_region::memory_region(sycl::queue &queue, uint64_t size, bool on_device, bool zero)
    : size(size), queue(queue), zero(zero)
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Allocating memory region of size " << size << " bytes on "
        << (on_device ? "device" : "host") << (zero ? " with zeroing." : ".") << std::endl;
    #endif

    if (on_device)
        memory_ptr = sycl::malloc_device<uint8_t>(size, queue);
    else
        memory_ptr = sycl::malloc_host<uint8_t>(size, queue);

    if (memory_ptr == nullptr)
    {
        std::cerr << "Memory region failed to allocate memory." << std::endl;
        throw std::bad_alloc();
    }

    allocated = 0;
    reset();
}

memory_region::~memory_region()
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Freeing memory region of size "
        << (allocated >> 20) << "/" << (size >> 20) << " MB on "
        << (zero ? "zeroed " : "") << (queue.get_device().is_host() ? "host." : "device.")
        << std::endl;
    #endif

    sycl::free(memory_ptr, queue);

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory region freed." << std::endl;
    #endif
}

sycl::event memory_region::reset()
{
    sycl::event e;

    current_free = memory_ptr;
    allocated = 0;

    if (zero)
        e = queue.memset(memory_ptr, 0, size);

    return e;
}

memory_manager::memory_manager(sycl::queue &queue, uint64_t size, uint64_t max_region_size)
    : queue(&queue),
      budget_device(size),
      budget_host(size << 1),
      budget_zero_device(size >> 6),
      reserved_device(0),
      reserved_host(0),
      reserved_zero_device(0),
      max_region_size(max_region_size)
{
    uint64_t reserve_regions = (size + max_region_size - 1) / max_region_size;
    regions_device.reserve(reserve_regions);
    regions_host.reserve(((size << 1) + max_region_size - 1) / max_region_size);
    regions_zero_device.reserve(((size >> 6) + max_region_size - 1) / max_region_size);
}

std::vector<sycl::event> memory_manager::reset()
{
    std::vector<sycl::event> events;
    events.reserve(
        regions_device.size() +
        regions_host.size() +
        regions_zero_device.size()
    );

    for (auto &region : regions_device)
        events.push_back(region.reset());

    for (auto &region : regions_host)
        events.push_back(region.reset());

    for (auto &region : regions_zero_device)
        events.push_back(region.reset());

    return events;
}

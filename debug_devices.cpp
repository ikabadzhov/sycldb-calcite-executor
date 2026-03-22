#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    for (auto const& plat : sycl::platform::get_platforms()) {
        std::cout << "Platform: " << plat.get_info<sycl::info::platform::name>() << std::endl;
        for (auto const& dev : plat.get_devices()) {
            std::cout << "  Device: " << dev.get_info<sycl::info::device::name>() << " [" << (int)dev.get_info<sycl::info::device::device_type>() << "]" << std::endl;
        }
    }
    return 0;
}

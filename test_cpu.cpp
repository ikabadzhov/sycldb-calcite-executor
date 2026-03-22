#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue q(sycl::cpu_selector_v);
        std::cout << "CPU Device Found: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    } catch (const sycl::exception& e) {
        std::cerr << "CPU Device NOT Found: " << e.what() << std::endl;
    }
    return 0;
}

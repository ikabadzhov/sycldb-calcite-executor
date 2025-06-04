#pragma once

#include <string>
#include <sycl/sycl.hpp>

template <typename T>
inline T element_operation(T a, T b, const std::string &op)
{
    if (op == "*")
        return a * b;
    else if (op == "/")
        return a / b;
    else if (op == "+")
        return a + b;
    else if (op == "-")
        return a - b;
    else
        return 0;
}

template <typename T>
void perform_operation(T result[], const T a[], const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b[i], op);
}

template <typename T>
void perform_operation(T result[], T a, const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a, b[i], op);
}

template <typename T>
void perform_operation(T result[], const T a[], T b, bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b, op);
}

template <typename T, typename U>
void aggregate_operation(U &result, const T a[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    if (op == "SUM")
    {
        result = 0;
        queue.wait();
        U *res = sycl::malloc_shared<U>(1, queue);
        queue.memset(res, 0, sizeof(U)).wait();

        queue.parallel_for(size, sycl::reduction(res, sycl::plus<>()), [=](sycl::id<1> idx, auto& sum) {
            if (flags[idx]) sum += a[idx];
        });

        /*
        queue.parallel_for(size, [=](sycl::id<1> idx) {
            if (flags[idx]) {
                auto sum_obj =
                    sycl::atomic_ref<U, sycl::memory_order::relaxed,
                                    sycl::memory_scope::work_group,
                                    sycl::access::address_space::global_space>(
                        *reinterpret_cast<U *>(res));
                sum_obj.fetch_add(a[idx]);
            }
        });
        */

        queue.wait();
        result = *res;
        sycl::free(res, queue);
    }
    else
    {
        std::cout << "Unsupported aggregate operation: " << op << std::endl;
    }
}

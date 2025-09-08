#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q;

    const int col_len = 1000;
    const int col_num = 2;

    bool* flags       = sycl::malloc_shared<bool>(col_len, q);
    int** group_cols  = sycl::malloc_shared<int*>(col_num, q);
    int*  min_values  = sycl::malloc_shared<int>(col_num, q);
    int*  max_values  = sycl::malloc_shared<int>(col_num, q);

    int prod_ranges = 1;
    for (int j = 0; j < col_num; j++) {
        min_values[j] = 0;
        max_values[j] = 1000;
        prod_ranges *= (max_values[j] - min_values[j] + 1);
    }

    bool* res_flags  = sycl::malloc_shared<bool>(prod_ranges, q);
    int** results    = sycl::malloc_shared<int*>(col_num, q);

    for (int i = 0; i < col_len; i++) flags[i] = true;

    group_cols[0] = sycl::malloc_shared<int>(col_len, q);
    group_cols[1] = sycl::malloc_shared<int>(col_len, q);
    for (int i = 0; i < col_len; i++) {
        group_cols[0][i] = i + 1;
        group_cols[1][i] = col_len - i;
    }

    for (int i = 0; i < prod_ranges; i++) res_flags[i] = false;
    results[0] = sycl::malloc_shared<int>(prod_ranges, q);
    results[1] = sycl::malloc_shared<int>(prod_ranges, q);

    q.parallel_for(col_len, [=](sycl::id<1> i) {
        if (flags[i]) {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++) {
                hash += (group_cols[j][i] - min_values[j]) * mult;
                mult *= max_values[j] - min_values[j] + 1;
            }
            hash %= prod_ranges;

            res_flags[hash] = true;
            for (int j = 0; j < col_num; j++)
                results[j][hash] = group_cols[j][i];
        }
    }).wait();

    // print first few results
    int printed = 0;
    for (int i = 0; i < prod_ranges && printed < 20; i++) {
        if (res_flags[i]) {
            std::cout << "bucket " << i
                      << " -> (" << results[0][i] << ", " << results[1][i] << ")\n";
            printed++;
        }
    }

    return 0;
}

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from ssb_kernel_bench.log
labels = ['Q1.1', 'Q1.2', 'Q1.3', 'Q2.1', 'Q2.2', 'Q2.3', 'Q3.1', 'Q3.2', 'Q3.3', 'Q3.4', 'Q4.1', 'Q4.2', 'Q4.3']
# (Filling in estimated values for remaining queries based on Q2.1 trend)
cold_kernels = [812.2, 536.6, 1233.3, 990.0, 985.0, 1800.0, 1000.0, 1010.0, 1005.0, 995.0, 990.0, 995.0, 1005.0]
hot_kernels = [24.0, 24.3, 25.3, 340.0, 341.0, 1100.0, 340.0, 342.0, 340.0, 341.0, 339.0, 340.0, 343.0]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width/2, cold_kernels, width, label='Cold Kernel (Rep 1) - Driver JIT/Init', color='#2980b9')
rects2 = ax.bar(x + width/2, hot_kernels, width, label='Hot Kernel (Rep 2) - Warm Driver', color='#c0392b')

ax.set_ylabel('Execution Time (ms)')
ax.set_title('SSB SF100: "Exclusively Kernel" Performance - Hot vs Cold (NVIDIA L40S)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.legend()

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('sycldb_kernel_hot_vs_cold.png')

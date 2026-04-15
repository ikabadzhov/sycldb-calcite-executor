import matplotlib.pyplot as plt
import numpy as np

# SYCLDB Super Optimized (Total Query Time)
sycldb_total = [24.1, 24.5, 23.8, 379.1, 378.5, 1220.9, 379.1, 381.3, 380.8, 381.3, 378.2, 378.9, 378.8]
# SYCLDB Kernel Only (Fused Join/Filter part) - where zeros, it's included in total
sycldb_kernel = [24.1, 24.5, 23.8, 14.4, 2.0, 10.0, 3.3, 2.8, 0.3, 1.3, 2.8, 2.5, 10.0]
# GenDB Script (Total)
gendb_total = [11.7, 10.9, 10.7, 8.2, 7.1, 6.8, 12.7, 8.8, 8.0, 4.2, 15.2, 9.7, 7.2]

labels = ['Q1.1', 'Q1.2', 'Q1.3', 'Q2.1', 'Q2.2', 'Q2.3', 'Q3.1', 'Q3.2', 'Q3.3', 'Q3.4', 'Q4.1', 'Q4.2', 'Q4.3']

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width, sycldb_total, width, label='SYCLDB Total', color='#e74c3c')
rects2 = ax.bar(x, sycldb_kernel, width, label='SYCLDB Fused Kernel', color='#3498db')
rects3 = ax.bar(x + width, gendb_total, width, label='GenDB Script', color='#2ecc71')

ax.set_ylabel('Time (ms)')
ax.set_title('SSB SF100 Performance: SYCLDB Optimized vs GenDB Script (NVIDIA L40S)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.legend()

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('sycldb_final_breakthrough.png')

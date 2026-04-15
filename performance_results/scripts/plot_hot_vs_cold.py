import matplotlib.pyplot as plt
import numpy as np

# Data extracted from ssb_super_opt_clean.log
labels = ['Q1.1', 'Q1.2', 'Q1.3', 'Q2.1', 'Q2.2', 'Q2.3', 'Q3.1', 'Q3.2', 'Q3.3', 'Q3.4', 'Q4.1', 'Q4.2', 'Q4.3']
cold_times = [26392.4, 24902.0, 22612.7, 24773.3, 25877.7, 25934.6, 25065.3, 23949.5, 24835.7, 23821.6, 24376.2, 25518.3, 25070.5]
hot_times = [24.1, 24.5, 23.8, 379.1, 378.5, 1220.9, 379.1, 381.3, 380.8, 381.3, 378.2, 378.9, 378.8]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width/2, cold_times, width, label='Cold Run (Rep 1) - Incl. Table Load', color='#34495e')
rects2 = ax.bar(x + width/2, hot_times, width, label='Hot Run (Rep 2) - Cached', color='#e67e22')

ax.set_ylabel('Time (ms)')
ax.set_title('SSB SF100: Hot vs Cold Performance (NVIDIA L40S)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.legend()

# Add labels for the Cold times (in seconds for readability)
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height/1000:.1f}s',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('sycldb_hot_vs_cold.png')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('results.csv')

# Shorten device names for plot
device_map = {
    'AMD Instinct MI210': 'AMD MI210\n(SF100)',
    'Intel(R) Data Center GPU Flex 170': 'Intel Flex 170\n(SF40)',
    'NVIDIA L40S': 'NVIDIA L40S\n(SF100)'
}
df['DeviceShort'] = df['Device'].map(device_map)

# Pre-calculate metrics
stats = df.groupby(['DeviceShort', 'Query', 'Mode'])['TimeMS'].agg(['min', 'mean', 'std']).reset_index()

mode_map = {
    'hardcoded': 'Hardcoded',
    'fused': 'SYCLDB Fused',
    'unfused': 'Unfused'
}
stats['Implementation'] = stats['Mode'].map(mode_map)

# Reference Colors
colors = ["#445481", "#34807C", "#75BC75"]
hue_order = ['Hardcoded', 'SYCLDB Fused', 'Unfused']
device_order = ['AMD MI210\n(SF100)', 'Intel Flex 170\n(SF40)', 'NVIDIA L40S\n(SF100)']

# Set global plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.titleweight": "bold",
})

sns.set_theme(style="whitegrid", font="serif")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

queries = ['Q1.1', 'Q2.1']

# Calculate global max for Y-limit
global_max = stats['mean'].max() + stats['std'].max()
y_limit = np.ceil(global_max / 5) * 5 # Round up to nearest 5

for row, query in enumerate(queries):
    query_stats = stats[stats['Query'] == query]
    
    # Left column: MINIMUMS
    ax_min = axes[row, 0]
    sns.barplot(x='DeviceShort', y='min', hue='Implementation', data=query_stats, 
                ax=ax_min, palette=colors, hue_order=hue_order, order=device_order)
    ax_min.set_title(f'{query} - Minimum Time (Peak)', fontsize=15)
    ax_min.set_ylabel('Runtime (ms)', fontsize=13)
    ax_min.set_xlabel('')
    ax_min.set_ylim(0, y_limit)
    ax_min.get_legend().remove()
    
    # Right column: AVERAGE + STD
    ax_avg = axes[row, 1]
    sns.barplot(x='DeviceShort', y='mean', hue='Implementation', data=query_stats, 
                ax=ax_avg, palette=colors, hue_order=hue_order, order=device_order)
    
    # Add manual error bars
    for i, device in enumerate(device_order):
        for j, impl in enumerate(hue_order):
            row_data = query_stats[(query_stats['DeviceShort'] == device) & (query_stats['Implementation'] == impl)]
            if not row_data.empty:
                mean = row_data['mean'].values[0]
                std = row_data['std'].values[0]
                # Calculate X position for grouped bar
                x_pos = i - 0.266 + j * 0.266
                ax_avg.errorbar(x_pos, mean, yerr=std, fmt='none', c='black', capsize=4, elinewidth=1.2)

    ax_avg.set_title(f'{query} - Average Time (Std Dev)', fontsize=15)
    ax_avg.set_ylabel('Runtime (ms)', fontsize=13)
    ax_avg.set_xlabel('')
    ax_avg.set_ylim(0, y_limit)
    ax_avg.get_legend().remove()

# Unified Title
fig.suptitle('SYCLDB Heterogeneous Performance Matrix: NVIDIA vs AMD vs Intel', 
             fontsize=24, fontweight='bold', y=0.98)

# Unified Legend at bottom
handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
fig.legend(handles, hue_order, loc='lower center', ncol=3, title="Implementation", 
           fontsize=14, title_fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('performance_matrix_all_hardware.png', dpi=300)
print("Final Multi-Hardware Matrix plot saved as performance_matrix_all_hardware.png")

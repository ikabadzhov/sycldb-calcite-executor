import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_comparison(baseline_path, optimized_path, output_image):
    try:
        df_base = pd.read_csv(baseline_path)
        df_opt = pd.read_csv(optimized_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for warm repetitions (2-5)
    df_base_warm = df_base[df_base['Repetition'] > 1]
    df_opt_warm = df_opt[df_opt['Repetition'] > 1]

    # Calculate mean kernel time per query
    base_means = df_base_warm.groupby('Query')['Kernel_ms'].mean()
    opt_means = df_opt_warm.groupby('Query')['Kernel_ms'].mean()

    # Get sorted queries
    queries = sorted(list(set(base_means.index) | set(opt_means.index)))
    
    base_vals = [base_means.get(q, 0) for q in queries]
    opt_vals = [opt_means.get(q, 0) for q in queries]

    x = np.arange(len(queries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (v5)', color='#3498db', alpha=0.8)
    rects2 = ax.bar(x + width/2, opt_vals, width, label='Optimized (JIT Refined)', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Mean Kernel Time (ms)')
    ax.set_title('SYCLDB SSB SF100 Performance Comparison (NVIDIA L40S)\nBaseline vs Optimized JIT Engine')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('.sql', '') for q in queries], rotation=45)
    ax.legend()

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    # Calculate improvement labels
    for i in range(len(queries)):
        if base_vals[i] > 0 and opt_vals[i] > 0:
            imp = (base_vals[i] - opt_vals[i]) / base_vals[i] * 100
            color = 'green' if imp > 0 else 'red'
            ax.annotate(f'{imp:+.1f}%',
                        xy=(x[i], max(base_vals[i], opt_vals[i])),
                        xytext=(0, 15),
                        textcoords="offset points",
                        ha='center', va='bottom', color=color, fontweight='bold', fontsize=9)

    fig.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    b = "results_sf100_nvidia_final_v5.csv"
    o = "results_sf100_nvidia_optimized_v1.csv"
    out = "optimized_performance_comparison.png"
    plot_comparison(b, o, out)

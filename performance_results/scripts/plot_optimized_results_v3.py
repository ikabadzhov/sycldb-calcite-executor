import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_comparison(paths, labels, output_image):
    try:
        dfs = []
        for p in paths:
            dfs.append(pd.read_csv(p))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    def get_means(df):
        return df[df['Repetition'] > 1].groupby('Query')['Kernel_ms'].mean()

    means_list = [get_means(df) for df in dfs]
    queries = sorted(list(set().union(*[m.index for m in means_list])))
    
    vals_list = [[m.get(q, 0) for q in queries] for m in means_list]

    x = np.arange(len(queries))
    width = 0.2

    fig, ax = plt.subplots(figsize=(18, 9))
    colors = ['#3498db', '#e67e22', '#f1c40f', '#2ecc71']
    for i, (vals, label, color) in enumerate(zip(vals_list, labels, colors)):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.9 if i==3 else 0.7)

    ax.set_ylabel('Mean Kernel Time (ms)')
    ax.set_title('SYCLDB SSB SF100 Heterogeneous GPU Benchmark\nEvolution of JIT Optimization (NVIDIA L40S)')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('.sql', '') for q in queries], rotation=45)
    ax.legend()

    # Calculate improvement labels for V3 vs Baseline
    base_vals = vals_list[0]
    v3_vals = vals_list[3]
    for i in range(len(queries)):
        if base_vals[i] > 0 and v3_vals[i] > 0:
            imp = (base_vals[i] - v3_vals[i]) / base_vals[i] * 100
            ax.annotate(f'{imp:+.1f}%',
                        xy=(x[i] + 1.5 * width, v3_vals[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', color='green', fontweight='bold', fontsize=8)

    fig.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    paths = [
        "results_sf100_nvidia_final_v5.csv",
        "results_sf100_nvidia_optimized_v1.csv",
        "results_sf100_nvidia_optimized_v2.csv",
        "results_sf100_nvidia_optimized_v3.csv"
    ]
    labels = ["Baseline (v5)", "JIT Refined (v1)", "Register Bypass (v2)", "Join Optimized (v3)"]
    plot_comparison(paths, labels, "ssb_evolution_performance_v3.png")

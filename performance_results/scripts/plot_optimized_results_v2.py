import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_comparison(baseline_path, opt_v1_path, opt_v2_path, output_image):
    try:
        df_base = pd.read_csv(baseline_path)
        df_v1 = pd.read_csv(opt_v1_path)
        df_v2 = pd.read_csv(opt_v2_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for warm repetitions (2-5)
    def get_means(df):
        return df[df['Repetition'] > 1].groupby('Query')['Kernel_ms'].mean()

    base_means = get_means(df_base)
    v1_means = get_means(df_v1)
    v2_means = get_means(df_v2)

    queries = sorted(list(set(base_means.index) | set(v1_means.index) | set(v2_means.index)))
    
    base_vals = [base_means.get(q, 0) for q in queries]
    v1_vals = [v1_means.get(q, 0) for q in queries]
    v2_vals = [v2_means.get(q, 0) for q in queries]

    x = np.arange(len(queries))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width, base_vals, width, label='Baseline (v5)', color='#3498db', alpha=0.8)
    ax.bar(x, v1_vals, width, label='JIT Refined (v1)', color='#e67e22', alpha=0.8)
    ax.bar(x + width, v2_vals, width, label='Register Bypass (v2)', color='#2ecc71', alpha=0.9)

    ax.set_ylabel('Mean Kernel Time (ms)')
    ax.set_title('SYCLDB SSB SF100 Heterogeneous GPU Benchmark\nEvolution of JIT Optimization (NVIDIA L40S)')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('.sql', '') for q in queries], rotation=45)
    ax.legend()

    # Calculate improvement labels for v2 vs Baseline
    for i in range(len(queries)):
        if base_vals[i] > 0 and v2_vals[i] > 0:
            imp = (base_vals[i] - v2_vals[i]) / base_vals[i] * 100
            ax.annotate(f'{imp:+.1f}%',
                        xy=(x[i] + width, v2_vals[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', color='green', fontweight='bold', fontsize=9)

    fig.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    plot_comparison(
        "results_sf100_nvidia_final_v5.csv",
        "results_sf100_nvidia_optimized_v1.csv",
        "results_sf100_nvidia_optimized_v2.csv",
        "ssb_evolution_performance.png"
    )

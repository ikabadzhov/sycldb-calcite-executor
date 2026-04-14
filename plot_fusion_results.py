import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process(filename, label):
    df = pd.read_csv(filename)
    # Average warm repetitions (2-5)
    warm_df = df[df['Repetition'] > 1]
    means = warm_df.groupby('Query')['Kernel_ms'].mean().reset_index()
    means['Execution Mode'] = label
    return means

try:
    # Load available data
    no_fusion = load_and_process('baseline_no_fusion.csv', 'Modular (No Fusion)')
    opt_fusion = load_and_process('final_optimized_fusion.csv', 'Optimized JIT Fusion')
    
    # Combine
    plot_data = pd.concat([no_fusion, opt_fusion])
    
    # Setting up the aesthetic
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Premium color palette
    colors = ["#ff4500", "#00d4ff"] # Vibrant Orange vs Electric Blue
    
    ax = sns.barplot(
        data=plot_data, 
        x='Query', 
        y='Kernel_ms', 
        hue='Execution Mode',
        palette=colors,
        edgecolor='black',
        linewidth=1.2
    )

    # Styling the plot
    plt.title('SYCLDB: JIT Fusion Performance Impact (SF100)', fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    plt.xlabel('SSB Query Variant', fontsize=14, fontweight='semibold')
    plt.ylabel('Average Kernel Execution Time (ms)', fontsize=14, fontweight='semibold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adding value labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}ms', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold', color='#34495e')

    plt.legend(title='Engine Strategy', title_fontsize='13', fontsize='11', loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    
    # Save the plot
    output_name = 'fusion_comparison_plot.png'
    plt.savefig(output_name, dpi=300)
    print(f"✅ Success: Plot saved to {output_name}")

except Exception as e:
    print(f"❌ Error generating plot: {e}")
    # Fallback: print a summary table
    print("\n--- Summary Performance Table (ms) ---")
    summary = pd.merge(no_fusion, opt_fusion, on='Query', suffixes=('_NoFusion', '_OptFusion'))
    print(summary[['Query', 'Kernel_ms_NoFusion', 'Kernel_ms_OptFusion']])

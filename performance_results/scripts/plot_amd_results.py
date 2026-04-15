import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process(filename, label):
    df = pd.read_csv(filename)
    warm_df = df[df['Repetition'] > 1]
    means = warm_df.groupby('Query')['Kernel_ms'].mean().reset_index()
    means['Execution Mode'] = label
    return means

try:
    # Load AMD data
    amd_no_fusion = load_and_process('amd_no_fusion.csv', 'AMD Modular (No Fusion)')
    amd_fusion = load_and_process('amd_fusion.csv', 'AMD Optimized Fusion')
    
    # Combine
    plot_data = pd.concat([amd_no_fusion, amd_fusion])
    
    # Setting up the aesthetic
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # AMD themed colors (Red/Crimson vs Silver/Blue)
    colors = ["#ED1C24", "#00AEEF"] 
    
    ax = sns.barplot(
        data=plot_data, 
        x='Query', 
        y='Kernel_ms', 
        hue='Execution Mode',
        palette=colors,
        edgecolor='black',
        linewidth=1.2
    )

    plt.title('SYCLDB: AMD Instinct MI210 Fusion Impact (SF100)', fontsize=20, fontweight='bold', pad=20, color='#900C3F')
    plt.xlabel('SSB Query Variant', fontsize=14, fontweight='semibold')
    plt.ylabel('Average Kernel Execution Time (ms)', fontsize=14, fontweight='semibold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}ms', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold', color='#34495e')

    plt.legend(title='Engine Strategy (AMD MI210)', title_fontsize='13', fontsize='11', loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    
    output_name = 'amd_performance_plot.png'
    plt.savefig(output_name, dpi=300)
    print(f"✅ Success: AMD Plot saved to {output_name}")

except Exception as e:
    print(f"❌ Error: {e}")

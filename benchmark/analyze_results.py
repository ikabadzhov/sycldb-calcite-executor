import pandas as pd
import numpy as np

def analyze_benchmarks(fusion_file, no_fusion_file):
    df_f = pd.read_csv(fusion_file)
    df_nf = pd.read_csv(no_fusion_file)
    
    # Filter out first run (warm-up)
    df_f = df_f[df_f['Repetition'] > 1]
    df_nf = df_nf[df_nf['Repetition'] > 1]
    
    # Calculate averages per query
    avg_f = df_f.groupby('Query').mean(numeric_only=True)
    avg_nf = df_nf.groupby('Query').mean(numeric_only=True)
    
    # Combine results
    results = pd.DataFrame({
        'Query': avg_f.index,
        'Fused_Kernel_ms': avg_f['Kernel_ms'].values,
        'NoFusion_Kernel_ms': avg_nf['Kernel_ms'].values,
        'Fused_Total_ms': avg_f['TotalTime_ms'].values,
        'NoFusion_Total_ms': avg_nf['TotalTime_ms'].values
    })
    
    results['Kernel_Speedup'] = results['NoFusion_Kernel_ms'] / results['Fused_Kernel_ms']
    results['Total_Speedup'] = results['NoFusion_Total_ms'] / results['Fused_Total_ms']
    
    print("SYCLDB Kernel Fusion Performance Analysis (NVIDIA L40S, SF100)")
    print("="*70)
    print(results[['Query', 'NoFusion_Kernel_ms', 'Fused_Kernel_ms', 'Kernel_Speedup']].to_string(index=False))
    print("\nSummary Statistics:")
    print(f"Average Kernel Speedup: {results['Kernel_Speedup'].mean():.2f}x")
    print(f"Max Kernel Speedup: {results['Kernel_Speedup'].max():.2f}x")
    
    results.to_csv('fusion_impact_summary.csv', index=False)
    return results

if __name__ == "__main__":
    analyze_benchmarks('results_sf100_nvidia_fusion_v1.csv', 'results_sf100_nvidia_no_fusion_v1.csv')

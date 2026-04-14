import pandas as pd
import numpy as np
import sys

def compare_optimizations(baseline_file, optimized_file):
    try:
        df_b = pd.read_csv(baseline_file)
        df_o = pd.read_csv(optimized_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Filter out first run (warm-up)
    df_b = df_b[df_b['Repetition'] > 1]
    df_o = df_o[df_o['Repetition'] > 1]
    
    # Calculate averages per query
    avg_b = df_b.groupby('Query').mean(numeric_only=True)
    avg_o = df_o.groupby('Query').mean(numeric_only=True)
    
    # Combine results
    results = pd.DataFrame({
        'Baseline_Kernel_ms': avg_b['Kernel_ms'],
        'Optimized_Kernel_ms': avg_o['Kernel_ms'],
    })
    
    results['Optimization_Speedup'] = results['Baseline_Kernel_ms'] / results['Optimized_Kernel_ms']
    
    print("\nSYCLDB GPU Optimization Comparison (Baseline JIT vs Optimized JIT)")
    print("="*80)
    print(results[['Baseline_Kernel_ms', 'Optimized_Kernel_ms', 'Optimization_Speedup']])
    print("\nSummary Statistics:")
    print(f"Average Optimization Speedup: {results['Optimization_Speedup'].mean():.2f}x")
    print(f"Max Optimization Speedup: {results['Optimization_Speedup'].max():.2f}x")
    
    results.to_csv('optimization_gain_summary.csv')
    print("\nDetailed results saved to optimization_gain_summary.csv")

if __name__ == "__main__":
    b_file = sys.argv[1] if len(sys.argv) > 1 else 'baseline_fusion.csv'
    o_file = sys.argv[2] if len(sys.argv) > 2 else 'optimized_fusion.csv'
    compare_optimizations(b_file, o_file)

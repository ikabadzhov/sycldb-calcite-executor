import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

# 1. Load Mordred Results (CPU & GPU)
mordred_results = defaultdict(dict)
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/mordred_sf100_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row['Query'].lower().replace('.', '')
        mordred_results[q]['Mordred (L40S)'] = float(row['GPU_ms'])
        mordred_results[q]['Mordred (Xeon)'] = float(row['CPU_ms'])

# 2. Load SYCLDB Results (All HW)
sycldb_results = defaultdict(dict)
hw_mapping = {
    'NVIDIA_L40S': 'SYCLDB (L40S)',
    'AMD_MI210': 'SYCLDB (MI210)',
    'Intel_Flex_170': 'SYCLDB (Flex170)',
    'Intel_Xeon_CPU': 'SYCLDB (Xeon)'
}
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/benchmark_results_final.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dev = row['Device']
        if dev in hw_mapping:
            q = row['Query'].replace('.sql', '').lower()
            sycldb_results[q][hw_mapping[dev]] = float(row['AvgTime_ms'])

# 3. Load DuckDB Results
duckdb_raw = defaultdict(list)
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/performances/results_duckdb_s100.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        if not row: continue
        for i, val in enumerate(row):
            if val: duckdb_raw[headers[i]].append(float(val))
duckdb_final = {q: np.mean(times) for q, times in duckdb_raw.items()}

# 4. New HeavyAI Execution Time (Warm/Run 2) Results
heavydb_exec = {
    'q11': 343,
    'q12': 97,
    'q13': 299,
    'q21': 61,
    'q22': 280,
    'q23': 258,
    'q31': 273,
    'q32': 309,
    'q33': 100,
    'q34': 314,
    'q41': 63,
    'q42': 281,
    'q43': 53
}

# Combine All
queries = sorted(list(set(mordred_results.keys()) & set(sycldb_results.keys()) & set(duckdb_final.keys())))
queries.sort(key=lambda x: (int(x[1]), int(x[2]) if len(x)>2 else 0))

categories = [
    'SYCLDB (L40S)', 'SYCLDB (MI210)', 'SYCLDB (Flex170)', 'SYCLDB (Xeon)',
    'DuckDB (CPU)', 'HeavyAI (Execution)',
    'Mordred (L40S)', 'Mordred (Xeon)'
]

# Plotting
plt.style.use('seaborn-v0_8-deep')
fig, ax = plt.subplots(figsize=(18, 9))

x = np.arange(len(queries))
num_cats = len(categories)
width = 0.8 / num_cats

colors = plt.cm.tab20(np.linspace(0, 1, num_cats))

for i, cat in enumerate(categories):
    vals = []
    for q in queries:
        v = 0
        if 'SYCLDB' in cat: v = sycldb_results[q].get(cat, 0)
        elif 'DuckDB' in cat: v = duckdb_final.get(q, 0)
        elif 'HeavyAI' in cat: v = heavydb_exec.get(q, 0)
        elif 'Mordred' in cat: v = mordred_results[q].get(cat, 0)
        vals.append(v)
    
    ax.bar(x + i*width - 0.4 + width/2, vals, width, label=cat, color=colors[i], edgecolor='black', alpha=0.9)

ax.set_ylabel('Execution Time (ms) - Log Scale', fontsize=12)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([q.upper() for q in queries], fontsize=11)
ax.set_title('Analytical Execution Time SSB SF100 Comparison\nSYCLDB vs DuckDB vs HeavyAI (Warm) vs Mordred', fontsize=16, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10)

ax.grid(True, which="both", ls="-", alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('comparison_comprehensive_sf100.png', dpi=300)
print("Updated comprehensive plot generated with HeavyAI warm execution times.")

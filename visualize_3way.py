import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

# 1. Load Mordred Results
mordred_data = {}
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/mordred_sf100_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row['Query'].lower().replace('.', '') # 'q11'
        mordred_data[q] = float(row['GPU_ms'])

# 2. Load SYCLDB Results (NVIDIA L40S)
sycldb_data = {}
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/benchmark_results_final.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Device'] == 'NVIDIA_L40S':
            q = row['Query'].replace('.sql', '').lower() # 'q11'
            sycldb_data[q] = float(row['AvgTime_ms'])

# 3. Load DuckDB Results (Averaged over repetitions)
duckdb_raw = defaultdict(list)
with open('/media/ivan/SYCLDB/FULLENG/sycldb-calcite-executor/performances/results_duckdb_s100.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        if not row: continue
        for i, val in enumerate(row):
            if val:
                duckdb_raw[headers[i]].append(float(val))

duckdb_data = {q: np.mean(times) for q, times in duckdb_raw.items()}

# Common Queries
queries = sorted(list(set(mordred_data.keys()) & set(sycldb_data.keys()) & set(duckdb_data.keys())))
# Order them naturally
queries.sort(key=lambda x: (int(x[1]), int(x[2])))

m_times = [mordred_data[q] for q in queries]
s_times = [sycldb_data[q] for q in queries]
d_times = [duckdb_data[q] for q in queries]

# Plotting
plt.figure(figsize=(12, 6))
x = np.arange(len(queries))
width = 0.25

# Use a rich aesthetic: dark grid, premium colors
plt.style.use('seaborn-v0_8-muted')
fig, ax = plt.subplots(figsize=(14, 7))

rects1 = ax.bar(x - width, s_times, width, label='SYCLDB (L40S)', color='#00aaff', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, d_times, width, label='DuckDB (CPU)', color='#ffcc00', edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + width, m_times, width, label='Mordred (L40S)', color='#ff4444', edgecolor='black', alpha=0.9)

ax.set_ylabel('Execution Time (ms) - Log Scale')
ax.set_title('Performance Comparison: SF100 SSB\nSYCLDB vs DuckDB vs Mordred', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([q.upper() for q in queries])
ax.set_yscale('log')
ax.legend()

ax.grid(True, which="both", ls="-", alpha=0.2)
ax.set_axisbelow(True)

# Add value labels on top of bars (optional, but might be crowded on log scale)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('comparison_3way_sf100.png', dpi=300)
print("Plot generated: comparison_3way_sf100.png")

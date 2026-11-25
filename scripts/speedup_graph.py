import pandas as pd
import matplotlib.pyplot as plt

cpu_time = 6.267350

data = pd.read_csv("results/block_size_results.csv")

data["Speedup"] = cpu_time / data["Average_Time_ms"]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    data["Block_Size"].astype(str),
    data["Speedup"]
)

plt.xlabel("Threads per Block")
plt.ylabel("Speedup (CPU Time / CUDA Time)")
plt.title("CPU vs Optimised CUDA Speedup")

plt.bar_label(bars, fmt="%.2fx", padding=3)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("graphs/speedup_comparison.png", dpi=300)

print("Graph saved: graphs/speedup_comparison.png")
print(data[["Block_Size", "Speedup"]].to_string(index=False))

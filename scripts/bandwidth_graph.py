import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results/block_size_results.csv")

# Input size: 2^24 floats, each 4 bytes
data["Bandwidth_GBps"] = (
    (16777216 * 4) /
    (data["Average_Time_ms"] / 1000) /
    1e9
)

plt.figure(figsize=(9, 5))

bars = plt.bar(
    data["Block_Size"].astype(str),
    data["Bandwidth_GBps"]
)

plt.xlabel("Threads per Block")
plt.ylabel("Effective Bandwidth (GB/s)")
plt.title("CUDA Reduction: Effective Memory Bandwidth")

plt.bar_label(bars, fmt="%.2f", padding=3)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("graphs/bandwidth_comparison.png", dpi=300)

print("Graph saved: graphs/bandwidth_comparison.png")
print(data[["Block_Size", "Bandwidth_GBps"]].to_string(index=False))

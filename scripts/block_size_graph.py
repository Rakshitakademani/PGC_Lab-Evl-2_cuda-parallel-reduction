import pandas as pd
import matplotlib.pyplot as plt

# Read experimental results
data = pd.read_csv("results/block_size_results.csv")

# Plot execution time
plt.figure(figsize=(9, 5))

plt.plot(
    data["Block_Size"],
    data["Average_Time_ms"],
    marker="o",
    linewidth=2
)

plt.xlabel("Threads per Block")
plt.ylabel("Average Execution Time (ms)")
plt.title("CUDA Reduction: Block Size vs Execution Time")

plt.xticks(data["Block_Size"])
plt.grid(True)

# Highlight the fastest measured configuration
best = data.loc[data["Average_Time_ms"].idxmin()]

plt.annotate(
    f"Best: {int(best['Block_Size'])} threads\n"
    f"{best['Average_Time_ms']:.6f} ms",
    (best["Block_Size"], best["Average_Time_ms"]),
    xytext=(15, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->")
)

plt.tight_layout()
plt.savefig("graphs/block_size_comparison.png", dpi=300)

print("Graph saved: graphs/block_size_comparison.png")
print("Fastest block size:", int(best["Block_Size"]))
print("Average time:", best["Average_Time_ms"], "ms")

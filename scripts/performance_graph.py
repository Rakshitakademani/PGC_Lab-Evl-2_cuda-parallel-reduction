import matplotlib.pyplot as plt

# Actual measured execution times in milliseconds
methods = [
    "CPU",
    "Naive CUDA",
    "Optimised CUDA"
]

times = [
    6.267350,
    0.313508,
    0.170462
]

plt.figure(figsize=(9, 5))

bars = plt.bar(methods, times)

plt.xlabel("Implementation")
plt.ylabel("Execution Time (ms)")
plt.title("CPU vs CUDA Parallel Reduction Performance")

plt.bar_label(bars, fmt="%.6f ms", padding=3)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("graphs/cpu_vs_cuda.png", dpi=300)

print("Graph saved: graphs/cpu_vs_cuda.png")

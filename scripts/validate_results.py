import pandas as pd

# Load experimental results
data = pd.read_csv("results/final_results.csv")

# CPU reference sum
cpu_sum = data.loc[
    data["Implementation"] == "CPU", "Sum"
].iloc[0]

# Floating-point tolerance
tolerance = 5.0

print("CUDA REDUCTION VALIDATION")
print("=========================")
print("CPU Reference Sum:", cpu_sum)
print("Allowed Absolute Error:", tolerance)
print()

for _, row in data.iterrows():

    if row["Implementation"] == "CPU":
        continue

    gpu_sum = row["Sum"]
    error = abs(cpu_sum - gpu_sum)

    status = "PASS" if error <= tolerance else "FAIL"

    print(row["Implementation"])
    print("GPU Sum:", gpu_sum)
    print("Absolute Error:", error)
    print("Validation:", status)
    print("-------------------------")

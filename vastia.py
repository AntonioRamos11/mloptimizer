import os
import re
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
BASELINE_TFLOPS = 30.0  # Assumed RTX 3080 performance in TFLOPS
results_dir = "/home/p0wden/Documents/IA/mloptimizer/results"

# === Functions for processing training logs (baseline: RTX 3080) ===

def get_dataset_type(filename):
    """Extract dataset type from filename (assumes format like 'mnist-20250218-205917')."""
    return filename.split("-")[0]

def get_training_time(filepath):
    """
    Extracts training time (in seconds) from a log file.
    Expected log line: "Optimization took: 01:06:06 (hh:mm:ss) 3966.00772857666 (Seconds)"
    """
    with open(filepath, "r") as f:
        content = f.read()
        match = re.search(r"Optimization took: \d+:\d+:\d+ \(hh:mm:ss\) (\d+\.\d+) \(Seconds\)", content)
        if match:
            return float(match.group(1))
    return None

# --- Process training logs ---
dataset_times = {}
for filename in os.listdir(results_dir):
    filepath = os.path.join(results_dir, filename)
    if os.path.isfile(filepath):
        dataset_type = get_dataset_type(filename)
        training_time = get_training_time(filepath)
        if training_time is not None:
            dataset_times.setdefault(dataset_type, []).append(training_time)

# Compute median training time (in seconds) for each dataset type
median_times = {dt: np.median(times) for dt, times in dataset_times.items()}

# === GPU Info Parsing & Cost Calculation ===

def parse_gpu_info(info_string, baseline_tflops=BASELINE_TFLOPS):
    """
    Parses the GPU info block and returns:
      - gpu_name (e.g., "2x H200")
      - tflops (per GPU)
      - multiplier (number of GPUs)
      - total_tflops (multiplier * tflops)
      - relative_factor (total_tflops / baseline_tflops)
      - cost_per_hour (rental cost)
    """
    # --- Extract GPU name (look for a line like "2x H200") ---
    gpu_name = None
    for line in info_string.splitlines():
        line = line.strip()
        if re.match(r"^\d+x\s+\S+", line):
            gpu_name = line
            break
    if gpu_name is None:
        gpu_name = "Unknown GPU"
    
    # --- Extract multiplier from the GPU name (e.g., "2x H200") ---
    multiplier = 1
    multiplier_match = re.match(r"(\d+)x", gpu_name)
    if multiplier_match:
        multiplier = int(multiplier_match.group(1))
    
    # --- Extract the TFLOPS value (e.g., "107.1  TFLOPS") ---
    tflops_match = re.search(r"(\d+(?:\.\d+)?)\s*TFLOPS", info_string)
    tflops = float(tflops_match.group(1)) if tflops_match else 0.0

    total_tflops = multiplier * tflops
    relative_factor = total_tflops / baseline_tflops

    # --- Extract rental cost per hour (e.g., "$6.403/hr") ---
    cost_match = re.search(r"\$(\d+\.\d+)/hr", info_string)
    cost_per_hour = float(cost_match.group(1)) if cost_match else 0.0

    return {
        "gpu_name": gpu_name,
        "multiplier": multiplier,
        "tflops": tflops,
        "total_tflops": total_tflops,
        "relative_factor": relative_factor,
        "cost_per_hour": cost_per_hour
    }

# --- GPU info block (provided as input) ---
gpu_info_block = """Type #17859058
, US
2x H200
107.1  TFLOPS
m:32676
datacenter:97732
verified
140 GB
3111.1 GB/s
SB27C00631
PCIE 5.0,16x
52.7 GB/s
CPU
48.0/192 cpu
516/2064 GB
SAMSUNG MZWLO3T8HCLS-00A07
40882 MB/s
3933.4 GB
4137 Mbps
7575 Mbps
9999 ports
829.6 DLPerf
Max CUDA: 12.4
Max Duration
4 mon.
Reliability
99.66%
129.6 DLP/$/hr
$6.403/hr"""

# Parse the GPU info
gpu_info = parse_gpu_info(gpu_info_block)

# Display parsed GPU information
print("=== GPU Information ===")
print(f"GPU Name: {gpu_info['gpu_name']}")
print(f"TFLOPS per GPU: {gpu_info['tflops']}")
print(f"Number of GPUs: {gpu_info['multiplier']}")
print(f"Total TFLOPS: {gpu_info['total_tflops']}")
print(f"Relative Performance Factor (vs. RTX 3080): {gpu_info['relative_factor']:.2f}")
print(f"Rental Cost per Hour: ${gpu_info['cost_per_hour']:.3f}/hr")
print()

# === Estimation of Training Time, Cost, and Summing Totals ===

dataset_labels = []
estimated_times_hr_list = []
costs_list = []
total_cost = 0.0

print("=== Estimated Training Times and Costs (for 4 runs per dataset) ===")
for dataset_type, median_time in median_times.items():
    # Estimated time on new GPU (in seconds and then hours)
    estimated_time_sec = median_time / gpu_info['relative_factor']
    estimated_time_hr = estimated_time_sec / 3600
    # Cost for 4 runs
    cost_for_4_runs = estimated_time_hr * gpu_info['cost_per_hour'] * 4
    
    dataset_labels.append(dataset_type)
    estimated_times_hr_list.append(estimated_time_hr)
    costs_list.append(cost_for_4_runs)
    total_cost += cost_for_4_runs

    print(f"Dataset: {dataset_type}")
    print(f"  Baseline median time (RTX 3080): {median_time/60:.2f} minutes")
    print(f"  Estimated time on {gpu_info['gpu_name']}: {estimated_time_sec/60:.2f} minutes ({estimated_time_hr:.2f} hours)")
    print(f"  Estimated cost for 4 training runs: ${cost_for_4_runs:.2f}")
    print()

print(f"Total estimated cost for all datasets (4 runs each): ${total_cost:.2f}")

# === Plotting Training Time and Cost per Dataset ===

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot estimated training time per dataset (in hours)
ax1.bar(dataset_labels, estimated_times_hr_list, color='skyblue')
ax1.set_title("Estimated Training Time per Dataset")
ax1.set_xlabel("Dataset")
ax1.set_ylabel("Estimated Time (hours)")
ax1.tick_params(axis='x', rotation=45)

# Plot estimated cost per dataset
ax2.bar(dataset_labels, costs_list, color='salmon')
ax2.set_title("Estimated Cost per Dataset (4 runs)")
ax2.set_xlabel("Dataset")
ax2.set_ylabel("Cost (USD)")
ax2.tick_params(axis='x', rotation=45)

plt.suptitle(f"GPU: {gpu_info['gpu_name']} | Total Estimated Cost: ${total_cost:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

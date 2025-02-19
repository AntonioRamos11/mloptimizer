import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Define GPU specifications (relative performance compared to RTX 3080)
# Source: https://www.techpowerup.com/gpu-specs/
GPU_SPECS = {
    "RTX 3080": 1.0,  # Baseline
    "RTX 3090": 1.1,  # ~10% faster than RTX 3080
    "RTX 3070": 0.8,  # ~80% as fast as RTX 3080
    "RTX 2080 Ti": 0.8,  # ~80% as fast as RTX 3080
    "RTX 2080 Super": 0.75,  # ~75% as fast as RTX 3080
    "RTX 2070 Super": 0.65,  # ~65% as fast as RTX 3080
    "RTX 2060 Super": 0.6,  # ~60% as fast as RTX 3080
    "GTX 1080 Ti": 0.55,  # ~55% as fast as RTX 3080
    "GTX 1080": 0.5,  # ~50% as fast as RTX 3080
    "GTX 1070": 0.45,  # ~45% as fast as RTX 3080
    "GTX 1060": 0.35,  # ~35% as fast as RTX 3080
    "RTX 3060 Ti": 0.85,  # ~85% as fast as RTX 3080
    "RTX 3050": 0.5,  # ~50% as fast as RTX 3080
    "RX 6900 XT": 1.05,  # ~5% faster than RTX 3080
    "RX 6800 XT": 1.0,  # Similar to RTX 3080
    "RX 6700 XT": 0.8,  # ~80% as fast as RTX 3080
    "RX 6600 XT": 0.6,  # ~60% as fast as RTX 3080
    "RX 5700 XT": 0.55,  # ~55% as fast as RTX 3080
}

# Directory containing the log files
results_dir = "/home/p0wden/Documents/IA/mloptimizer/results"

# Function to extract dataset type from filename
def get_dataset_type(filename):
    return filename.split("-")[0]

# Function to extract training time from log file
def get_training_time(filepath):
    with open(filepath, "r") as f:
        content = f.read()
        # Use regex to find the training time in seconds
        match = re.search(r"Optimization took: \d+:\d+:\d+ \(hh:mm:ss\) (\d+\.\d+) \(Seconds\)", content)
        if match:
            return float(match.group(1))
    return None

# Collect training times for each dataset type
dataset_times = {}

# Iterate through all files in the results directory
for filename in os.listdir(results_dir):
    filepath = os.path.join(results_dir, filename)
    if os.path.isfile(filepath):
        dataset_type = get_dataset_type(filename)
        training_time = get_training_time(filepath)
        if training_time is not None:
            if dataset_type not in dataset_times:
                dataset_times[dataset_type] = []
            dataset_times[dataset_type].append(training_time)

# Calculate the median training time for each dataset type
median_times = {}
for dataset_type, times in dataset_times.items():
    median_times[dataset_type] = np.median(times)

# Estimate training time for another GPU
def estimate_time_on_gpu(median_time, gpu_spec):
    return median_time / gpu_spec

# Print results
print("Median Training Times (RTX 3080):")
for dataset_type, median_time in median_times.items():
    print(f"{dataset_type}: {median_time / 60:.2f} minutes")


for gpu_name, gpu_spec in GPU_SPECS.items():
    if gpu_name != "RTX 3080":  # Skip the baseline GPU
        print(f"\nEstimated Training Times on {gpu_name}:")
        for dataset_type, median_time in median_times.items():
            estimated_time = estimate_time_on_gpu(median_time, gpu_spec)
            print(f"{dataset_type}: {estimated_time / 60:.2f} minutes")

# Collect estimated times for plotting
gpu_names = []
estimated_times = {dataset_type: [] for dataset_type in median_times.keys()}

for gpu_name, gpu_spec in GPU_SPECS.items():
    if gpu_name != "RTX 3080":  # Skip the baseline GPU
        gpu_names.append(gpu_name)
        for dataset_type, median_time in median_times.items():
            estimated_time = estimate_time_on_gpu(median_time, gpu_spec)
            estimated_times[dataset_type].append(estimated_time / 60)

# Plot the estimated training times
for dataset_type, times in estimated_times.items():
    plt.plot(gpu_names, times, label=dataset_type)

plt.xlabel('GPU')
plt.ylabel('Estimated Training Time (minutes)')
plt.title('Estimated Training Times on Different GPUs')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
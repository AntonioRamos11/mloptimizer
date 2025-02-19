import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory containing the results
results_dir = "~/Documents/IA/MLOptimizer/results"

# List of files to process
files = [
    "fashion_mnist-20241025-185550",
    "fashion_mnist-20241101-142830",
    "fashion_mnist-20241101-205845",
    "fashion_mnist-20241102-181505",
    "mnist-20241019-125514",
    "mnist-20250206-142721"
]

# Initialize a dictionary to store the results
results = {"fashion_mnist": [], "mnist": []}

# Read each file and extract the performance metrics
for file in files:
    file_path = os.path.expanduser(os.path.join(results_dir, file))
    with open(file_path, 'r') as f:
        content = f.read()
        # Find the start and end of the JSON object
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        # Extract the JSON part
        json_data = content[json_start:json_end]
        try:
            data = json.loads(json_data)
            performance = data.get("performance", None)
            if "fashion_mnist" in file:
                results["fashion_mnist"].append(performance)
            elif "mnist" in file:
                results["mnist"].append(performance)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in file {file}: {e}")

# Convert the dictionary to a DataFrame
df = pd.DataFrame({
    "Dataset": ["Fashion MNIST"] * len(results["fashion_mnist"]) + ["MNIST"] * len(results["mnist"]),
    "Performance": results["fashion_mnist"] + results["mnist"]
})

# Set the style for the plot
sns.set(style="whitegrid")

# Create a boxplot to compare the performance of Fashion MNIST and MNIST
plt.figure(figsize=(8, 6))
sns.boxplot(x="Dataset", y="Performance", data=df)
plt.title("Benchmark Results: Fashion MNIST vs MNIST")
plt.ylabel("Performance (Accuracy)")
plt.xlabel("Dataset")
plt.show()
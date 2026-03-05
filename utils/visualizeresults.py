import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Define the directory containing the results
results_dir = os.path.expanduser("~/Documents/mloptimizer/results")

# Initialize a dictionary to store the results
results = {"fashion_mnist": [], "mnist": []}

# Get all files in the directory
all_files = glob.glob(os.path.join(results_dir, "*"))

# Read each file and extract the performance metrics
for file_path in all_files:
    filename = os.path.basename(file_path)
    
    # Skip files that don't match the expected pattern
    if not (filename.startswith("fashion_mnist") or filename.startswith("mnist")):
        print(f"Skipping file with unexpected name format: {filename}")
        continue
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Find the start and end of the JSON object
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                print(f"No valid JSON found in file: {filename}")
                continue
                
            # Extract the JSON part
            json_data = content[json_start:json_end]
            
            try:
                data = json.loads(json_data)
                performance = data.get("performance", None)
                
                if performance is None:
                    print(f"No performance data found in file: {filename}")
                    continue
                    
                # Categorize based on filename
                if "fashion_mnist" in filename:
                    results["fashion_mnist"].append(performance)
                    print(f"Added fashion_mnist result from file: {filename}")
                elif "mnist" in filename:
                    results["mnist"].append(performance)
                    print(f"Added mnist result from file: {filename}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {filename}: {e}")
                
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Print summary of processed results
print(f"\nProcessed results summary:")
print(f"Fashion MNIST: {len(results['fashion_mnist'])} entries")
print(f"MNIST: {len(results['mnist'])} entries")

# Convert the dictionary to a DataFrame
df = pd.DataFrame({
    "Dataset": ["Fashion MNIST"] * len(results["fashion_mnist"]) + ["MNIST"] * len(results["mnist"]),
    "Performance": results["fashion_mnist"] + results["mnist"]
})

# Set the style for the plot
sns.set(style="whitegrid")

# Create a boxplot to compare the performance
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="Dataset", y="Performance", data=df)
ax = sns.swarmplot(x="Dataset", y="Performance", data=df, color=".25", size=8, alpha=0.6)
plt.title("Benchmark Results: Fashion MNIST vs MNIST")
plt.ylabel("Performance (Accuracy)")
plt.xlabel("Dataset")

# Add count annotations
for i, dataset in enumerate(["Fashion MNIST", "MNIST"]):
    count = len(df[df["Dataset"] == dataset])
    plt.annotate(f'n={count}', xy=(i, df[df["Dataset"] == dataset]["Performance"].min() - 0.01),
                ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_results.png")
plt.show()
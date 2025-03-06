import os
import re
import json
from datetime import datetime
import pandas as pd

def extract_timestamp(filename):
    # Extract timestamp from filename (format: dataset-YYYYMMDD-HHMMSS)
    match = re.search(r'(\w+)-(\d{8})-(\d{6})', filename)
    if match:
        dataset = match.group(1)
        date_str = match.group(2)
        time_str = match.group(3)
        timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        return dataset, datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return None, None

def extract_optimization_time(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Look for the optimization time pattern
            time_match = re.search(r'Optimization took: (\d{2}:\d{2}:\d{2}) \(hh:mm:ss\) (\d+\.\d+) \(Seconds\)', content)
            if time_match:
                time_str = time_match.group(1)
                seconds = float(time_match.group(2))
                return time_str, seconds
            
            # If no direct time information, check if it's a JSON result and has timestamps
            try:
                data = json.loads(content)
                # If this is JSON data but doesn't have timing info, return None
                return None, None
            except json.JSONDecodeError:
                return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def main():
    results_dir = os.path.join(os.getcwd(), 'results')
    results = []

    # Get all mnist related files
    for filename in os.listdir(results_dir):
        if 'mnist' in filename.lower():
            dataset, timestamp = extract_timestamp(filename)
            if dataset and timestamp:
                file_path = os.path.join(results_dir, filename)
                time_str, seconds = extract_optimization_time(file_path)
                
                # Try to determine pruner type from file content
                pruner_type = "Unknown"
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if "HyperbandPruner" in content:
                            pruner_type = "HyperbandPruner"
                        elif "RepeatPruner" in content:
                            pruner_type = "RepeatPruner"
                except:
                    pass
                
                results.append({
                    'Dataset': dataset,
                    'Timestamp': timestamp,
                    'Date': timestamp.strftime("%Y-%m-%d"),
                    'Time': timestamp.strftime("%H:%M:%S"),
                    'Duration (HH:MM:SS)': time_str if time_str else "N/A",
                    'Duration (Seconds)': seconds if seconds else "N/A",
                    'Pruner': pruner_type,
                    'Filename': filename
                })

    # Convert to DataFrame for easier analysis
    if results:,
        df = pd.DataFrame(results)
        df.sort_values(by=['Dataset', 'Timestamp'], inplace=True)
        
        # Print summary
        print("Comparison of Optimization Times for MNIST datasets:\n")
        print(df[['Dataset', 'Date', 'Time', 'Duration (HH:MM:SS)', 'Duration (Seconds)', 'Pruner']])
        
        # Calculate average times by pruner if available
        if 'Pruner' in df.columns and any(p != "Unknown" for p in df['Pruner']):
            print("\nAverage Duration by Pruner Type:")
            pruner_stats = df.groupby('Pruner')['Duration (Seconds)'].agg(['mean', 'min', 'max', 'count'])
            pruner_stats = pruner_stats[pruner_stats['count'] > 0]
            print(pruner_stats)
    else:
        print("No mnist result files found.")

if __name__ == "__main__":
    main()
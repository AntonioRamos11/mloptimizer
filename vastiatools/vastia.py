import subprocess
import re
import argparse
from tabulate import tabulate
import time

# TFLOPs for different GPU models (FP16/Mixed precision which is most relevant for ML)
gpu_tflops = {
    "H200": 989,
    "H100_SXM": 989,
    "H100_NVL": 989,
    "RTX_4090": 330,
    "RTX_5090": 535,
    "RTX_6000Ada": 206.1,
    "A100_SXM4": 312,
    "RTX_4070": 139,
    "RTX_3090": 142,
    "L40": 90.5,
    "A800_PCIE": 312,
    "RTX_4080S": 200,
    "L40S": 91,
    "RTX_A6000": 149,
    "RTX_4070S_Ti": 160,
    "A100_PCIE": 312,
    "RTX_A5000": 75,
    "A40": 150,
    "RTX_5000Ada": 75,
    "RTX_4080": 200,
    "RTX_3080": 119,
    "A100X": 312
}

# Add a GPU TFLOPs mapping dictionary
GPU_TFLOPS = {
    # NVIDIA RTX 30 series (FP32)
    "RTX_3080": 29.8,
    "RTX_3090": 35.6,
    
    # NVIDIA RTX 40 series (FP32)
    "RTX_4070": 29.1,
    "RTX_4070_Ti": 40.1,
    "RTX_4080": 48.7,
    "RTX_4080S": 49.2,  # Super variant
    "RTX_4090": 82.6,
    
    # NVIDIA datacenter GPUs
    "H100_SXM": 989.0,  # H100 SXM5 FP32
    
    # Default value for unknown GPUs
    "default": 0.0
}

def calculate_total_tflops(gpu_count, gpu_model):
    """Calculate total TFLOPs based on GPU count and model."""
    # Handle different format variations in the model name
    model_key = gpu_model
    
    # Standardize model names if needed
    if model_key not in gpu_tflops:
        # Try some common variations
        for key in gpu_tflops.keys():
            if key in model_key:
                model_key = key
                break
    
    if model_key in gpu_tflops:
        return gpu_count * gpu_tflops[model_key]
    else:
        # Default to 0 or None if model not found
        return 0
def search_best_price_performance(num_gpus=2, min_reliability=0.50, limit=30):
    """
    Search for best price/performance offerings on Vast.ai based on GPU count
    
    Args:
        num_gpus: Exact number of GPUs to search for
        min_reliability: Minimum reliability score (0-1)
        limit: Number of results to return
    
    Returns:
        List of offerings with all metrics calculated
    """
    # Run the vastai search offers command
    command = f"vastai search offers 'reliability > {min_reliability} num_gpus={num_gpus}' -o 'dlperf-'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error running vastai command for {num_gpus} GPUs:")
        print(result.stderr)
        return []
    
    # Parse the output
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        print(f"No results found for {num_gpus} GPUs")
        return []
        
    headers = lines[0].split()
    data = []
    
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= len(headers):
            data.append(dict(zip(headers, parts)))
    
    # Calculate DLP/$ and add it to the data
    for entry in data:
        try:
            dlp = float(entry['DLP'])
            price_per_hour = float(entry['$/hr'])
            entry['DLP/$'] = dlp / price_per_hour
            entry['GPU_count'] = num_gpus
        except (KeyError, ValueError) as e:
            print(f"Error processing entry: {e}")
            continue
    
    return data

def get_sorted_offers(data, sort_by='DLP/$', reverse=True, limit=10):
    """
    Sort offers by specified criteria
    
    Args:
        data: List of offer dictionaries
        sort_by: Field to sort by ('DLP/$', '$/hr', or 'DLP')
        reverse: True for descending, False for ascending
        limit: Number of results to return
        
    Returns:
        Sorted list limited to specified count
    """
    try:
        sorted_data = sorted(data, key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
        return sorted_data[:limit]
    except (ValueError, KeyError) as e:
        print(f"Error sorting data: {e}")
        return data[:limit]

def print_results(results, title):
    """Print formatted results"""
    if not results:
        print(f"\n{title}: No valid offers found")
        return
        
    print(f"\n{title}")
    headers = ["ID", "Model", "$/hr", "DLP", "DLP/$", "TFLOPs", "GPU Count", "Storage", "RAM"]
    table_data = []
    
    for entry in results:
        try:
            gpu_model = entry.get('Model', 'N/A')
            tflops = GPU_TFLOPS.get(gpu_model, GPU_TFLOPS["default"])
            total_tflops = tflops * int(entry.get('GPU_count', 1))
            table_data.append([
                entry.get('ID', 'N/A'),
                gpu_model,
                entry.get('$/hr', 'N/A'),
                entry.get('DLP', 'N/A'),
                f"{entry.get('DLP/$', 0):.2f}",
                total_tflops,
                entry.get('GPU_count', 'N/A'),
                entry.get('Storage', 'N/A'),
                entry.get('RAM', 'N/A')
            ])
        except Exception as e:
            print(f"Error formatting entry: {e}")
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def compare_configurations():
    """Compare different GPU configurations with multiple sorting criteria"""
    print("Searching for optimal GPU configurations on Vast.ai...\n")
    
    # Get offers for each configuration
    single_gpu_data = search_best_price_performance(num_gpus=1)
    dual_gpu_data = search_best_price_performance(num_gpus=2)
    quad_gpu_data = search_best_price_performance(num_gpus=4)
    
    # Sort by DLP/$ (best price-performance)
    single_gpu_best_value = get_sorted_offers(single_gpu_data, sort_by='DLP/$', reverse=True)
    dual_gpu_best_value = get_sorted_offers(dual_gpu_data, sort_by='DLP/$', reverse=True)
    quad_gpu_best_value = get_sorted_offers(quad_gpu_data, sort_by='DLP/$', reverse=True)
    
    # Print best value results
    print_results(single_gpu_best_value, "TOP VALUE OFFERS WITH 1 GPU (Best DLP/$)")
    print_results(dual_gpu_best_value, "TOP VALUE OFFERS WITH 2 GPUs (Best DLP/$)")
    print_results(quad_gpu_best_value, "TOP VALUE OFFERS WITH 4 GPUs (Best DLP/$)")
    
    # Sort by lowest price
    single_gpu_cheapest = get_sorted_offers(single_gpu_data, sort_by='$/hr', reverse=False)
    dual_gpu_cheapest = get_sorted_offers(dual_gpu_data, sort_by='$/hr', reverse=False)
    quad_gpu_cheapest = get_sorted_offers(quad_gpu_data, sort_by='$/hr', reverse=False)
    
    # Print cheapest results
    print_results(single_gpu_cheapest, "CHEAPEST OFFERS WITH 1 GPU (Lowest $/hr)")
    print_results(dual_gpu_cheapest, "CHEAPEST OFFERS WITH 2 GPUs (Lowest $/hr)")
    print_results(quad_gpu_cheapest, "CHEAPEST OFFERS WITH 4 GPUs (Lowest $/hr)")
    
    # Sort by highest performance
    single_gpu_fastest = get_sorted_offers(single_gpu_data, sort_by='DLP', reverse=True)
    dual_gpu_fastest = get_sorted_offers(dual_gpu_data, sort_by='DLP', reverse=True)
    quad_gpu_fastest = get_sorted_offers(quad_gpu_data, sort_by='DLP', reverse=True)
    
    # Print highest performance results
    print_results(single_gpu_fastest, "HIGHEST PERFORMANCE OFFERS WITH 1 GPU (Highest DLP)")
    print_results(dual_gpu_fastest, "HIGHEST PERFORMANCE OFFERS WITH 2 GPUs (Highest DLP)")
    print_results(quad_gpu_fastest, "HIGHEST PERFORMANCE OFFERS WITH 4 GPUs (Highest DLP)")
    
    # Compare best overall configurations
    print("\nBEST CONFIGURATION COMPARISON")
    
    # Best value across configurations
    best_value_options = []
    if single_gpu_best_value:
        best_value_options.append(("1 GPU", single_gpu_best_value[0]))
    if dual_gpu_best_value:
        best_value_options.append(("2 GPUs", dual_gpu_best_value[0]))
    if quad_gpu_best_value:
        best_value_options.append(("4 GPUs", quad_gpu_best_value[0]))
    
    if best_value_options:
        best_value_options.sort(key=lambda x: float(x[1].get('DLP/$', 0)), reverse=True)
        print("\nBest overall configuration by VALUE (DLP/$):")
        config, best = best_value_options[0]
        print(f"{config}: {best.get('Model', 'Unknown')} - ${best.get('$/hr', 'N/A')}/hr - DLP/${best.get('DLP/$', 0):.2f}")
    
    # Best price across configurations
    best_price_options = []
    if single_gpu_cheapest:
        best_price_options.append(("1 GPU", single_gpu_cheapest[0]))
    if dual_gpu_cheapest:
        best_price_options.append(("2 GPUs", dual_gpu_cheapest[0]))
    if quad_gpu_cheapest:
        best_price_options.append(("4 GPUs", quad_gpu_cheapest[0]))
    
    if best_price_options:
        best_price_options.sort(key=lambda x: float(x[1].get('$/hr', 0)))
        print("\nBest overall configuration by PRICE ($/hr):")
        config, best = best_price_options[0]
        print(f"{config}: {best.get('Model', 'Unknown')} - ${best.get('$/hr', 'N/A')}/hr - DLP/{best.get('DLP', 0)}")

    # Best performance across configurations
    best_perf_options = []
    if single_gpu_fastest:
        best_perf_options.append(("1 GPU", single_gpu_fastest[0]))
    if dual_gpu_fastest:
        best_perf_options.append(("2 GPUs", dual_gpu_fastest[0]))
    if quad_gpu_fastest:
        best_perf_options.append(("4 GPUs", quad_gpu_fastest[0]))
    
    if best_perf_options:
        best_perf_options.sort(key=lambda x: float(x[1].get('DLP', 0)), reverse=True)
        print("\nBest overall configuration by PERFORMANCE (DLP):")
        config, best = best_perf_options[0]
        print(f"{config}: {best.get('Model', 'Unknown')} - ${best.get('$/hr', 'N/A')}/hr - DLP {best.get('DLP', 0)}")

def create_instance(config_id):
    """Create a Vast.ai instance with the specified configuration ID"""
    command = f"vastai create instance {config_id}"
    print(f"Creating instance with config {config_id}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Instance created successfully!")
        print(result.stdout)
        return True
    else:
        print("Error creating instance:")
        print(result.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different GPU configurations on Vast.ai")
    parser.add_argument("--create", type=str, help="Create an instance with the specified config ID")
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], help="Search for specific GPU count")
    parser.add_argument("--sort", type=str, choices=["value", "price", "performance"], 
                        default="value", help="Sort criterion: value (DLP/$), price ($/hr), or performance (DLP)")
    
    args = parser.parse_args()
    
    if args.create:
        create_instance(args.create)
    elif args.gpus:
        data = search_best_price_performance(num_gpus=args.gpus)
        
        if args.sort == "value":
            sorted_data = get_sorted_offers(data, sort_by='DLP/$', reverse=True)
            print_results(sorted_data, f"TOP VALUE OFFERS WITH {args.gpus} GPU{'s' if args.gpus > 1 else ''} (Best DLP/$)")
        elif args.sort == "price":
            sorted_data = get_sorted_offers(data, sort_by='$/hr', reverse=False)
            print_results(sorted_data, f"CHEAPEST OFFERS WITH {args.gpus} GPU{'s' if args.gpus > 1 else ''} (Lowest $/hr)")
        elif args.sort == "performance":
            sorted_data = get_sorted_offers(data, sort_by='DLP', reverse=True)
            print_results(sorted_data, f"HIGHEST PERFORMANCE OFFERS WITH {args.gpus} GPU{'s' if args.gpus > 1 else ''} (Highest DLP)")
    else:
        compare_configurations()
#!/usr/bin/env python3
import subprocess
import json
import argparse
import os
from typing import Optional, Dict

# GPU model to FP32 TFLOPs mapping (update as needed)
GPU_TFLOPS = {
    'RTX 3090': 35.58,
    'RTX 4090': 83.0,
    'A100': 19.5,       # FP32
    'A6000': 38.7,
    'V100': 15.7,
    'H100': 51.6,       # FP32
    'RTX 3080': 29.77,
    'TITAN RTX': 16.3,
    'RTX 3060': 12.74,
    'RTX 4070': 29.15,
    'A4000': 30.6,
}

def get_tflops(gpu_name: str) -> Optional[float]:
    """Get TFLOPs for a given GPU model name"""
    for model, tflops in GPU_TFLOPS.items():
        if model in gpu_name:
            return tflops
    return None

def get_best_offer(gpu_count: int) -> Optional[Dict]:
    """Find the best offer for a specific GPU count per machine based on price/TFLOPs"""
    cmd = f'vastai search offers "gpu_count={gpu_count} reliability>0.98" -o json --raw'
    print(f"Executing: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error searching offers: {result.stderr}")
            return None
            
        offers = json.loads(result.stdout)
        if not offers:
            print("No offers found")
            return None
            
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        print(f"Error processing offers: {e}")
        return None

    best_offer = None
    best_ppt = float('inf')

    for offer in offers:
        try:
            gpu_name = offer.get('gpu_name', 'Unknown')
            dph = float(offer.get('dph_total', 0))
            num_gpus = int(offer.get('gpu_count', 0))
            
            tflops = get_tflops(gpu_name)
            if not tflops or dph <= 0:
                continue

            total_tflops = num_gpus * tflops
            price_per_tflops = dph / total_tflops

            if price_per_tflops < best_ppt:
                best_ppt = price_per_tflops
                best_offer = {
                    'id': offer.get('id', 'N/A'),
                    'gpu_name': gpu_name,
                    'gpu_count': num_gpus,
                    'dph': dph,
                    'tflops_per_gpu': tflops,
                    'total_tflops': total_tflops,
                    'price_per_tflops': price_per_tflops,
                    'reliability': float(offer.get('reliability', 0)),
                    'provider': offer.get('provider', {}).get('name', 'Unknown')
                }
        except (ValueError, KeyError) as e:
            print(f"Skipping invalid offer: {e}")
            continue

    return best_offer

def get_best_instance_id(config_name: str) -> Optional[str]:
    """Get the instance ID for the best price/TFLOPs configuration"""
    scenarios = {
        '1x1': {'machines': 1, 'gpu_per_machine': 1},
        '2x1': {'machines': 2, 'gpu_per_machine': 1},
        '1x2': {'machines': 1, 'gpu_per_machine': 2},
        '1x4': {'machines': 1, 'gpu_per_machine': 4},
        '2x2': {'machines': 2, 'gpu_per_machine': 2}
    }
    
    if config_name not in scenarios:
        print(f"Invalid configuration: {config_name}")
        return None
        
    gpu_count = scenarios[config_name]['gpu_per_machine']
    best_offer = get_best_offer(gpu_count)
    
    if not best_offer:
        print(f"No valid offers found for {config_name}")
        return None

    print(f"\n=== Best {config_name} Configuration ===")
    print(f"GPU: {best_offer['gpu_name']} ({best_offer['gpu_count']}x)")
    print(f"Provider: {best_offer['provider']}")
    print(f"Price/hr: ${best_offer['dph']:.4f}")
    print(f"Total TFLOPs: {best_offer['total_tflops']:.2f}")
    print(f"Price/TFLOPs/hr: ${best_offer['price_per_tflops']:.6f}")
    print(f"Reliability: {best_offer['reliability']:.2%}")
    print(f"Offer ID: {best_offer['id']}")
    
    return best_offer['id']

def launch_instance(instance_id: str, image: str = "tensorflow/tensorflow:latest-gpu", disk: int = 20):
    """Launch a Vast.ai instance with verification"""
    if not instance_id or not instance_id.startswith('o-'):
        print("Invalid instance ID")
        return

    print(f"\nLaunch Configuration:")
    print(f"Offer ID: {instance_id}")
    print(f"Image: {image}")
    print(f"Disk: {disk}GB")
    
    confirm = input("\nConfirm launch (y/n): ").strip().lower()
    if confirm != 'y':
        print("Launch canceled")
        return

    cmd = f"vastai create instance {instance_id} --image {image} --disk {disk}"
    print(f"Executing: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("\nSuccessfully launched instance:")
            print(result.stdout)
        else:
            print("\nFailed to launch instance:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("Instance launch timed out")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAST.ai Price/TFLOPs Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", choices=['1x1', '2x1', '1x2', '1x4', '2x2'], 
                      help="Machine configuration (MxN = M machines with N GPUs each)")
    parser.add_argument("--launch", action="store_true", 
                      help="Launch instance after finding best offer")
    parser.add_argument("--image", default="tensorflow/tensorflow:latest-gpu",
                      help="Docker image for instance")
    parser.add_argument("--disk", type=int, default=20,
                      help="Disk space allocation in GB")
    
    args = parser.parse_args()
    
    if not os.environ.get("VAST_API_KEY"):
        print("Warning: VAST_API_KEY environment variable not set")
    
    offer_id = get_best_instance_id(args.config)
    
    if offer_id and args.launch:
        launch_instance(offer_id, args.image, args.disk)
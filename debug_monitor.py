#!/usr/bin/env python3

import os
import time
import psutil
import datetime
import signal
import sys
import glob
import subprocess
import re

# For NVIDIA GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil not found. To monitor GPU, install with: pip install gputil")

def find_processes():
    """Find all python processes related to our application"""
    our_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'run_master.py' in cmdline or 'run_slave.py' in cmdline:
                    our_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    return our_processes

def format_gpu_name(full_name):
    """Extract just the RTX model name from the full GPU name"""
    if "RTX" in full_name:
        # Look for RTX pattern
        parts = full_name.split()
        for i, part in enumerate(parts):
            if "RTX" in part:
                # Return "RTX XXXX" format
                if i+1 < len(parts) and parts[i+1].isdigit() or parts[i+1][0].isdigit():
                    return f"{parts[i]} {parts[i+1]}"
                else:
                    return parts[i]
    # Fallback to the original name, shortened
    return full_name.split()[-1] if full_name else "Unknown"

def get_gpu_info():
    """Get NVIDIA GPU information using GPUtil or nvidia-smi"""
    gpu_info = []
    
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memoryUsed': gpu.memoryUsed,
                    'memoryTotal': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            print(f"Error getting GPU info via GPUtil: {e}")
    else:
        # Fallback to nvidia-smi if GPUtil is not available
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                universal_newlines=True
            )
            
            for line in output.strip().split("\n"):
                values = [x.strip() for x in line.split(',')]
                if len(values) >= 6:
                    gpu_info.append({
                        'id': int(values[0]),
                        'name': values[1],
                        'load': float(values[2]),
                        'memoryUsed': float(values[3]),
                        'memoryTotal': float(values[4]),
                        'temperature': float(values[5])
                    })
        except Exception as e:
            print(f"Error getting GPU info via nvidia-smi: {e}")
    
    return gpu_info

def print_status(processes):
    """Print process status"""
    print(f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f"{'PID':<7} {'TYPE':<10} {'STATUS':<10} {'CPU%':<7} {'MEM%':<7} {'THREADS':<7} {'RUNTIME'}")
    print("-" * 80)
    
    for proc in processes:
        try:
            proc_type = "MASTER" if "run_master.py" in " ".join(proc.cmdline()) else "SLAVE"
            status = proc.status()
            cpu_percent = proc.cpu_percent()
            mem_percent = proc.memory_percent()
            num_threads = proc.num_threads()
            create_time = datetime.datetime.fromtimestamp(proc.create_time())
            runtime = datetime.datetime.now() - create_time
            
            print(f"{proc.pid:<7} {proc_type:<10} {status:<10} {cpu_percent:<7.1f} {mem_percent:<7.1f} {num_threads:<7} {str(runtime)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Display GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\nGPU Information:")
        print("-" * 80)
        print(f"{'ID':<3} {'Name':<20} {'Usage%':<7} {'Memory':<15} {'Temp(°C)':<8}")
        print("-" * 80)
        
        for gpu in gpu_info:
            mem_str = f"{gpu['memoryUsed']:.0f}MB/{gpu['memoryTotal']:.0f}MB"
            print(f"{gpu['id']:<3} {gpu['name'][:20]:<20} {gpu['load']:<7.1f} {mem_str:<15} {gpu['temperature']:<8.1f}")
    
    # Update log file paths
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_files = [
        os.path.join(log_dir, 'debug.log'), 
        os.path.join(log_dir, 'master.log'),
        os.path.join(log_dir, 'slave.log')
    ]

    # Check for logs and print their status
    print("\nLog Files:")
    print("-" * 80)
    for log in log_files:
        if os.path.exists(log):
            size = os.path.getsize(log) / 1024  # KB
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(log))
            log_name = os.path.basename(log)  # Just show filename, not full path
            print(f"{log_name}: {size:.1f} KB, Last modified: {mtime.strftime('%H:%M:%S')}")

def extract_training_progress(log_files):
    """Extract epoch and training progress information from logs"""
    training_info = {}
    for log in log_files:
        if not os.path.exists(log):
            continue
            
        try:
            with open(log, 'r') as f:
                content = f.read()
                
                # Extract the latest epoch info
                epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', content)
                if epoch_matches:
                    current, total = epoch_matches[-1]  # Get the last match
                    training_info['epoch'] = f"Epoch {current}/{total}"
                
                # Extract the latest step progress
                progress_lines = re.findall(r'(\d+)/(\d+) [━\s]+.+accuracy: ([\d\.]+) - loss: ([\d\.]+)', content)
                if progress_lines:
                    # Get the latest progress line
                    current, total, acc, loss = progress_lines[-1]
                    training_info['progress'] = {
                        'current': int(current),
                        'total': int(total),
                        'accuracy': float(acc),
                        'loss': float(loss)
                    }
            
            if 'epoch' in training_info:
                # Found what we need, no need to check other logs
                break
                
        except Exception as e:
            print(f"Error extracting training progress: {e}")
    
    return training_info

def extract_model_architecture(log_files):
    """Extract neural network architecture information from logs"""
    architecture_info = {
        'found': False,
        'total_params': 0,
        'trainable_params': 0,
        'layer_counts': {},
        'model_name': '',
        'build_time': 0
    }
    
    for log in log_files:
        if not os.path.exists(log):
            continue
            
        try:
            with open(log, 'r') as f:
                content = f.read()
                
                # Extract model name
                model_name_match = re.search(r'Model: "([^"]+)"', content)
                if model_name_match:
                    architecture_info['model_name'] = model_name_match.group(1)
                    architecture_info['found'] = True
                
                # Extract build time
                build_time_match = re.search(r'Model building took (\d+) \(miliseconds\)', content)
                if build_time_match:
                    architecture_info['build_time'] = int(build_time_match.group(1))
                
                # Extract layer info
                layer_pattern = r'│ ([^\(]+)\s+\(([^\)]+)\)'
                layers = re.findall(layer_pattern, content)
                for layer_name, layer_type in layers:
                    layer_name = layer_name.strip()
                    layer_type = layer_type.strip()
                    
                    if layer_type in architecture_info['layer_counts']:
                        architecture_info['layer_counts'][layer_type] += 1
                    else:
                        architecture_info['layer_counts'][layer_type] = 1
                
                # Extract parameter info
                params_match = re.search(r'Total params: ([\d,]+) \(([\d\.]+) MB\)', content)
                if params_match:
                    architecture_info['total_params'] = int(params_match.group(1).replace(',', ''))
                    architecture_info['params_mb'] = params_match.group(2)
                
                trainable_match = re.search(r'Trainable params: ([\d,]+) \(([\d\.]+) MB\)', content)
                if trainable_match:
                    architecture_info['trainable_params'] = int(trainable_match.group(1).replace(',', ''))
                
                # Extract the GPU count
                gpu_match = re.search(r'Total weights \d+ using (\d+) GPU\(s\)', content)
                if gpu_match:
                    architecture_info['gpu_count'] = int(gpu_match.group(1))
                
                if architecture_info['found']:
                    break  # Stop once we've found architecture info
                    
        except Exception as e:
            print(f"Error extracting model architecture: {e}")
    
    return architecture_info

def save_architecture(architecture_info):
    """Save the current architecture to a file for future comparison"""
    if not architecture_info['found']:
        return
        
    # Create directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to file
    save_path = os.path.join(save_dir, "last_architecture.json")
    try:
        import json
        with open(save_path, 'w') as f:
            json.dump(architecture_info, f)
    except Exception as e:
        print(f"Error saving architecture: {e}")

def load_last_architecture():
    """Load the last saved architecture for comparison"""
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "last_architecture.json")
    
    if not os.path.exists(save_path):
        return None
        
    try:
        import json
        with open(save_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading previous architecture: {e}")
        return None

def compare_architectures(current, previous):
    """Compare current and previous architecture and return differences"""
    if not current['found'] or not previous:
        return {}
        
    differences = {}
    
    # Compare basic metrics
    for key in ['total_params', 'trainable_params', 'gpu_count']:
        if key in current and key in previous and current[key] != previous[key]:
            differences[key] = {
                'current': current[key],
                'previous': previous[key],
                'diff': current[key] - previous[key] if isinstance(current[key], (int, float)) else None
            }
    
    # Compare layer counts
    layer_diffs = {}
    all_layer_types = set(list(current['layer_counts'].keys()) + list(previous.get('layer_counts', {}).keys()))
    
    for layer_type in all_layer_types:
        current_count = current['layer_counts'].get(layer_type, 0)
        previous_count = previous.get('layer_counts', {}).get(layer_type, 0)
        
        if current_count != previous_count:
            layer_diffs[layer_type] = {
                'current': current_count,
                'previous': previous_count,
                'diff': current_count - previous_count
            }
    
    if layer_diffs:
        differences['layer_counts'] = layer_diffs
        
    return differences

def print_network_summary(architecture_info):
    """Display a formatted summary of the neural network architecture"""
    if not architecture_info['found']:
        return
    
    # Load previous architecture
    previous_architecture = load_last_architecture()
    
    # Compare architectures
    differences = compare_architectures(architecture_info, previous_architecture)
    
    # Save current architecture for future comparison
    save_architecture(architecture_info)
    
    print("\n🧠 Neural Network Architecture:")
    print("-" * 80)
    print(f"Model: {architecture_info['model_name']} (built in {architecture_info['build_time']}ms)")
    
    # Display layer types summary
    print("\nLayer composition:")
    for layer_type, count in sorted(architecture_info['layer_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)[:6]:  # Show top 6 layer types
        if 'layer_counts' in differences and layer_type in differences['layer_counts']:
            diff = differences['layer_counts'][layer_type]['diff']
            diff_str = f"({'+' if diff > 0 else ''}{diff})" if diff != 0 else ""
            print(f"• {layer_type}: {count} \033[1;{'32' if diff > 0 else '31'}\033[0m {diff_str}")
        else:
            print(f"• {layer_type}: {count}")
    
    # Display parameter information
    if architecture_info['total_params'] > 0:
        params_diff = ""
        if 'total_params' in differences:
            diff = differences['total_params']['diff']
            params_diff = f" \033[1;{'32' if diff > 0 else '31'}\033[0m ({'+' if diff > 0 else ''}{diff:,})"
            
        print(f"\nParameters: {architecture_info['total_params']:,} total ({architecture_info['params_mb']} MB){params_diff}")
        
        trainable_percent = (architecture_info['trainable_params'] / architecture_info['total_params']) * 100
        
        trainable_diff = ""
        if 'trainable_params' in differences:
            diff = differences['trainable_params']['diff']
            trainable_diff = f" \033[1;{'32' if diff > 0 else '31'}\033[0m ({'+' if diff > 0 else ''}{diff:,})"
            
        print(f"Training: {architecture_info['trainable_params']:,} trainable ({trainable_percent:.1f}%){trainable_diff}")
    
    # Display GPUs if available
    if 'gpu_count' in architecture_info:
        gpu_diff = ""
        if 'gpu_count' in differences:
            diff = differences['gpu_count']['diff']
            gpu_diff = f" \033[1;{'32' if diff > 0 else '31'}\033[0m ({'+' if diff > 0 else ''}{diff})"
            
        print(f"Hardware: Training on {architecture_info['gpu_count']} GPU(s){gpu_diff}")
    
    # Show overall comparison summary if there are differences
    if differences and previous_architecture:
        print("\n📊 Architecture Changes:")
        if 'total_params' in differences:
            param_diff = differences['total_params']['diff']
            percent = (param_diff / previous_architecture['total_params']) * 100 if previous_architecture['total_params'] > 0 else 0
            print(f"• Parameters: {'Increased' if param_diff > 0 else 'Decreased'} by {abs(param_diff):,} ({abs(percent):.1f}%)")
        
        if 'layer_counts' in differences:
            added = []
            removed = []
            changed = []
            
            for layer, diff in differences['layer_counts'].items():
                if diff['previous'] == 0:
                    added.append(f"{layer} ({diff['current']})")
                elif diff['current'] == 0:
                    removed.append(f"{layer} ({diff['previous']})")
                else:
                    changed.append(f"{layer} ({diff['previous']} → {diff['current']})")
            
            if added:
                print(f"• Added layers: {', '.join(added)}")
            if removed:
                print(f"• Removed layers: {', '.join(removed)}")
            if changed:
                print(f"• Changed layers: {', '.join(changed)}")

def tail_logs(num_lines=10):
    """Show recent log entries"""
    # Update log file paths
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_files = [
        os.path.join(log_dir, 'debug.log'), 
        os.path.join(log_dir, 'master.log'),
        os.path.join(log_dir, 'slave.log')
    ]
    
    # For optimization logs, check all files matching the pattern
    optimization_logs = glob.glob(os.path.join(log_dir, 'optimization_job*.log'))
    if optimization_logs:
        log_files.extend(sorted(optimization_logs, key=os.path.getmtime, reverse=True)[:1])
    
    # Extract and display network architecture summary
    architecture_info = extract_model_architecture(log_files)
    if architecture_info['found']:
        print_network_summary(architecture_info)
    
    # Extract training progress
    training_info = extract_training_progress(log_files)
    if training_info:
        print("\n🔄 Training Progress:")
        print("-" * 80)
        if 'epoch' in training_info:
            print(f"Current: {training_info['epoch']}")
        if 'progress' in training_info:
            prog = training_info['progress']
            percent = (prog['current'] / prog['total']) * 100
            bar_length = 40
            filled_length = int(bar_length * prog['current'] // prog['total'])
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"Steps: {prog['current']}/{prog['total']} [{bar}] {percent:.1f}%")
            print(f"Metrics: accuracy={prog['accuracy']:.4f}, loss={prog['loss']:.4f}")
    
    # Show last few lines of logs
    for log in log_files:
        if os.path.exists(log):
            log_name = os.path.basename(log)
            print(f"\nLast {num_lines} lines from {log_name}:")
            print("-" * 80)
            try:
                with open(log, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-num_lines:]:
                        # Highlight epoch lines
                        if "Epoch" in line and "/" in line:
                            print(f"\033[1;32m{line.strip()}\033[0m")  # Green for epoch lines
                        else:
                            print(line.strip())
            except Exception as e:
                print(f"Error reading log file: {e}")

def signal_handler(sig, frame):
    print("\nExiting monitor...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Process Monitor for MLOptimizer")
    print("Press Ctrl+C to exit")
    
    while True:
        os.system('clear')  # Clear screen
        processes = find_processes()
        print_status(processes)
        tail_logs(5)
        time.sleep(2)

if __name__ == "__main__":
    main()
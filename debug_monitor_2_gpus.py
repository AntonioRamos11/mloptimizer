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
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'environ']):
        try:
            if 'python' in proc.info['name'] and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'run_master.py' in cmdline or 'run_slave.py' in cmdline:
                    # Try to get GPU assignment from environment
                    gpu_id = None
                    try:
                        env = proc.info.get('environ', {})
                        if env and 'CUDA_VISIBLE_DEVICES' in env:
                            gpu_id = env['CUDA_VISIBLE_DEVICES']
                    except (psutil.AccessDenied, KeyError):
                        pass
                    
                    proc.gpu_id = gpu_id
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
    print("-" * 90)
    print(f"{'PID':<7} {'TYPE':<10} {'GPU':<5} {'STATUS':<10} {'CPU%':<7} {'MEM%':<7} {'THREADS':<7} {'RUNTIME'}")
    print("-" * 90)
    
    for proc in processes:
        try:
            proc_type = "MASTER" if "run_master.py" in " ".join(proc.cmdline()) else "SLAVE"
            status = proc.status()
            cpu_percent = proc.cpu_percent()
            mem_percent = proc.memory_percent()
            num_threads = proc.num_threads()
            create_time = datetime.datetime.fromtimestamp(proc.create_time())
            runtime = datetime.datetime.now() - create_time
            
            # Get GPU assignment
            gpu_str = getattr(proc, 'gpu_id', 'N/A') or 'N/A'
            
            print(f"{proc.pid:<7} {proc_type:<10} {gpu_str:<5} {status:<10} {cpu_percent:<7.1f} {mem_percent:<7.1f} {num_threads:<7} {str(runtime)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Display GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\n🎮 GPU Information:")
        print("-" * 90)
        print(f"{'ID':<3} {'Name':<25} {'Usage%':<8} {'Memory':<20} {'Temp(°C)':<8}")
        print("-" * 90)
        
        for gpu in gpu_info:
            mem_str = f"{gpu['memoryUsed']:.0f}MB/{gpu['memoryTotal']:.0f}MB"
            mem_percent = (gpu['memoryUsed'] / gpu['memoryTotal']) * 100 if gpu['memoryTotal'] > 0 else 0
            
            # Color code GPU usage
            usage_color = ""
            if gpu['load'] >= 90:
                usage_color = "\033[1;32m"  # Green for 90%+
            elif gpu['load'] >= 70:
                usage_color = "\033[1;33m"  # Yellow for 70-89%
            elif gpu['load'] < 30:
                usage_color = "\033[1;31m"  # Red for <30%
            
            print(f"{gpu['id']:<3} {format_gpu_name(gpu['name']):<25} {usage_color}{gpu['load']:<8.1f}\033[0m {mem_str:<20} {gpu['temperature']:<8.1f}")
    
    # Update log file paths - now including per-GPU slave logs
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_files = [
        os.path.join(log_dir, 'debug.log'), 
        os.path.join(log_dir, 'master.log'),
    ]
    
    # Add all slave_gpuX.log files
    slave_logs = sorted(glob.glob(os.path.join(log_dir, 'slave_gpu*.log')))
    if slave_logs:
        log_files.extend(slave_logs)
    else:
        # Fallback to generic slave.log if no GPU-specific logs found
        generic_slave_log = os.path.join(log_dir, 'slave.log')
        if os.path.exists(generic_slave_log):
            log_files.append(generic_slave_log)

    # Check for logs and print their status
    print("\n📄 Log Files:")
    print("-" * 90)
    for log in log_files:
        if os.path.exists(log):
            size = os.path.getsize(log) / 1024  # KB
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(log))
            log_name = os.path.basename(log)  # Just show filename, not full path
            
            # Highlight active logs (modified in last 5 seconds)
            time_diff = (datetime.datetime.now() - mtime).total_seconds()
            active_marker = "🟢" if time_diff < 5 else "  "
            
            print(f"{active_marker} {log_name:<20}: {size:>8.1f} KB, Last: {mtime.strftime('%H:%M:%S')}")

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
                
                # Extract model ID being trained
                model_id_match = re.search(r'Starting training for model (\d+)', content)
                if model_id_match:
                    training_info['model_id'] = int(model_id_match.group(1))
                
                # Extract dataset name
                dataset_match = re.search(r'Dataset: (\w+)', content)
                if dataset_match:
                    training_info['dataset'] = dataset_match.group(1)
            
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
                params_match = re.search(r'Total params: ([\d,]+) \(([\d\.]+) [MKB]+\)', content)
                if params_match:
                    architecture_info['total_params'] = int(params_match.group(1).replace(',', ''))
                    architecture_info['params_mb'] = params_match.group(2)
                
                trainable_match = re.search(r'Trainable params: ([\d,]+) \(([\d\.]+) [MKB]+\)', content)
                if trainable_match:
                    architecture_info['trainable_params'] = int(trainable_match.group(1).replace(',', ''))
                
                # Extract the GPU count
                gpu_match = re.search(r'Using (\w+Strategy) with (\d+) GPU', content)
                if gpu_match:
                    architecture_info['strategy'] = gpu_match.group(1)
                    architecture_info['gpu_count'] = int(gpu_match.group(2))
                else:
                    # Check for single GPU
                    single_gpu_match = re.search(r'Using GPU (\d+):', content)
                    if single_gpu_match:
                        architecture_info['gpu_count'] = 1
                        architecture_info['strategy'] = 'OneDeviceStrategy'
                
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
    print("-" * 90)
    print(f"Model: {architecture_info['model_name']} (built in {architecture_info['build_time']}ms)")
    
    # Display strategy info
    if 'strategy' in architecture_info:
        print(f"Strategy: {architecture_info['strategy']}")
    
    # Display layer types summary
    print("\nLayer composition:")
    for layer_type, count in sorted(architecture_info['layer_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)[:6]:  # Show top 6 layer types
        if 'layer_counts' in differences and layer_type in differences['layer_counts']:
            diff = differences['layer_counts'][layer_type]['diff']
            diff_str = f"({'+' if diff > 0 else ''}{diff})" if diff != 0 else ""
            print(f"• {layer_type}: {count} \033[1;{'32' if diff > 0 else '31'}m{diff_str}\033[0m")
        else:
            print(f"• {layer_type}: {count}")
    
    # Display parameter information
    if architecture_info['total_params'] > 0:
        params_diff = ""
        if 'total_params' in differences:
            diff = differences['total_params']['diff']
            params_diff = f" \033[1;{'32' if diff > 0 else '31'}m({'+' if diff > 0 else ''}{diff:,})\033[0m"
            
        print(f"\nParameters: {architecture_info['total_params']:,} total ({architecture_info.get('params_mb', 'N/A')} MB){params_diff}")
        
        if architecture_info['trainable_params'] > 0:
            trainable_percent = (architecture_info['trainable_params'] / architecture_info['total_params']) * 100
            
            trainable_diff = ""
            if 'trainable_params' in differences:
                diff = differences['trainable_params']['diff']
                trainable_diff = f" \033[1;{'32' if diff > 0 else '31'}m({'+' if diff > 0 else ''}{diff:,})\033[0m"
                
            print(f"Training: {architecture_info['trainable_params']:,} trainable ({trainable_percent:.1f}%){trainable_diff}")
    
    # Display GPUs if available
    if 'gpu_count' in architecture_info:
        gpu_diff = ""
        if 'gpu_count' in differences:
            diff = differences['gpu_count']['diff']
            gpu_diff = f" \033[1;{'32' if diff > 0 else '31'}m({'+' if diff > 0 else ''}{diff})\033[0m"
            
        print(f"Hardware: Training on {architecture_info['gpu_count']} GPU(s){gpu_diff}")

def tail_logs(num_lines=8):
    """Show recent log entries from all slave logs"""
    # Update log file paths
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # Get all slave GPU logs
    slave_logs = sorted(glob.glob(os.path.join(log_dir, 'slave_gpu*.log')))
    
    if not slave_logs:
        # Fallback to generic logs
        slave_logs = [os.path.join(log_dir, 'slave.log')]
    
    # Add master log
    log_files = [os.path.join(log_dir, 'master.log')] + slave_logs
    
    # Extract and display network architecture summary
    architecture_info = extract_model_architecture(log_files)
    if architecture_info['found']:
        print_network_summary(architecture_info)
    
    # Extract training progress from each GPU's log
    print("\n🔄 Training Progress per GPU:")
    print("-" * 90)
    
    for log in slave_logs:
        if not os.path.exists(log):
            continue
            
        gpu_num = re.search(r'slave_gpu(\d+)\.log', log)
        gpu_id = gpu_num.group(1) if gpu_num else "?"
        
        training_info = extract_training_progress([log])
        if training_info:
            model_str = f"Model {training_info.get('model_id', '?')}" if 'model_id' in training_info else "Model ?"
            dataset_str = f"on {training_info.get('dataset', '?')}" if 'dataset' in training_info else ""
            
            print(f"\n🎮 GPU {gpu_id}: {model_str} {dataset_str}")
            
            if 'epoch' in training_info:
                print(f"   {training_info['epoch']}")
            if 'progress' in training_info:
                prog = training_info['progress']
                percent = (prog['current'] / prog['total']) * 100
                bar_length = 30
                filled_length = int(bar_length * prog['current'] // prog['total'])
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"   Steps: {prog['current']}/{prog['total']} [{bar}] {percent:.1f}%")
                print(f"   Metrics: acc={prog['accuracy']:.4f}, loss={prog['loss']:.4f}")
        else:
            print(f"\n🎮 GPU {gpu_id}: No active training")
    
    # Show last few lines of each log
    print("\n📝 Recent Log Entries:")
    print("-" * 90)
    
    for log in log_files:
        if os.path.exists(log):
            log_name = os.path.basename(log)
            print(f"\n{log_name}:")
            print("─" * 90)
            try:
                with open(log, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-num_lines:]:
                        # Highlight important lines
                        if "Epoch" in line and "/" in line:
                            print(f"\033[1;32m{line.strip()}\033[0m")  # Green for epoch lines
                        elif "ERROR" in line.upper():
                            print(f"\033[1;31m{line.strip()}\033[0m")  # Red for errors
                        elif "GPU" in line and "%" in line:
                            print(f"\033[1;36m{line.strip()}\033[0m")  # Cyan for GPU info
                        else:
                            print(line.strip())
            except Exception as e:
                print(f"Error reading log file: {e}")

def signal_handler(sig, frame):
    print("\nExiting monitor...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Process Monitor for MLOptimizer (Multi-GPU Edition)")
    print("Press Ctrl+C to exit")
    
    while True:
        os.system('clear')  # Clear screen
        processes = find_processes()
        print_status(processes)
        tail_logs(5)
        time.sleep(2)

if __name__ == "__main__":
    main()
import platform
import subprocess
import os
import json

def get_hardware_info():
    """
    Gather hardware information about the system
    
    Returns:
        dict: Dictionary containing hardware information
    """
    hardware_info = {}
    
    # CPU information
    hardware_info["cpu_model"] = platform.processor()
    hardware_info["cpu_cores"] = os.cpu_count()
    hardware_info["python_version"] = platform.python_version()
    hardware_info["system"] = platform.system()
    
    # RAM information
    try:
        import psutil
        ram = psutil.virtual_memory()
        hardware_info["ram_total"] = f"{ram.total / (1024**3):.2f} GB"
        hardware_info["ram_available"] = f"{ram.available / (1024**3):.2f} GB"
    except ImportError:
        hardware_info["ram_total"] = "Unknown (psutil not installed)"
    
    # GPU information
    try:
        # Try to get GPU info using NVIDIA tools
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader"], 
            universal_newlines=True
        )
        gpus = []
        for line in nvidia_output.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) >= 2:
                name, memory = parts[0], parts[1]
                driver = parts[2] if len(parts) > 2 else "Unknown"
                gpus.append({"model": name, "memory": memory, "driver": driver})
        
        hardware_info["gpu_count"] = len(gpus)
        hardware_info["gpus"] = gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If nvidia-smi isn't available or fails, try TensorFlow
        try:
            import tensorflow as tf
            physical_gpus = tf.config.list_physical_devices('GPU')
            hardware_info["gpu_count"] = len(physical_gpus)
            hardware_info["gpus"] = [{"device": str(gpu)} for gpu in physical_gpus]
            
            # Try to get more GPU details
            if len(physical_gpus) > 0:
                for i, gpu in enumerate(physical_gpus):
                    with tf.device(f'/GPU:{i}'):
                        gpu_name = tf.test.gpu_device_name()
                        if i < len(hardware_info["gpus"]):
                            hardware_info["gpus"][i]["name"] = gpu_name
        except:
            hardware_info["gpu_count"] = 0
            hardware_info["gpus"] = []
    
    return hardware_info

def check_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        import tensorflow as tf
        import nvidia_smi
        
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        print("\nGPU Memory Information:")
        print(f"Total memory: {info.total / 1024**2:.2f} MB")
        print(f"Used memory: {info.used / 1024**2:.2f} MB")
        print(f"Free memory: {info.free / 1024**2:.2f} MB")
        print(f"Memory usage: {info.used / info.total * 100:.2f}%\n")
        
        return info.used / info.total  # Return memory usage percentage
    except:
        print("Could not access GPU memory information")
        return 0.0
print(json.dumps(get_hardware_info(), indent=2))
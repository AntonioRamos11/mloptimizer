import tensorflow as tf
import time
import numpy as np
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

class GPUMonitor:
    def __init__(self, gpu_indices):
        nvmlInit()
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]
        
    def get_metrics(self):
        metrics = {
            'memory_used': [],
            'utilization': []
        }
        for handle in self.handles:
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            metrics['memory_used'].append(mem_info.used / 1024**3)  # in GB
            util = nvmlDeviceGetUtilizationRates(handle)
            metrics['utilization'].append(util.gpu)
        return metrics

def create_model(input_dim=2048):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(4096, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

def benchmark(config_name, strategy, batch_size, input_dim=2048):
    # Initialize monitoring
    gpu_indices = [0] if config_name == "single" else [0, 1]
    monitor = GPUMonitor(gpu_indices)
    
    # Create dataset
    num_samples = 100000
    data = tf.random.normal((num_samples, input_dim))
    labels = tf.random.normal((num_samples, 1))
    
    with strategy.scope():
        model = create_model(input_dim)
        model.compile(optimizer='adam', loss='mse')
    
    # Warm-up run
    model.fit(data, labels, epochs=1, batch_size=batch_size, verbose=0)
    
    metrics = {
        'epoch_times': [],
        'memory_usage': [],
        'gpu_utilization': [],
        'throughput': []
    }
    
    start_time = time.time()
    history = model.fit(
        data, labels, 
        epochs=10, 
        batch_size=batch_size, 
        verbose=0,
        callbacks=[tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: metrics['epoch_times'].append(time.time()),
            on_batch_end=lambda batch, logs: metrics['throughput'].append(batch_size/(time.time()-start_time))
        )]  # Close the callbacks list with ]
    )  # Close the model.fit() call with )
    
    # Collect metrics
    total_time = time.time() - start_time
    metrics['total_time'] = total_time
    metrics['avg_epoch_time'] = np.mean(np.diff(metrics['epoch_times']))
    metrics['avg_throughput'] = np.mean(metrics['throughput'])
    
    # Get final memory usage
    gpu_metrics = monitor.get_metrics()
    metrics['max_memory'] = np.max(gpu_metrics['memory_used'])
    metrics['avg_utilization'] = np.mean(gpu_metrics['utilization'])
    
    return metrics

def print_metrics(config, metrics):
    print(f"\n{config} GPU Configuration Results:")
    print(f"Total Training Time: {metrics['total_time']:.2f}s")
    print(f"Average Epoch Time: {metrics['avg_epoch_time']:.2f}s")
    print(f"Throughput: {metrics['avg_throughput']:.2f} samples/sec")
    print(f"Max GPU Memory Used: {metrics['max_memory']:.2f} GB")
    print(f"Average GPU Utilization: {metrics['avg_utilization']:.2f}%")
    if len(metrics.get('memory_usage', [])) > 0:
        print(f"Peak Memory per GPU: {[f'{m:.2f}GB' for m in metrics['memory_usage']]}")

# Main execution
gpus = tf.config.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Detected {num_gpus} GPUs")

results = {}

if num_gpus >= 1:
    # Single GPU
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    results['single'] = benchmark("single", strategy, batch_size=256)

if num_gpus >= 2:
    # Multi-GPU
    strategy = tf.distribute.MirroredStrategy()
    results['multi'] = benchmark("multi", strategy, batch_size=256*2)

# Print comparison
if 'single' in results and 'multi' in results:
    print("\nComparative Analysis:")
    print(f"Speed Ratio: {results['single']['total_time']/results['multi']['total_time']:.2f}x")
    print(f"Throughput Ratio: {results['multi']['avg_throughput']/results['single']['avg_throughput']:.2f}x")
    print(f"Memory Efficiency: {results['multi']['max_memory']/results['single']['max_memory']:.2f}x")
    
    if results['multi']['total_time'] > results['single']['total_time']:
        print("\nWARNING: Multi-GPU configuration is slower than Single-GPU!")
        print("Possible issues:")
        print("- Insufficient batch size")
        print("- GPU communication overhead")
        print("- Memory bandwidth limitations")
        print("- Suboptimal distribution strategy")

# Print individual metrics
for config, metrics in results.items():
    print_metrics(config.capitalize(), metrics)
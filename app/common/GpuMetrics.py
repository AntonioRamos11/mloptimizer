import io
import os
import time
import json
import psutil
import numpy as np
from datetime import datetime
import subprocess
from typing import Dict, List, Any, Optional

class GPUMetricsCollector:
    def __init__(self, framework='tensorflow',min_collection_interval=1.0):
        self.framework = framework
        self.gpu_devices = self._detect_gpus()
        self.metrics_history = []
        self.last_collection_time = None
        self.idle_start_time = time.time()  # Track when GPU went idle
        self.is_idle = True
        self.model_architecture = None

        self.min_collection_interval = min_collection_interval
        self.last_save_time = time.time()
        self.save_interval = 60  # Default: save every 60 seconds
        self.pending_save = False
        self.auto_save_path = None
        self.auto_save_enabled = False
        self.max_history_size = 1000  # Maximum number of metrics to keep in memory
            
    def _detect_gpus(self):
        """Detect available GPUs and their capabilities"""
        devices = []
        
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                for i, gpu in enumerate(gpus):
                    # Get detailed info using nvidia-smi
                    try:
                        output = subprocess.check_output(
                            [
                                "nvidia-smi", 
                                f"--id={i}", 
                                "--query-gpu=name,memory.total,compute_cap,uuid", 
                                "--format=csv,noheader"
                            ], 
                            universal_newlines=True
                        ).strip()
                        
                        name, memory, compute_cap, uuid = output.split(",")
                        devices.append({
                            'index': i,
                            'name': name.strip(),
                            'memory_total': memory.strip(),
                            'compute_capability': compute_cap.strip(),
                            'uuid': uuid.strip(),
                            'tf_device': str(gpu)
                        })
                    except Exception as e:
                        devices.append({
                            'index': i,
                            'name': f"GPU {i}",
                            'tf_device': str(gpu),
                            'error': str(e)
                        })
            except Exception as e:
                print(f"Error detecting TensorFlow GPUs: {e}")
                
        return devices
    
    def register_model(self, model):
        """Register a neural network model to collect its architecture"""
        self.model_architecture = self._extract_model_architecture(model)
        
    def _extract_model_architecture(self, model, include_text_summary=False):
        """Extract architecture details from a model
        
        Args:
            model: The TensorFlow model to analyze
            include_text_summary: Whether to include the text-based summary with box characters
        """
        if self.framework == 'tensorflow':
            try:
                # Get layer details
                layers = []
                for i, layer in enumerate(model.layers):
                    # Fix: Safely get output shape to avoid attribute error
                    try:
                        output_shape = str(layer.output.shape)
                    except (AttributeError, ValueError):
                        try:
                            output_shape = str(layer.get_output_shape_at(0))
                        except (AttributeError, ValueError):
                            output_shape = "Unknown"
                    
                    layer_info = {
                        'name': layer.name,
                        'type': layer.__class__.__name__,
                        'shape': output_shape,  # Using the safely obtained shape
                        'params': layer.count_params()
                    }
                    
                    # Get additional config if available
                    if hasattr(layer, 'get_config'):
                        try:
                            config = layer.get_config()
                            # Extract key parameters from config
                            if 'filters' in config:
                                layer_info['filters'] = config['filters']
                            if 'kernel_size' in config:
                                layer_info['kernel_size'] = config['kernel_size']
                            if 'units' in config:
                                layer_info['units'] = config['units']
                            if 'activation' in config:
                                layer_info['activation'] = config['activation']
                        except:
                            pass
                            
                    layers.append(layer_info)
                
                # Overall model stats
                total_params = model.count_params()
                trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
                non_trainable_params = total_params - trainable_params
                
                result = {
                    'layers': layers,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'non_trainable_params': non_trainable_params,
                    'input_shape': str(model.input_shape),
                    'output_shape': str(model.output_shape)
                }
                
                # Only include text summary if requested
                if include_text_summary:
                    # Get model summary as string
                    string_io = io.StringIO()
                    model.summary(print_fn=lambda x: string_io.write(x + '\n'))
                    summary_string = string_io.getvalue()
                    result['summary'] = summary_string
                    
                return result
            except Exception as e:
                return {'error': str(e)}
        return None
        
    def collect_metrics(self, force=False):
        """
        Collect comprehensive GPU, CPU, and latency metrics with rate limiting
        """
        current_time = time.time()
        
        # Check if we should collect (rate limiting)
        if not force and self.last_collection_time and \
           (current_time - self.last_collection_time) < self.min_collection_interval:
            # Too soon since last collection
            return None
        
        # Get GPU utilization first
        gpu_util = self._get_gpu_utilization()
        
        # Calculate idle time if we were idle
        idle_time = 0
        if self.is_idle and self.last_collection_time:
            idle_time = current_time - self.idle_start_time
        
        # Update last collection time
        self.last_collection_time = current_time
        
        # AFTER calculating idle time, now check if we should change idle state
        # If all GPUs have utilization below threshold, mark as idle
        threshold = 5  # Consider idle if below 5%
        all_idle = True
        for gpu_id, util in gpu_util.items():
            if isinstance(util, (int, float)) and util >= threshold:
                all_idle = False
                break
        
        # Update idle state but don't reset it automatically
        if all_idle and not self.is_idle:
            self.idle_start_time = current_time
            self.is_idle = True
        elif not all_idle and self.is_idle:
            self.is_idle = False
        
        # Note: We DON'T set self.is_idle = False here unconditionally

        # Collect all metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'gpu': {
                'utilization': gpu_util,  # Reuse the utilization we already got
                'memory': self._get_memory_usage(),
                'temperature': self._get_temperature(),
                'power': self._get_power_usage(),
                'clock_speeds': self._get_clock_speeds(),
                'pcie_throughput': self._get_pcie_throughput(),
                'idle_time': idle_time,
                'is_currently_idle': self.is_idle  # Add this for better tracking
            },
            'cpu': {
                'utilization': self._get_cpu_utilization(),
                'unused_cores': self._get_unused_cpu_cores(),
                'memory': self._get_cpu_memory()
            },
            'latency': {
                'inference': self._get_inference_latency(),
                'training_step': self._get_training_step_latency(),
                'data_pipeline': self._get_data_pipeline_latency()
            },
            'framework_specific': self._get_framework_metrics()
        }
        
        self.metrics_history.append(metrics)
        
        # Limit history size to prevent memory issues
        if len(self.metrics_history) > self.max_history_size:
            # Remove oldest items
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # Check if we should auto-save
        if self.auto_save_enabled and \
           (current_time - self.last_save_time) > self.save_interval:
            self.pending_save = True
            
        # Perform auto-save if needed
        if self.pending_save and self.auto_save_path:
            self.save_organized_metrics(
                base_dir=self.auto_save_path,
                clear_after_save=True
            )
            self.last_save_time = current_time
            self.pending_save = False
        
        # Get GPU utilization
        
        
        return metrics
    
    def mark_idle(self):
        """Mark the GPU as idle and start tracking idle time"""
        if not self.is_idle:
            self.idle_start_time = time.time()
            self.is_idle = True
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                utilization = int(output.strip(' %'))
                results[index] = utilization
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def _get_memory_usage(self):
        """Get GPU memory usage"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                used_str, total_str = output.split(',')
                used = int(used_str.strip(' MiB'))
                total = int(total_str.strip(' MiB'))
                results[index] = {
                    'used_mb': used,
                    'total_mb': total,
                    'percentage': (used / total) * 100 if total > 0 else 0
                }
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def _get_cpu_utilization(self):
        """Get CPU utilization metrics"""
        try:
            # Overall CPU utilization
            overall = psutil.cpu_percent(interval=0.1)
            
            # Per-core utilization
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            return {
                'overall': overall,
                'per_core': per_core,
                'logical_cores': len(per_core),
                'physical_cores': psutil.cpu_count(logical=False)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_unused_cpu_cores(self):
        """Get information about unused CPU cores"""
        try:
            # Get per-core utilization
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Define threshold for "unused" (e.g., less than 10% usage)
            threshold = 10
            unused_cores = [i for i, usage in enumerate(per_core) if usage < threshold]
            
            return {
                'unused_core_count': len(unused_cores),
                'unused_core_indices': unused_cores,
                'threshold': threshold
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_cpu_memory(self):
        """Get CPU memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_temperature(self):
        """Get GPU temperature"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                temperature = int(output.strip(' C'))
                results[index] = temperature
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def _get_power_usage(self):
        """Get GPU power usage"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=power.draw,power.limit", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                draw_str, limit_str = output.split(',')
                draw = float(draw_str.strip(' W'))
                limit = float(limit_str.strip(' W'))
                results[index] = {
                    'draw_watts': draw,
                    'limit_watts': limit,
                    'percentage': (draw / limit) * 100 if limit > 0 else 0
                }
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def _get_clock_speeds(self):
        """Get GPU clock speeds"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=clocks.gr,clocks.mem", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                gr_clock_str, mem_clock_str = output.split(',')
                gr_clock = int(gr_clock_str.strip(' MHz'))
                mem_clock = int(mem_clock_str.strip(' MHz'))
                results[index] = {
                    'graphics_mhz': gr_clock,
                    'memory_mhz': mem_clock
                }
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def _get_pcie_throughput(self):
        """Get PCIe throughput stats using memory controller activity as proxy"""
        results = {}
        try:
            for device in self.gpu_devices:
                index = device['index']
                # Get PCIe link configuration first (indicates maximum theoretical bandwidth)
                link_output = subprocess.check_output(
                    ["nvidia-smi", f"--id={index}", "--query-gpu=pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"],
                    universal_newlines=True
                ).strip()
                gen, width = link_output.split(',')
                
                # Calculate theoretical max bandwidth
                width_str = width.strip()
                try:
                    gen_num = float(gen.strip())
 
                    if width_str.startswith('x') and len(width_str) > 1:
                        width_num = int(width_str[1:])
                    else:
                        width_num = int(width_str)                 
                    bandwidth = 0
                    if gen_num == 3.0:
                        bandwidth = width_num * 0.985  # PCIe 3.0: ~0.985 GB/s per lane
                    elif gen_num == 4.0:
                        bandwidth = width_num * 1.97   # PCIe 4.0: ~1.97 GB/s per lane
                    elif gen_num == 5.0:
                        bandwidth = width_num * 3.94   # PCIe 5.0: ~3.94 GB/s per lane
                except (ValueError, IndexError) as e:
                    print(f"Error parsing PCIe info: {e}")
                # PCIe bandwidth in GB/s (approximate):
                
                
                # Get memory controller utilization as proxy for PCIe activity
                try:
                    # Run dmon with utilization stats (-s u)
                    dmon_process = subprocess.Popen(
                        ["nvidia-smi", "dmon", "-s", "u", "-c", "3", "-d", "1"],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    stdout, stderr = dmon_process.communicate(timeout=4)
                    
                    # Parse output to get memory controller utilization (column 3)
                    lines = stdout.strip().split('\n')
                    mem_utils = []
                    
                    for i, line in enumerate(lines):
                        # Skip header lines (first two lines)
                        if i < 2 or line.startswith('#') or not line.strip():
                            continue
                        
                        parts = line.split()
                        # Check if this is our GPU and we have enough columns
                        if len(parts) >= 6 and parts[0].strip() == str(index):
                            # Memory controller utilization is the 3rd column (index 2)
                            try:
                                mem_util = parts[2].strip()
                                if mem_util:  # Make sure it's not empty
                                    mem_utils.append(int(mem_util))
                            except (ValueError, IndexError):
                                # Skip invalid values
                                
                                pass
                    
                    # Calculate average memory controller utilization
                    if mem_utils:
                        avg_mem_util = sum(mem_utils) / len(mem_utils)
                        
                        results[index] = {
                            'link_gen': gen.strip(),
                            'link_width': width.strip(),
                            'max_bandwidth_GBps': bandwidth,
                            'mem_controller_util': avg_mem_util,
                            'estimated_usage_GBps': (avg_mem_util / 100.0) * bandwidth if bandwidth > 0 else 0
                        }
                    else:
                        # No valid data found
                        results[index] = {
                            'link_gen': gen.strip(),
                            'link_width': width.strip(),
                            'max_bandwidth_GBps': bandwidth,
                            'mem_controller_util': 0,
                            'estimated_usage_GBps': 0,
                            'note': 'No valid memory controller data found'
                        }
                        
                except subprocess.TimeoutExpired:
                    dmon_process.kill()
                    results[index] = {
                        'link_gen': gen.strip(),
                        'link_width': width.strip(),
                        'max_bandwidth_GBps': bandwidth,
                        'error': 'Timeout collecting memory utilization data'
                    }
                    
                except Exception as e:
                    results[index] = {
                        'link_gen': gen.strip(),
                        'link_width': width.strip(),
                        'max_bandwidth_GBps': bandwidth,
                        'error': str(e)
                    }
                    
        except Exception as e:
            results['error'] = str(e)
        
        return results
        
    def _get_inference_latency(self, batch_size=1, num_runs=10):
        """Measure inference latency if model is available"""
        if not hasattr(self, 'model') or self.model is None:
            return {'status': 'no model registered'}
            
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                
                # Create dummy input based on model's input shape
                input_shape = self.model.input_shape
                if isinstance(input_shape, list):
                    # Handle multiple inputs
                    dummy_input = [tf.random.normal([batch_size] + list(shape[1:])) for shape in input_shape]
                else:
                    # Single input
                    dummy_input = tf.random.normal([batch_size] + list(input_shape[1:]))
                
                # Warm-up run
                _ = self.model(dummy_input)
                
                # Measure latency
                start_time = time.time()
                for _ in range(num_runs):
                    _ = self.model(dummy_input)
                end_time = time.time()
                
                avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
                
                return {
                    'batch_size': batch_size,
                    'num_runs': num_runs,
                    'avg_latency_ms': avg_latency_ms,
                    'throughput_samples_per_sec': (batch_size * num_runs) / (end_time - start_time)
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {'status': 'framework not supported'}
    
    def _get_training_step_latency(self, model=None, batch_size=32, num_runs=5):
        """Measure training step latency with a dummy batch"""
        if model is None:
            if hasattr(self, 'model') and self.model is not None:
                model = self.model
            else:
                return {'status': 'no model available'}
        
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                import time
                
                # Create dummy input based on model's input shape
                input_shape = model.input_shape
                
                # Fix: Safely get output shape dimension for target
                try:
                    output_dim = model.output.shape[-1]
                except (AttributeError, ValueError):
                    # Try to get from the last layer
                    try:
                        output_dim = model.layers[-1].output.shape[-1]
                    except (AttributeError, ValueError, IndexError):
                        return {'error': 'Could not determine output shape for model'}
                
                if isinstance(input_shape, list):
                    # Handle multiple inputs
                    dummy_input = [tf.random.normal([batch_size] + list(shape[1:])) for shape in input_shape]
                    dummy_target = tf.random.normal([batch_size, output_dim])
                else:
                    # Single input
                    dummy_input = tf.random.normal([batch_size] + list(input_shape[1:]))
                    dummy_target = tf.random.normal([batch_size, output_dim])
                
                @tf.function
                def train_step(x, y):
                    with tf.GradientTape() as tape:
                        y_pred = model(x, training=True)
                        loss = model.loss(y, y_pred)
                        
                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    return loss
                
                # Warm-up run
                _ = train_step(dummy_input, dummy_target)
                
                # Measure latency
                latencies = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = train_step(dummy_input, dummy_target)
                    latencies.append((time.time() - start_time) * 1000)  # ms
                    
                return {
                    'batch_size': batch_size,
                    'num_runs': num_runs,
                    'avg_latency_ms': sum(latencies) / len(latencies),
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'throughput_samples_per_sec': (batch_size * num_runs) / (sum(latencies) / 1000)
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {'status': 'framework not supported'}

    def _get_data_pipeline_latency(self, dataset=None, num_batches=10):
        """Measure data pipeline latency"""
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                import time
                
                if dataset is None:
                    return {'status': 'no dataset provided'}
                    
                # Create a simple iterator
                iterator = iter(dataset)
                
                # Warm-up
                _ = next(iterator)
                
                # Measure latency
                latencies = []
                for _ in range(num_batches):
                    start_time = time.time()
                    _ = next(iterator, None)
                    end_time = time.time()
                    
                    # If we've exhausted the dataset, break
                    if _ is None:
                        break
                        
                    latencies.append((end_time - start_time) * 1000)  # ms
                    
                if not latencies:
                    return {'status': 'dataset exhausted during measurement'}
                    
                return {
                    'num_batches': len(latencies),
                    'avg_latency_ms': sum(latencies) / len(latencies),
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies)
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {'status': 'framework not supported'}
    
    def _get_framework_metrics(self):
        """Get framework-specific metrics"""
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                return {
                    'version': tf.__version__,
                    'built_with_cuda': tf.test.is_built_with_cuda(),
                    'gpu_available': tf.test.is_gpu_available(),
                    'eager_execution': tf.executing_eagerly()
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {}
    
    def save_metrics_to_file(self, filename='gpu_metrics.json'):
        """Save collected metrics to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        print(f"Metrics saved to {filename}")
    
    def get_latest_metrics(self):
        """Get the most recently collected metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self):
        """Get a summary of collected metrics including idle time"""
        if not self.metrics_history:
            return "No metrics collected"
            
        num_records = len(self.metrics_history)
        
        # Calculate averages for key metrics
        avg_gpu_util = {}
        avg_gpu_mem = {}
        avg_cpu_util = 0
        avg_inference_latency = 0
        
        # Add idle time tracking
        total_idle_time_seconds = 0
        max_idle_time_seconds = 0
        idle_count = 0
        gpu_idle_periods = 0
        
        inference_count = 0
        
        for record in self.metrics_history:
            # GPU utilization
            for gpu_id, util in record.get('gpu', {}).get('utilization', {}).items():
                if isinstance(gpu_id, int) and isinstance(util, (int, float)):
                    avg_gpu_util[gpu_id] = avg_gpu_util.get(gpu_id, 0) + util / num_records
            
            # GPU memory
            for gpu_id, mem in record.get('gpu', {}).get('memory', {}).items():
                if isinstance(gpu_id, int) and isinstance(mem, dict) and 'percentage' in mem:
                    avg_gpu_mem[gpu_id] = avg_gpu_mem.get(gpu_id, 0) + mem['percentage'] / num_records
            
            # CPU utilization
            cpu_data = record.get('cpu', {}).get('utilization', {})
            if isinstance(cpu_data, dict) and 'overall' in cpu_data:
                avg_cpu_util += cpu_data['overall'] / num_records
            
            # Inference latency
            latency_data = record.get('latency', {}).get('inference', {})
            if isinstance(latency_data, dict) and 'avg_latency_ms' in latency_data:
                avg_inference_latency += latency_data['avg_latency_ms']
                inference_count += 1
                
            # Idle time (measured in seconds)
            gpu_data = record.get('gpu', {})
            if 'idle_time' in gpu_data:
                idle_time = gpu_data['idle_time']
                if isinstance(idle_time, (int, float)) and idle_time > 0:
                    total_idle_time_seconds += idle_time
                    max_idle_time_seconds = max(max_idle_time_seconds, idle_time)
                    idle_count += 1
                    
            # Track when GPU transitions to idle state
            if gpu_data.get('is_currently_idle', False):
                gpu_idle_periods += 1
        
        if inference_count > 0:
            avg_inference_latency /= inference_count
        
        # Calculate average idle time and percentage of time spent idle
        avg_idle_time_seconds = total_idle_time_seconds / idle_count if idle_count > 0 else 0
        
        # Add idle metrics to the summary
        return {
            'num_records': num_records,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_gpu_memory': avg_gpu_mem,
            'avg_cpu_utilization': avg_cpu_util,
            'avg_inference_latency_ms': avg_inference_latency if inference_count > 0 else 'Not measured',
            'model_architecture': self.model_architecture,
            'idle_time': {
                'total_seconds': total_idle_time_seconds,
                'average_seconds': avg_idle_time_seconds,
                'max_seconds': max_idle_time_seconds,
                'idle_records_count': idle_count,
                'idle_periods_detected': gpu_idle_periods,
                'time_unit': 'seconds'
            }
        }
    
    def save_organized_metrics(self, model_id=None, experiment_id=None, 
                          base_dir='hardware_metrics', clear_after_save=False,
                          save_raw=True, save_compressed=False):
        """
        Save metrics in an organized directory structure for later analysis
        
        Args:
            model_id: Unique identifier for the model
            experiment_id: Experiment identifier
            base_dir: Base directory for metrics storage
            clear_after_save: Whether to clear metrics history after saving
            save_raw: Whether to save complete raw data (can be large)
            save_compressed: Whether to save a compressed version
        """
        import os
        from datetime import datetime
        
        # Check if we have metrics to save
        if not self.metrics_history:
            print("No metrics to save")
            return None
        
        # Get I/O stats before saving (for monitoring)
        process = psutil.Process(os.getpid())
        io_before = process.io_counters() if hasattr(process, 'io_counters') else None
        
        # Create timestamp for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        if model_id and experiment_id:
            # Organize by experiment and model
            save_dir = os.path.join(base_dir, experiment_id, model_id, timestamp)
        elif model_id:
            # Organize by model only
            save_dir = os.path.join(base_dir, model_id, timestamp)
        else:
            # Just use timestamp
            save_dir = os.path.join(base_dir, timestamp)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get summary metrics
        summary = self.get_metrics_summary()
        
        # Separate metrics by type
        gpu_metrics = []
        cpu_metrics = []
        latency_metrics = []
        
        for record in self.metrics_history:
            timestamp = record.get('timestamp', '')
            
            if 'gpu' in record:
                gpu_metrics.append({
                    'timestamp': timestamp,
                    'metrics': record['gpu']
                })
            
            if 'cpu' in record:
                cpu_metrics.append({
                    'timestamp': timestamp,
                    'metrics': record['cpu']
                })
                
            if 'latency' in record:
                latency_metrics.append({
                    'timestamp': timestamp,
                    'metrics': record['latency']
                })
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'experiment_id': experiment_id,
            'collection_time': datetime.now().isoformat(),
            'total_records': len(self.metrics_history),
            'framework': self.framework,
            'devices': self.gpu_devices,
        }
        
        # Save all files - main summary always saved
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        with open(os.path.join(save_dir, 'gpu_metrics.json'), 'w') as f:
            json.dump(gpu_metrics, f, indent=2, default=str)
            
        with open(os.path.join(save_dir, 'cpu_metrics.json'), 'w') as f:
            json.dump(cpu_metrics, f, indent=2, default=str)
            
        with open(os.path.join(save_dir, 'latency_metrics.json'), 'w') as f:
            json.dump(latency_metrics, f, indent=2, default=str)
        
        # Optionally save raw data (can be large)
        if save_raw:
            with open(os.path.join(save_dir, 'raw_metrics.json'), 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        
        # For larger datasets, optionally save a compressed version
        if save_compressed and len(self.metrics_history) > 100:
            import gzip
            with gzip.open(os.path.join(save_dir, 'raw_metrics.json.gz'), 'wt') as f:
                json.dump(self.metrics_history, f)
        
        # Save model architecture if available
        if self.model_architecture:
            with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
                json.dump(self.model_architecture, f, indent=2, default=str)
        
        # Create a quick readme
        with open(os.path.join(save_dir, 'README.txt'), 'w') as f:
            f.write(f"Hardware Metrics Report\n")
            f.write(f"=======================\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {model_id or 'Not specified'}\n")
            f.write(f"Experiment ID: {experiment_id or 'Not specified'}\n")
            f.write(f"Framework: {self.framework}\n")
            f.write(f"Number of records: {len(self.metrics_history)}\n\n")
            f.write(f"Files in this directory:\n")
            f.write(f"  - metadata.json: General information about this collection\n")
            f.write(f"  - summary.json: Summary statistics of all metrics\n")
            f.write(f"  - gpu_metrics.json: GPU-specific metrics\n")
            f.write(f"  - cpu_metrics.json: CPU-specific metrics\n")
            f.write(f"  - latency_metrics.json: Latency measurements\n")
            if save_raw:
                f.write(f"  - raw_metrics.json: Complete raw metrics data\n")
            if self.model_architecture:
                f.write(f"  - model_architecture.json: Neural network model architecture\n")
        
        # Report I/O stats if available
        if io_before:
            io_after = process.io_counters()
            write_bytes = io_after.write_bytes - io_before.write_bytes
            print(f"Metrics saved to {save_dir} (wrote {write_bytes/1024/1024:.2f} MB)")
        else:
            print(f"Metrics saved to {save_dir}")
        
        # Clear history if requested to free memory
        if clear_after_save:
            original_size = len(self.metrics_history)
            self.metrics_history = []
            print(f"Cleared metrics history ({original_size} records)")
        
        return save_dir

    def generate_hardware_report(self, save_path=None, model_id=None, experiment_id=None):
        """
        Generate a comprehensive hardware usage report
        
        Args:
            save_path: Path to save the report (if None, just returns the report data)
            model_id: Model identifier
            experiment_id: Experiment identifier
        
        Returns:
            dict: Report data
        """
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Get first and last timestamp from records
        first_timestamp = self.metrics_history[0].get('timestamp', 'Unknown')
        last_timestamp = self.metrics_history[-1].get('timestamp', 'Unknown')
        
        # Calculate time ranges
        try:
            # Parse timestamps if they're strings
            if isinstance(first_timestamp, str):
                first_time = datetime.fromisoformat(first_timestamp)
                last_time = datetime.fromisoformat(last_timestamp)
                duration_seconds = (last_time - first_time).total_seconds()
            else:
                # Assume they're already datetime objects
                duration_seconds = (last_timestamp - first_timestamp).total_seconds()
        except:
            duration_seconds = None
        
        # GPU utilization over time
        gpu_util_over_time = {}
        gpu_mem_over_time = {}
        
        for record in self.metrics_history:
            ts = record.get('timestamp', '')
            gpu_data = record.get('gpu', {})
            
            # GPU utilization
            for gpu_id, util in gpu_data.get('utilization', {}).items():
                if isinstance(gpu_id, int) and isinstance(util, (int, float)):
                    if gpu_id not in gpu_util_over_time:
                        gpu_util_over_time[gpu_id] = []
                    gpu_util_over_time[gpu_id].append((ts, util))
            
            # GPU memory
            for gpu_id, mem in gpu_data.get('memory', {}).items():
                if isinstance(gpu_id, int) and isinstance(mem, dict) and 'percentage' in mem:
                    if gpu_id not in gpu_mem_over_time:
                        gpu_mem_over_time[gpu_id] = []
                    gpu_mem_over_time[gpu_id].append((ts, mem['percentage']))
        
        # Calculate key statistics
        summary = self.get_metrics_summary()
        
        # Detect potential issues
        issues = []
        
        # Check for low GPU utilization
        for gpu_id, avg_util in summary.get('avg_gpu_utilization', {}).items():
            if avg_util < 30:  # Less than 30% utilization might indicate bottlenecks
                issues.append(f"Low GPU utilization for GPU {gpu_id}: {avg_util:.1f}%")
        
        # Check for high CPU usage (might indicate data pipeline bottleneck)
        if summary.get('avg_cpu_utilization', 0) > 90:
            issues.append(f"High CPU utilization: {summary.get('avg_cpu_utilization'):.1f}%")
        
        # Prepare comprehensive report
        report = {
            "model_id": model_id,
            "experiment_id": experiment_id,
            "collection_period": {
                "start": first_timestamp,
                "end": last_timestamp,
                "duration_seconds": duration_seconds
            },
            "hardware": {
                "gpus": self.gpu_devices,
            },
            "summary_metrics": summary,
            "gpu_utilization_trends": gpu_util_over_time,
            "gpu_memory_trends": gpu_mem_over_time,
            "potential_issues": issues,
            "model_architecture_summary": {
                "total_params": self.model_architecture.get('total_params', 'N/A') if self.model_architecture else 'N/A',
                "layer_count": len(self.model_architecture.get('layers', [])) if self.model_architecture else 0
            }
        }
        
        # Save report if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Hardware report saved to {save_path}")
        
        return report

    def configure_auto_save(self, enabled=True, interval_seconds=60, 
                            save_path='hardware_metrics', max_history=1000):
        """
        Configure automatic periodic saving of metrics
        
        Args:
            enabled: Whether to enable auto-saving
            interval_seconds: How often to save metrics (in seconds)
            save_path: Directory to save metrics to
            max_history: Maximum number of metrics to keep in memory
        """
        self.auto_save_enabled = enabled
        self.save_interval = interval_seconds
        self.auto_save_path = save_path
        self.max_history_size = max_history
        self.last_save_time = time.time()
        
        print(f"Auto-save {'enabled' if enabled else 'disabled'}, interval: {interval_seconds}s, "
              f"path: {save_path}, max history: {max_history} records")
        
        # Create save directory if it doesn't exist
        if enabled and save_path:
            os.makedirs(save_path, exist_ok=True)
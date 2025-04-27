import os
import json
import time
import platform
import psutil
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional

@dataclass
class ModelMetrics:
    model_id: str
    experiment_id: str
    model_type: str
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    layer_count: int
    layer_details: List[Dict[str, Any]]
    
@dataclass
class TrainingMetrics:
    build_time_ms: int
    train_time_ms: int
    train_time_per_epoch_ms: List[int]
    epochs_completed: int
    batch_size: int
    optimizer: str
    learning_rate: float
    final_loss: float
    final_accuracy: Optional[float] = None

@dataclass
class HardwareInfo:
    hostname: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_count: int
    gpu_models: List[str]
    gpu_memory_gb: List[float]
    tensorflow_version: str
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

class HardwarePerformanceLogger:
    def __init__(self, base_log_dir="hardware_performance_logs"):
        """Initialize the hardware performance logger"""
        self.base_log_dir = base_log_dir
        os.makedirs(base_log_dir, exist_ok=True)
        self.hardware_info = self._collect_hardware_info()
        self.log_file = None
        self.start_time = None
        self.epoch_start_times = []
        self.epoch_durations = []
        
    def _collect_hardware_info(self) -> HardwareInfo:
        """Collect hardware information about the system"""
        # CPU information
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False) or 0
        cpu_threads = psutil.cpu_count(logical=True) or 0
        
        # RAM information
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        
        # GPU information
        gpu_models = []
        gpu_memory_gb = []
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_count = len(gpus)
            
            # Try to get detailed GPU info
            if gpu_count > 0:
                try:
                    import subprocess
                    # Try nvidia-smi for NVIDIA GPUs
                    nvidia_output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        universal_newlines=True
                    )
                    for line in nvidia_output.strip().split("\n"):
                        parts = line.split(", ")
                        if len(parts) >= 2:
                            gpu_models.append(parts[0])
                            # Convert MiB to GB
                            memory_str = parts[1].split(" ")[0]
                            gpu_memory_gb.append(float(memory_str) / 1024)
                except (subprocess.SubprocessError, FileNotFoundError):
                    # If nvidia-smi fails, use basic info
                    gpu_models = [f"GPU {i}" for i in range(gpu_count)]
                    gpu_memory_gb = [0.0] * gpu_count
            else:
                gpu_count = 0
        except:
            gpu_count = 0
        
        # TensorFlow and CUDA versions
        tf_version = tf.__version__
        cuda_version = None
        cudnn_version = None
        
        # Try to get CUDA version
        try:
            cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
            cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
        except:
            # If the above method fails, try another approach
            try:
                from tensorflow.python.platform import build_info
                cuda_version = build_info.build_info["cuda_version"]
                cudnn_version = build_info.build_info["cudnn_version"]
            except:
                pass
                
        return HardwareInfo(
            hostname=platform.node(),
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            gpu_memory_gb=gpu_memory_gb,
            tensorflow_version=tf_version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version
        )
    
    def get_model_metrics(self, model: tf.keras.Model, model_id: str, 
                        experiment_id: str, model_type: str) -> ModelMetrics:
        """Extract and record model architecture metrics"""
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_variables])
        total_params = trainable_params + non_trainable_params
        
        layer_details = []
        for i, layer in enumerate(model.layers):
            config = layer.get_config()
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            else:
                output_shape = "Unknown"
                
            layer_info = {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "output_shape": output_shape,
                "parameter_count": layer.count_params()
            }
            layer_details.append(layer_info)
        
        return ModelMetrics(
            model_id=model_id,
            experiment_id=experiment_id,
            model_type=model_type,
            total_parameters=int(total_params),
            trainable_parameters=int(trainable_params),
            non_trainable_parameters=int(non_trainable_params),
            layer_count=len(model.layers),
            layer_details=layer_details
        )
    
    def start_timing(self):
        """Start timing for training"""
        self.start_time = time.time()
        self.epoch_start_times = []
        self.epoch_durations = []
        
    def record_epoch_start(self):
        """Record the start time of an epoch"""
        self.epoch_start_times.append(time.time())
        
    def record_epoch_end(self):
        """Record the end time of an epoch"""
        if len(self.epoch_start_times) > len(self.epoch_durations):
            epoch_duration = (time.time() - self.epoch_start_times[-1]) * 1000  # ms
            self.epoch_durations.append(epoch_duration)
    
    def get_training_metrics(self, history, build_time_ms: int, 
                           batch_size: int, optimizer: str, 
                           learning_rate: float) -> TrainingMetrics:
        """Create training metrics from completed training"""
        train_time_ms = int((time.time() - self.start_time) * 1000)
        
        # Get the final epoch metrics
        epochs_completed = len(history.history['loss'])
        final_loss = history.history['loss'][-1]
        
        # Check if accuracy is available
        final_accuracy = None
        for metric in ['accuracy', 'val_accuracy', 'acc', 'val_acc']:
            if metric in history.history:
                final_accuracy = history.history[metric][-1]
                break
        
        return TrainingMetrics(
            build_time_ms=build_time_ms,
            train_time_ms=train_time_ms,
            train_time_per_epoch_ms=self.epoch_durations,
            epochs_completed=epochs_completed,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            final_loss=final_loss,
            final_accuracy=final_accuracy
        )
    
    def save_log(self, model_metrics: ModelMetrics, training_metrics: TrainingMetrics):
        """Save all collected metrics to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{model_metrics.model_id}_{timestamp}.json"
        
        log_dir = os.path.join(self.base_log_dir, model_metrics.experiment_id)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        
        log_data = {
            "timestamp": timestamp,
            "hardware_info": asdict(self.hardware_info),
            "model_metrics": asdict(model_metrics),
            "training_metrics": asdict(training_metrics)
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        return log_path
    
    @staticmethod
    def compare_logs(log_files):
        """
        Compare multiple hardware performance logs
        
        Args:
            log_files: List of log file paths to compare
        """
        comparison_data = []
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                
            # Extract key metrics for comparison
            hardware = log_data["hardware_info"]
            model = log_data["model_metrics"]
            training = log_data["training_metrics"]
            
            comparison_entry = {
                "model_id": model["model_id"],
                "hardware": f"{hardware['cpu_model']} | {hardware['gpu_count']} GPU(s): {', '.join(hardware['gpu_models'])}",
                "parameters": model["total_parameters"],
                "build_time_ms": training["build_time_ms"],
                "train_time_ms": training["train_time_ms"],
                "epochs_completed": training["epochs_completed"],
                "time_per_epoch_ms": training["train_time_ms"] / max(1, training["epochs_completed"]),
                "final_loss": training["final_loss"],
                "final_accuracy": training["final_accuracy"]
            }
            comparison_data.append(comparison_entry)
        
        # Sort by training time
        comparison_data.sort(key=lambda x: x["train_time_ms"])
        
        return comparison_data
MLOptimizer: Automated Machine Learning Model Generation & Hardware Optimization

<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg">
<img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue.svg">
MLOptimizer is a comprehensive framework for automatically generating optimized machine learning models while capturing detailed hardware performance metrics. It provides a distributed system for model training, hyperparameter optimization, and hardware resource monitoring.

Features
Automated Model Generation: Automatically design and train models for:

Image classification
Regression (simple and multiple output)
Time series prediction
Hardware Performance Optimization:

Real-time GPU metrics tracking (memory, utilization, temperature)
CPU resource monitoring
Idle time detection and reporting
Training latency measurements
Distributed Architecture:

Master/slave processing model
RabbitMQ-based communication
Fault tolerance with automatic retry mechanisms
Comprehensive Reporting:

Detailed performance metrics in JSON format
Hardware utilization summaries  
Model architecture snapshots
System Requirements
Python 3.10+
TensorFlow 2.x
CUDA-compatible GPU
RabbitMQ server
Installation

# Clone the repository
git clone https://github.com/yourusername/mloptimizer.git
cd mloptimizer

# Create and activate conda environment
conda create -n mlopt python=3.10
conda activate mlopt

# Install dependencies
pip install -r requirements.txt

# Start RabbitMQ (if not already running)
sudo systemctl start rabbitmq-serverv


# Start the system (master and slave processes)
cd cloudDeployment
./den.sh


Architecture
MLOptimizer consists of several key components:

Master Process: Coordinates training jobs and optimization strategy
Slave Process: Executes model training and collects performance metrics
GPU Metrics Collector: Captures detailed hardware performance data
Socket Communication: Provides messaging between components
Model Generator: Creates model architectures based on problem specifications
Usage Examples
Training an Image Classification Model


from app.common.dataset import Dataset
from app.common.model import Model
from app.common.search_space import SearchSpaceType

# Configure dataset
dataset = Dataset(dataset_path="path/to/images", type=SearchSpaceType.IMAGE)

# Create and train model
model = Model(dataset=dataset)
accuracy = model.build_and_train()
print(f"Model accuracy: {accuracy}")

Analyzing Hardware Performance


# Access metrics after training
metrics_dir = model.get_metrics_directory()
print(f"Performance metrics saved to: {metrics_dir}")

# Example metrics include:
# - GPU utilization
# - Memory usage
# - Training latency
# - Idle time periods


Documentation
For more detailed documentation, see the following:

Hardware Metrics Guide
Model Architecture Options
Optimization Strategies
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.# mloptimizer

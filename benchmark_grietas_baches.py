"""
Benchmark script to find the best architecture for GRIETAS and BACHES dataset
using the MLOptimizer AutoML system.

This script:
1. Loads the GRIETAS (cracks) and BACHES (potholes) dataset
2. Configures the system for image classification
3. Runs the AutoML optimization to find the best architecture
4. Logs hardware performance metrics
"""

import os
import sys
from load_grietas_baches_dataset import load_grietas_baches_dataset, save_grietas_baches_dataset
from system_parameters import SystemParameters as SP
from app.init_nodes import InitNodes
from app.common.dataset import ImageClassificationBenchmarkDataset, Dataset
from app.common.search_space import ImageModelArchitectureFactory
from app.master_node.optimization_job import OptimizationJob

def setup_grietas_baches_parameters():
    """
    Configure system parameters for GRIETAS and BACHES dataset
    """
    print("=" * 60)
    print("Configuring MLOptimizer for GRIETAS and BACHES Dataset")
    print("=" * 60)
    
    # Dataset parameters
    SP.DATASET_NAME = 'grietas_baches'
    SP.DATASET_TYPE = 1  # Image classification
    SP.DATASET_SHAPE = (224, 224, 3)  # RGB images at 224x224
    SP.DATASET_CLASSES = 2  # Binary classification: BACHES vs GRIETAS
    SP.DATASET_BATCH_SIZE = 32
    SP.DATASET_VALIDATION_SPLIT = 0.2
    
    # AutoML parameters - adjust based on dataset size
    SP.TRIALS = 30  # Number of different architectures to try
    SP.EXPLORATION_SIZE = 15  # Initial quick evaluations
    SP.EXPLORATION_EPOCHS = 15  # Quick training for exploration
    SP.EXPLORATION_EARLY_STOPPING_PATIENCE = 3
    SP.HALL_OF_FAME_SIZE = 10  # Top models to train longer
    SP.HALL_OF_FAME_EPOCHS = 100  # Full training for best models
    SP.HOF_EARLY_STOPPING_PATIENCE = 10
    
    # Training parameters
    SP.TRAIN_GPU = True
    
    # Model parameters
    SP.DTYPE = 'float32'
    SP.OPTIMIZER = 'adam'
    SP.LAYERS_ACTIVATION_FUNCTION = 'relu'
    SP.OUTPUT_ACTIVATION_FUNCTION = 'softmax'
    SP.KERNEL_INITIALIZER = 'he_uniform'
    SP.LOSS_FUNCTION = 'sparse_categorical_crossentropy'
    SP.METRICS = ['accuracy']
    SP.PADDING = 'same'
    SP.WEIGHT_DECAY = 1e-4
    
    print("\nDataset Configuration:")
    print(f"  Name: {SP.DATASET_NAME}")
    print(f"  Type: Image Classification (Binary)")
    print(f"  Shape: {SP.DATASET_SHAPE}")
    print(f"  Classes: {SP.DATASET_CLASSES} (BACHES, GRIETAS)")
    print(f"  Batch Size: {SP.DATASET_BATCH_SIZE}")
    print(f"  Validation Split: {SP.DATASET_VALIDATION_SPLIT}")
    
    print("\nAutoML Configuration:")
    print(f"  Total Trials: {SP.TRIALS}")
    print(f"  Exploration Phase: {SP.EXPLORATION_SIZE} models × {SP.EXPLORATION_EPOCHS} epochs")
    print(f"  Hall of Fame Phase: {SP.HALL_OF_FAME_SIZE} models × {SP.HALL_OF_FAME_EPOCHS} epochs")
    print(f"  GPU Training: {SP.TRAIN_GPU}")
    print("=" * 60)


class GrietasBachesDataset(Dataset):
    """
    Custom Dataset class for GRIETAS and BACHES that integrates with MLOptimizer
    """
    def __init__(self, dataset_path='./dataset_grietas_baches', 
                 target_size=(224, 224), batch_size=32, 
                 validation_split=0.2):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.name = 'grietas_baches'
        self.num_classes = 2
        self.input_shape = (target_size[0], target_size[1], 3)
        
        # Load the dataset
        print(f"\nLoading GRIETAS and BACHES dataset from {dataset_path}...")
        (self.train_data, self.train_labels), \
        (self.val_data, self.val_labels), \
        (self.test_data, self.test_labels) = load_grietas_baches_dataset(
            dataset_path=dataset_path,
            target_size=target_size,
            validation_split=validation_split,
            test_split=0.1
        )
        
        # Convert from CHW to HWC format (TensorFlow format)
        import numpy as np
        self.train_data = np.transpose(self.train_data, (0, 2, 3, 1))
        self.val_data = np.transpose(self.val_data, (0, 2, 3, 1))
        self.test_data = np.transpose(self.test_data, (0, 2, 3, 1))
        
        print(f"\nDataset loaded successfully!")
        print(f"  Training samples: {len(self.train_data)}")
        print(f"  Validation samples: {len(self.val_data)}")
        print(f"  Test samples: {len(self.test_data)}")
        print(f"  Input shape: {self.input_shape}")
    
    def get_train_data(self):
        """Return training data"""
        return self.train_data, self.train_labels
    
    def get_validation_data(self):
        """Return validation data"""
        return self.val_data, self.val_labels
    
    def get_test_data(self):
        """Return test data"""
        return self.test_data, self.test_labels
    
    def get_shape(self):
        """Return input shape"""
        return self.input_shape
    
    def get_classes(self):
        """Return number of classes"""
        return self.num_classes


def run_architecture_search(dataset_path='./dataset_grietas_baches'):
    """
    Run the AutoML architecture search for GRIETAS and BACHES dataset
    """
    # Setup system parameters
    setup_grietas_baches_parameters()
    
    # Create dataset instance
    print("\n" + "=" * 60)
    print("Initializing Dataset")
    print("=" * 60)
    dataset = GrietasBachesDataset(
        dataset_path=dataset_path,
        target_size=(224, 224),
        batch_size=SP.DATASET_BATCH_SIZE,
        validation_split=SP.DATASET_VALIDATION_SPLIT
    )
    
    # Create model architecture factory
    print("\n" + "=" * 60)
    print("Initializing Architecture Search")
    print("=" * 60)
    model_architecture_factory = ImageModelArchitectureFactory()
    
    # Create optimization job
    print("\nStarting AutoML optimization...")
    print("This will explore different CNN architectures to find the best one.")
    print("The process includes:")
    print("  1. Exploration Phase: Quick evaluation of diverse architectures")
    print("  2. Hall of Fame Phase: In-depth training of top performers")
    print("  3. Hardware Performance Logging: GPU/CPU metrics collection")
    print("\nResults will be saved in:")
    print("  - metrics_data/ (detailed performance metrics)")
    print("  - hardware_performance_logs/ (GPU/CPU usage)")
    print("=" * 60)
    
    optimization_job = OptimizationJob(dataset, model_architecture_factory)
    optimization_job.start_optimization(trials=SP.TRIALS)
    
    print("\n" + "=" * 60)
    print("Architecture Search Complete!")
    print("=" * 60)
    print("\nCheck the following directories for results:")
    print("  - metrics_data/: JSON files with model performance metrics")
    print("  - hardware_performance_logs/: Hardware utilization logs")
    print("\nThe best architecture will be identified in the logs.")


def run_simple_benchmark(dataset_path='./dataset_grietas_baches'):
    """
    Run a simple benchmark without the full AutoML pipeline
    Useful for quick testing or when RabbitMQ is not available
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from app.common.hardware_performance_logger import HardwarePerformanceLogger
    from app.common.hardware_performance_callback import HardwarePerformanceCallback
    import time
    
    setup_grietas_baches_parameters()
    
    # Load dataset
    print("\nLoading dataset for simple benchmark...")
    dataset = GrietasBachesDataset(dataset_path=dataset_path)
    train_data, train_labels = dataset.get_train_data()
    val_data, val_labels = dataset.get_validation_data()
    test_data, test_labels = dataset.get_test_data()
    
    # Create a sample CNN model
    print("\nCreating sample CNN architecture...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Initialize performance logger
    logger = HardwarePerformanceLogger(base_log_dir="hardware_performance_logs")
    logger.start_timing()
    
    # Train model
    print("\nTraining model...")
    performance_callback = HardwarePerformanceCallback(logger)
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_data, train_labels,
        batch_size=32,
        epochs=50,
        validation_data=(val_data, val_labels),
        callbacks=[performance_callback, early_stopping]
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save performance metrics
    model_id = f"grietas_baches_cnn_{int(time.time())}"
    model_metrics = logger.get_model_metrics(
        model, 
        model_id, 
        "grietas_baches_benchmark",
        "CNN"
    )
    
    training_metrics = logger.get_training_metrics(
        history,
        0,  # build_time_ms
        32,  # batch_size
        "Adam",
        0.001
    )
    
    log_path = logger.save_log(model_metrics, training_metrics)
    print(f"\nPerformance log saved to: {log_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find the best architecture for GRIETAS and BACHES dataset'
    )
    parser.add_argument(
        '--dataset-path', 
        default='./dataset_grietas_baches',
        help='Path to dataset folder containing GRIETAS and BACHES subfolders'
    )
    parser.add_argument(
        '--mode',
        choices=['automl', 'simple'],
        default='simple',
        help='Mode: automl (full architecture search) or simple (single model benchmark)'
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"\nERROR: Dataset path not found: {args.dataset_path}")
        print("Please ensure the dataset folder exists with GRIETAS and BACHES subfolders.")
        sys.exit(1)
    
    # Configure GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\nGPU Configuration: {len(gpus)} GPU(s) available")
    except Exception as e:
        print(f"\nGPU Configuration: Running on CPU (Error: {e})")
    
    # Run selected mode
    if args.mode == 'automl':
        print("\n" + "=" * 60)
        print("AUTOML MODE: Full Architecture Search")
        print("=" * 60)
        print("\nNote: This mode requires RabbitMQ to be running.")
        print("If you haven't started RabbitMQ, please run:")
        print("  sudo systemctl start rabbitmq-server")
        print("\nOr use --mode simple for a quick benchmark without RabbitMQ.")
        print("=" * 60 + "\n")
        
        try:
            run_architecture_search(args.dataset_path)
        except Exception as e:
            print(f"\nERROR: {e}")
            print("\nTip: If RabbitMQ connection failed, try using --mode simple instead.")
    else:
        print("\n" + "=" * 60)
        print("SIMPLE MODE: Single Model Benchmark")
        print("=" * 60 + "\n")
        run_simple_benchmark(args.dataset_path)

import os
import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from app.common.hardware_performance_logger import HardwarePerformanceLogger
from app.common.hardware_performance_callback import HardwarePerformanceCallback

def create_cnn_model(input_shape=(28, 28, 1)):
    """Create CNN model matching the architecture in the logs"""
    model = Sequential()
    
    # Initial block - removed explicit name from first Conv2D layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Second block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Continue adding layers to match parameter count (~411,210)
    for i in range(5):  # Add more blocks to reach the parameter count
        # Use explicit names with loop index to ensure uniqueness
        model.add(Conv2D(128, (3, 3), padding='same', name=f'conv2d_block{i+3}_1'))
        model.add(BatchNormalization(name=f'bn_block{i+3}_1'))
        model.add(Conv2D(128, (3, 3), padding='same', name=f'conv2d_block{i+3}_2'))
        model.add(BatchNormalization(name=f'bn_block{i+3}_2'))
        model.add(MaxPooling2D((2, 2), name=f'pool_block{i+3}'))
        model.add(Dropout(0.25, name=f'dropout_block{i+3}'))
    
    # Final layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model
def load_mnist_data():
    """Load and prepare MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert to one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def run_benchmark(batch_size=64, epochs=20, experiment_id="mnist_benchmark"):
    """Run the benchmark and log performance"""
    # Initialize performance logger
    logger = HardwarePerformanceLogger(base_log_dir="hardware_performance_logs")
    logger.start_timing()
    build_start_time = int(round(time.time() * 1000))
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Create model
    model = create_cnn_model()
    build_time_ms = int(round(time.time() * 1000)) - build_start_time
    
    # Print model summary
    model.summary()
    
    # Count parameters to verify we match the target
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callback for performance logging
    performance_callback = HardwarePerformanceCallback(logger)
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[performance_callback, early_stopping]
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Record performance metrics
    model_id = f"mnist_cnn_{int(time.time())}"
    
    # Get model metrics
    model_metrics = logger.get_model_metrics(
        model, 
        model_id, 
        experiment_id,
        "CNN"
    )
    
    # Get training metrics
    training_metrics = logger.get_training_metrics(
        history,
        build_time_ms,
        batch_size,
        "Adam",
        0.001
    )
    
    # Save log file
    log_path = logger.save_log(model_metrics, training_metrics)
    print(f"\nPerformance log saved to: {log_path}")

if __name__ == "__main__":
    # Configure GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPUs")
    except:
        print("No GPUs found or error setting memory growth")
    
    # Run the benchmark
    run_benchmark(batch_size=128, epochs=30)
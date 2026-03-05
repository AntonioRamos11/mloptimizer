import numpy as np
import tensorflow as tf
from app.common.dataset import Dataset

class SyntheticDataset(Dataset):
    """Very small synthetic dataset for testing the optimization flow"""
    def __init__(self, batch_size=32, normalize=True):
    # Call the parent constructor with no arguments
        super().__init__()
        
        # Initialize these fields directly
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Rest of your initialization
        self.input_shape = (8, 8, 1)
        self.classes = 3
        self.sample_count = 200
        
        # Generate data once during initialization
        self._generate_data()
            
    def _generate_data(self):
        """Generate synthetic dataset with recognizable patterns for each class"""
        samples = self.sample_count
        
        # Create dataset with simple patterns
        X = np.random.normal(0, 1, (samples, *self.input_shape)).astype('float32')
        y = np.zeros(samples, dtype='int32')
        
        # Create three distinct patterns
        for i in range(samples):
            class_id = i % self.classes
            y[i] = class_id
            
            # Add a specific pattern for each class
            if class_id == 0:
                # Class 0: bright top-left corner
                X[i, :4, :4, :] += 2.0
            elif class_id == 1:
                # Class 1: bright top-right corner
                X[i, :4, 4:, :] += 2.0
            else:
                # Class 2: bright bottom half
                X[i, 4:, :, :] += 2.0
        
        # Split into train/val/test
        train_size = int(samples * 0.7)
        val_size = int(samples * 0.15)
        
        self.x_train = X[:train_size]
        self.y_train = y[:train_size]
        
        self.x_val = X[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        
        self.x_test = X[train_size+val_size:]
        self.y_test = y[train_size+val_size:]
    
    def get_tag(self):
        return "synthetic"
    
    def load(self):
        # Already loaded during initialization
        return True
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_classes_count(self):
        return self.classes
    
    def get_train_data(self, use_augmentation=False):
        return tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
    
    def get_validation_data(self):
        return tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
    
    def get_test_data(self):
        return tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
    
    def get_training_steps(self, use_augmentation=False):
        return max(1, len(self.x_train) // self.batch_size)
    
    def get_validation_steps(self):
        return max(1, len(self.x_val) // self.batch_size)
    
    def get_test_steps(self):
        return max(1, len(self.x_test) // self.batch_size)
    
    def get_testing_steps(self):
        """Alias for get_test_steps to satisfy the abstract base class"""
        return self.get_test_steps()
import tensorflow as tf
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.common.search_space import SearchSpaceType

@dataclass
class ModelParameters:
    """Base class for model architecture parameters"""
    pass

@dataclass
class ImageModelArchitectureParameters(ModelParameters):
    """Parameters for image classification models"""
    filters_1: int = 32
    filters_2: int = 64
    kernel_size: int = 3
    dense_units: int = 128
    dropout_rate: float = 0.2

@dataclass
class RegressionModelArchitectureParameters(ModelParameters):
    """Parameters for regression models"""
    units_1: int = 64
    units_2: int = 32
    dropout_rate: float = 0.2

@dataclass
class TimeSeriesModelArchitectureParameters(ModelParameters):
    """Parameters for time series models"""
    lstm_units: int = 64
    lstm_layers: int = 2
    dense_units: int = 32
    dropout_rate: float = 0.2
class SearchSpace:
    """Base search space class that defines model architecture possibilities"""
    
    def __init__(self, search_space_type):
        self.search_space_type = search_space_type
        self.hash_value = random.randint(1000, 9999)  # Mock hash value
    
    def get_type(self):
        return self.search_space_type
    
    def get_hash(self):
        return self.hash_value

class ImageSearchSpace(SearchSpace):
    """Search space for image classification models"""
    
    def __init__(self):
        super().__init__(SearchSpaceType.IMAGE)
    
    def generate_params(self, trial, input_shape):
        """Generate random parameters for the image model"""
        return ImageModelArchitectureParameters(
            filters_1=trial.suggest_int("filters_1", 16, 64, 16),
            filters_2=trial.suggest_int("filters_2", 32, 128, 32),
            kernel_size=trial.suggest_int("kernel_size", 3, 5, 2),
            dense_units=trial.suggest_int("dense_units", 64, 256, 64),
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        )

class RegressionSearchSpace(SearchSpace):
    """Search space for regression models"""
    
    def __init__(self):
        super().__init__(SearchSpaceType.REGRESSION)
    
    def generate_params(self, trial, input_shape):
        """Generate random parameters for the regression model"""
        return RegressionModelArchitectureParameters(
            units_1=trial.suggest_int("units_1", 32, 128, 32),
            units_2=trial.suggest_int("units_2", 16, 64, 16),
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        )

class TimeSeriesSearchSpace(SearchSpace):
    """Search space for time series models"""
    
    def __init__(self):
        super().__init__(SearchSpaceType.TIME_SERIES)
    
    def generate_params(self, trial, input_shape):
        """Generate random parameters for the time series model"""
        return TimeSeriesModelArchitectureParameters(
            lstm_units=trial.suggest_int("lstm_units", 32, 128, 32),
            lstm_layers=trial.suggest_int("lstm_layers", 1, 3),
            dense_units=trial.suggest_int("dense_units", 16, 64, 16),
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        )

class ModelArchitectureFactory:
    """Factory class that creates the appropriate search space based on the type"""
    
    @staticmethod
    def create(search_space_type):
        """Create model architecture factory based on search space type"""
        if search_space_type == SearchSpaceType.IMAGE:
            return ImageModelArchitectureFactory()
        elif search_space_type == SearchSpaceType.REGRESSION:
            return RegressionModelArchitectureFactory()
        elif search_space_type == SearchSpaceType.TIME_SERIES:
            return TimeSeriesModelArchitectureFactory()
        else:
            raise ValueError(f"Unsupported search space type: {search_space_type}")

class ImageModelArchitectureFactory:
    """Factory for image model architectures"""
    
    def __init__(self):
        self.search_space = ImageSearchSpace()
    
    def get_search_space(self):
        return self.search_space
    
    def generate_model_params(self, trial, input_shape):
        return self.search_space.generate_params(trial, input_shape)
    
    def build_model(self, params, input_shape, num_classes):
        """Build a simple CNN model for image classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(params.filters_1, params.kernel_size, activation='relu', 
                                  input_shape=input_shape, padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(params.filters_2, params.kernel_size, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params.dense_units, activation='relu'),
            tf.keras.layers.Dropout(params.dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

class RegressionModelArchitectureFactory:
    """Factory for regression model architectures"""
    
    def __init__(self):
        self.search_space = RegressionSearchSpace()
    
    def get_search_space(self):
        return self.search_space
    
    def generate_model_params(self, trial, input_shape):
        return self.search_space.generate_params(trial, input_shape)
    
    def build_model(self, params, input_shape, output_dim=1):
        """Build a simple MLP model for regression"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(params.units_1, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(params.dropout_rate),
            tf.keras.layers.Dense(params.units_2, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])
        return model

class TimeSeriesModelArchitectureFactory:
    """Factory for time series model architectures"""
    
    def __init__(self):
        self.search_space = TimeSeriesSearchSpace()
    
    def get_search_space(self):
        return self.search_space
    
    def generate_model_params(self, trial, input_shape):
        return self.search_space.generate_params(trial, input_shape)
    
    def build_model(self, params, input_shape, output_dim=1):
        """Build a simple LSTM model for time series"""
        model = tf.keras.Sequential()
        
        # Add LSTM layers
        for i in range(params.lstm_layers):
            return_sequences = i < params.lstm_layers - 1
            if i == 0:
                model.add(tf.keras.layers.LSTM(params.lstm_units, return_sequences=return_sequences, 
                                             input_shape=input_shape))
            else:
                model.add(tf.keras.layers.LSTM(params.lstm_units, return_sequences=return_sequences))
                
        model.add(tf.keras.layers.Dense(params.dense_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(params.dropout_rate))
        model.add(tf.keras.layers.Dense(output_dim))
        
        return model
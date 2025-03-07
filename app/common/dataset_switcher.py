from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
from system_parameters import SystemParameters as SP

class DatasetSwitchCallback(Callback):
    def __init__(self, datasets_list, on_switch_callback=None):
        """
        Callback to switch between datasets during training
        
        Args:
            datasets_list: List of dataset tuples (x_train, y_train, x_val, y_val)
            on_switch_callback: Function to call when switching datasets
        """
        super().__init__()
        self.datasets_list = datasets_list
        self.current_dataset_index = 0
        self.on_switch_callback = on_switch_callback
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Check if early stopping patience is reached
        early_stopping_patience = SP.EXPLORATION_EARLY_STOPPING_PATIENCE
        
        if self.patience_counter >= early_stopping_patience:
            self.current_dataset_index += 1
            
            # Check if we have more datasets to train on
            if self.current_dataset_index < len(self.datasets_list):
                print(f"\nSwitching to dataset {self.current_dataset_index + 1}/{len(self.datasets_list)}")
                
                # Reset metrics for the new dataset
                self.best_val_loss = float('inf')
                self.patience_counter = 0
                
                # Call the switch callback if provided
                if self.on_switch_callback:
                    self.on_switch_callback(self.current_dataset_index)
                
                # Stop current training loop
                self.model.stop_training = True
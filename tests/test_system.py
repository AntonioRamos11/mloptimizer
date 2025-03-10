import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system_parameters import SystemParameters
from app.init_nodes import InitNodes
from app.common.dataset import DatasetFactory
from app.common.model_builder import ModelBuilder
from app.common.search_space import SearchSpace


class TestMLOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Create minimal test configuration"""
        self.original_trials = SystemParameters.TRIALS
        self.original_epochs = SystemParameters.EXPLORATION_EPOCHS
        self.original_batch_size = SystemParameters.DATASET_BATCH_SIZE
        
        # Override system parameters for testing
        SystemParameters.TRIALS = 2
        SystemParameters.EXPLORATION_SIZE = 2
        SystemParameters.EXPLORATION_EPOCHS = 1
        SystemParameters.DATASET_BATCH_SIZE = 10
        
    def tearDown(self):
        """Restore original parameters"""
        SystemParameters.TRIALS = self.original_trials
        SystemParameters.EXPLORATION_EPOCHS = self.original_epochs
        SystemParameters.DATASET_BATCH_SIZE = self.original_batch_size
        
    def test_dataset_loading(self):
        """Test that datasets can be loaded correctly"""
        dataset_factory = DatasetFactory()
        train_data, test_data = dataset_factory.get_dataset(SystemParameters.DATASET_NAME)
        
        # Verify dataset structure
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        
        # Check if dataset has the expected shape
        for images, labels in train_data.take(1):
            self.assertEqual(images.shape[1:], SystemParameters.DATASET_SHAPE)
            
    def test_model_generation(self):
        """Test that models can be generated with the search space"""
        search_space = SearchSpace()
        params = search_space.generate_random_parameters()
        
        # Verify parameters contain expected keys
        self.assertIn('filters', params)
        self.assertIn('kernel_size', params)
        self.assertIn('pool_size', params)
        
        # Test model building
        model_builder = ModelBuilder()
        model = model_builder.build_model(params)
        
        # Verify model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.output_shape[-1], SystemParameters.DATASET_CLASSES)
        
    @patch('app.common.slave_node.SlaveNode.start_consuming')
    def test_slave_initialization(self, mock_start_consuming):
        """Test slave node initialization without actual training"""
        from app.init_nodes import InitNodes
        
        # Mock RabbitMQ connection
        with patch('app.common.slave_node.SlaveNode._setup_connection'):
            init_nodes = InitNodes()
            init_nodes.slave()
            
            # Verify slave started consuming
            mock_start_consuming.assert_called_once()
            
    @patch('app.master.master_node.MasterNode.start_optimization')
    def test_master_initialization(self, mock_start_optimization):
        """Test master node initialization without actual optimization"""
        from app.init_nodes import InitNodes
        
        # Mock RabbitMQ connection
        with patch('app.master.master_node.MasterNode._setup_connection'):
            init_nodes = InitNodes()
            init_nodes.master()
            
            # Verify optimization started
            mock_start_optimization.assert_called_once()
            
    def test_tiny_dataset_training(self):
        """Test training on a very small subset of data"""
        # Create tiny dataset for testing
        x_train = np.random.random((20, *SystemParameters.DATASET_SHAPE))
        y_train = np.random.randint(0, SystemParameters.DATASET_CLASSES, (20,))
        
        # Create and compile model
        model_builder = ModelBuilder()
        search_space = SearchSpace()
        params = search_space.generate_random_parameters()
        model = model_builder.build_model(params)
        
        # Compile model
        model.compile(
            optimizer=SystemParameters.OPTIMIZER,
            loss=SystemParameters.LOSS_FUNCTION,
            metrics=SystemParameters.METRICS
        )
        
        # Train for one epoch
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=10,
            verbose=0
        )
        
        # Verify training occurred
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)


if __name__ == '__main__':
    unittest.main()
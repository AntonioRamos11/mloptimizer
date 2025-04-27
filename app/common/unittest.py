import unittest
from unittest.mock import MagicMock, patch
import tensorflow as tf
import numpy as np
import time
from model import Model
from search_space import SearchSpaceType
from model_communication import MSGType, SocketCommunication

# Import the Model class from the parent directory

class TestModelBuildAndTrain(unittest.TestCase):
	def setUp(self):
		# Create mock objects for testing
		self.mock_request = MagicMock()
		self.mock_request.id = 1
		self.mock_request.experiment_id = "test_experiment"
		self.mock_request.training_type = "test"
		self.mock_request.search_space_type = 1  # IMAGE type
		self.mock_request.architecture = MagicMock()
		self.mock_request.epochs = 5
		self.mock_request.early_stopping_patience = 3
		self.mock_request.is_partial_training = False
		
		# Mock dataset
		self.mock_dataset = MagicMock()
		self.mock_dataset.batch_size = 32
		self.mock_dataset.get_input_shape.return_value = (28, 28, 1)
		self.mock_dataset.get_classes_count.return_value = 10
		
		# Create mock training data
		mock_train_data = tf.data.Dataset.from_tensor_slices(
			(np.random.random((10, 28, 28, 1)), np.random.randint(0, 10, size=(10,)))
		).batch(2)
		mock_val_data = tf.data.Dataset.from_tensor_slices(
			(np.random.random((6, 28, 28, 1)), np.random.randint(0, 10, size=(6,)))
		).batch(2)
		mock_test_data = tf.data.Dataset.from_tensor_slices(
			(np.random.random((4, 28, 28, 1)), np.random.randint(0, 10, size=(4,)))
		).batch(2)
		
		self.mock_dataset.get_train_data.return_value = mock_train_data
		self.mock_dataset.get_validation_data.return_value = mock_val_data
		self.mock_dataset.get_test_data.return_value = mock_test_data
		self.mock_dataset.get_training_steps.return_value = 5
		self.mock_dataset.get_validation_steps.return_value = 3
		
		# Create the model instance with mocks
		self.model = Model(self.mock_request, self.mock_dataset)
		
		# Mock the build_model method
		self.model.build_model = MagicMock(return_value=self._create_test_model())
		
		# Mock SocketCommunication
		self.original_decide_print_form = SocketCommunication.decide_print_form
		SocketCommunication.decide_print_form = MagicMock()
		
	def tearDown(self):
		# Restore original method
		SocketCommunication.decide_print_form = self.original_decide_print_form
		
	def _create_test_model(self):
		"""Create a simple model for testing"""
		model = tf.keras.Sequential([
			tf.keras.layers.Dense(10, input_shape=(28*28,), activation='relu'),
			tf.keras.layers.Dense(10, activation='softmax')
		])
		model.compile(
			optimizer='adam', 
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		return model

	@patch('tensorflow.config.list_physical_devices')
	def test_build_and_train_cpu_only(self, mock_list_devices):
		"""Test training on CPU only"""
		# Mock no GPUs available
		mock_list_devices.return_value = []
		
		with patch('tensorflow.distribute.OneDeviceStrategy') as mock_strategy:
			# Mock strategy context
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			# Mock fit and evaluate
			with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					# Test method
					start_time = time.time()
					result, did_finish = self.model.build_and_train()
					elapsed = time.time() - start_time
					
					# Assertions
					self.assertIsInstance(result, float)
					self.assertTrue(did_finish)
					mock_strategy.assert_called_once_with(device='/cpu:0')
					
					# Check hardware logging
					hardware_logged = False
					for call in SocketCommunication.decide_print_form.call_args_list:
						args = call[0]
						if len(args) > 1 and isinstance(args[1], dict):
							if "Total weights" in args[1].get('msg', '') and "GPU(s)" in args[1].get('msg', ''):
								hardware_logged = True
					self.assertTrue(hardware_logged, "Hardware info should be logged")
	
	@patch('tensorflow.config.list_physical_devices')
	def test_build_and_train_single_gpu(self, mock_list_devices):
		"""Test training on a single GPU"""
		# Mock single GPU
		mock_list_devices.return_value = [MagicMock()]
		
		with patch('tensorflow.distribute.OneDeviceStrategy') as mock_strategy:
			# Mock strategy context
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			# Mock fit and evaluate
			with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					# Test method
					start_time = time.time()
					result, did_finish = self.model.build_and_train()
					elapsed = time.time() - start_time
					
					# Assertions
					self.assertIsInstance(result, float)
					mock_strategy.assert_called_once_with(device='/gpu:0')
					
					# Verify batch size wasn't changed for single GPU
					self.assertEqual(self.mock_dataset.batch_size, 32)

	@patch('tensorflow.config.list_physical_devices')
	def test_build_and_train_multi_gpu(self, mock_list_devices):
		"""Test training on multiple GPUs with batch size scaling"""
		# Mock multiple GPUs
		mock_list_devices.return_value = [MagicMock(), MagicMock()]
		
		with patch('tensorflow.distribute.MirroredStrategy') as mock_strategy:
			# Mock strategy context
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			# Mock fit and evaluate
			with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					original_batch_size = self.mock_dataset.batch_size
					
					# Test method
					start_time = time.time()
					result, did_finish = self.model.build_and_train()
					elapsed = time.time() - start_time
					
					# Verify batch size was reset after training
					self.assertEqual(self.mock_dataset.batch_size, original_batch_size)
					
					# Check that MirroredStrategy was used with correct parameters
					self.assertTrue(mock_strategy.called)
					self.assertEqual(mock_strategy.call_args[1]['devices'], ['/gpu:0', '/gpu:1'])

	@patch('tensorflow.config.list_physical_devices')
	def test_early_stopping_behavior(self, mock_list_devices):
		"""Test early stopping behavior when training doesn't complete all epochs"""
		# Mock no GPUs
		mock_list_devices.return_value = []
		
		with patch('tensorflow.distribute.OneDeviceStrategy') as mock_strategy:
			# Mock strategy context
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			# Create history with fewer epochs than requested (early stopping)
			history = MagicMock()
			history.history = {'loss': [0.3, 0.2, 0.1]}  # Only 3 epochs, not 5
			
			# Mock fit and evaluate with our custom history
			with patch.object(tf.keras.Model, 'fit', return_value=history):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					result, did_finish = self.model.build_and_train()
					
					# Should detect incomplete epochs
					self.assertFalse(did_finish)

	@patch('tensorflow.config.list_physical_devices')
	def test_performance_comparison_across_devices(self, mock_list_devices):
		"""Test performance can be compared across different hardware setups"""
		# Create performance records for comparison
		performance_data = {}
		
		# Test with CPU
		mock_list_devices.return_value = []
		with patch('tensorflow.distribute.OneDeviceStrategy') as mock_strategy:
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					start_time = time.time()
					self.model.build_and_train()
					cpu_time = time.time() - start_time
					performance_data['CPU'] = cpu_time
		
		# Test with single GPU
		mock_list_devices.return_value = [MagicMock()]
		with patch('tensorflow.distribute.OneDeviceStrategy') as mock_strategy:
			mock_strategy_instance = MagicMock()
			mock_strategy.return_value = mock_strategy_instance
			mock_strategy_instance.__enter__ = MagicMock()
			mock_strategy_instance.__exit__ = MagicMock(return_value=None)
			
			with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
				with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
					start_time = time.time()
					self.model.build_and_train()
					gpu_time = time.time() - start_time
					performance_data['Single GPU'] = gpu_time
		
		# Compare times (in actual testing with real hardware, these would be different)
		self.assertIsInstance(performance_data['CPU'], float)
		self.assertIsInstance(performance_data['Single GPU'], float)
		
		# In real scenarios, we would expect GPU to be faster
		# For mock tests, we only verify data is collected properly
		
	def test_different_architecture_comparison(self):
		"""Test comparing different architecture types"""
		# Create a logger to capture different architecture metrics
		architecture_metrics = {}
		
		# Test with IMAGE CNN architecture
		self.mock_request.search_space_type = 1  # IMAGE
		self.model.search_space_type = SearchSpaceType.IMAGE
		self.model.model_params.base_architecture = 'cnn'
		
		# Inject our metrics capture into the model method
		original_build_image_model = self.model.build_image_model
		def capture_cnn_metrics(*args, **kwargs):
			start = time.time()
			model = original_build_image_model(*args, **kwargs)
			elapsed = time.time() - start
			architecture_metrics['CNN'] = {
				'build_time': elapsed,
				'parameter_count': model.count_params()
			}
			return model
		
		# Testing with architecture type tracking
		with patch.object(self.model, 'build_image_model', side_effect=capture_cnn_metrics):
			with patch('tensorflow.config.list_physical_devices', return_value=[]):
				with patch('tensorflow.distribute.OneDeviceStrategy'):
					with patch.object(tf.keras.Model, 'fit', return_value=MagicMock(history={'loss': [0.1]*5})):
						with patch.object(tf.keras.Model, 'evaluate', return_value=[0.1, 0.9]):
							# We only need to verify architecture metrics are captured
							self.model.build_and_train()
							
		# Verify metrics were collected
		self.assertIn('CNN', architecture_metrics)
		self.assertIn('build_time', architecture_metrics['CNN'])
		self.assertIn('parameter_count', architecture_metrics['CNN'])


if __name__ == '__main__':
	unittest.main()
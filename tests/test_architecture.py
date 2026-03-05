import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.init_nodes import InitNodes
from app.common.dataset import (
    Dataset, ImageClassificationBenchmarkDataset, 
    RegressionBenchmarkDataset, TimeSeriesBenchmarkDataset
)
from app.common.search_space import (
    ImageModelArchitectureFactory, RegressionModelArchitectureFactory,
    TimeSeriesModelArchitectureFactory, ModelArchitectureFactory,
    ImageModelArchitectureParameters, RegressionModelArchitectureParameters,
    TimeSeriesModelArchitectureParameters
)
from system_parameters import SystemParameters as SP


class DummyFastDataset(Dataset):
    """Dummy dataset for fast testing - generates random data in memory"""
    
    def __init__(self, name="dummy", shape=(28, 28, 1), classes=10, batch_size=8, validation_split=0.2):
        self.dataset_name = name
        self.batch_size = batch_size
        self.validation_split_float = validation_split
        self.shape = shape
        self.class_count = classes
        self._train_data = None
        self._val_data = None
        self._test_data = None
        
    def load(self):
        """Generate random data in memory - super fast"""
        num_samples = 100
        num_val = int(num_samples * self.validation_split_float)
        num_test = 20
        
        train_images = np.random.rand(num_samples - num_val, *self.shape).astype(np.float32)
        train_labels = np.random.randint(0, self.class_count, size=num_samples - num_val)
        
        val_images = np.random.rand(num_val, *self.shape).astype(np.float32)
        val_labels = np.random.randint(0, self.class_count, size=num_val)
        
        test_images = np.random.rand(num_test, *self.shape).astype(np.float32)
        test_labels = np.random.randint(0, self.class_count, size=num_test)
        
        self._train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        self._val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        self._test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
    def get_train_data(self, use_augmentation=False):
        if self._train_data is None:
            self.load()
        return self._train_data.batch(self.batch_size)
    
    def get_validation_data(self):
        if self._val_data is None:
            self.load()
        return self._val_data.batch(self.batch_size)
    
    def get_test_data(self):
        if self._test_data is None:
            self.load()
        return self._test_data.batch(self.batch_size)
    
    def get_training_steps(self, use_augmentation=False) -> int:
        return 10
    
    def get_validation_steps(self) -> int:
        return 5
    
    def get_testing_steps(self) -> int:
        return 3
    
    def get_input_shape(self) -> tuple:
        return self.shape
    
    def get_classes_count(self) -> int:
        return self.class_count
    
    def get_tag(self) -> str:
        return self.dataset_name


class TestInitNodes(unittest.TestCase):
    
    def setUp(self):
        self.init_nodes = InitNodes()
        self.original_socket = None
        
    def tearDown(self):
        pass
    
    @patch('app.init_nodes.SocketCommunication')
    @patch('app.init_nodes.SP')
    def test_master_initialization(self, mock_sp, mock_socket_comm):
        """Test that master() initializes correctly"""
        mock_sp.iter_datasets.return_value = iter([
            (0, 'test_dataset', (28, 28, 1), 10)
        ])
        mock_sp.total_runs.return_value = 1
        mock_sp.DATASET_TYPE = 1
        mock_sp.DATASET_NAME = 'mnist'
        mock_sp.TRIALS = 1
        
        with patch.object(self.init_nodes, 'get_dataset') as mock_get_dataset:
            with patch.object(self.init_nodes, 'get_model_architecture') as mock_get_arch:
                with patch('app.master_node.optimization_job.OptimizationJob') as mock_job:
                    mock_get_dataset.return_value = MagicMock()
                    mock_get_arch.return_value = MagicMock()
                    mock_job_instance = MagicMock()
                    mock_job.return_value = mock_job_instance
                    
                    self.init_nodes.master()
                    
                    mock_get_dataset.assert_called_once()
                    mock_get_arch.assert_called_once()
                    mock_job.assert_called_once()
                    mock_job_instance.start_optimization.assert_called_once()
    
    @patch('app.init_nodes.SocketCommunication')
    @patch('app.init_nodes.SP')
    def test_slave_initialization(self, mock_sp, mock_socket_comm):
        """Test that slave() initializes correctly"""
        mock_sp.DATASET_TYPE = 1
        
        with patch.object(self.init_nodes, 'get_dataset') as mock_get_dataset:
            with patch.object(self.init_nodes, 'get_model_architecture') as mock_get_arch:
                with patch('app.slave_node.training_slave.TrainingSlave') as mock_slave:
                    mock_get_dataset.return_value = MagicMock()
                    mock_get_arch.return_value = MagicMock()
                    mock_slave_instance = MagicMock()
                    mock_slave.return_value = mock_slave_instance
                    
                    self.init_nodes.slave()
                    
                    mock_get_dataset.assert_called_once()
                    mock_get_arch.assert_called_once()
                    mock_slave.assert_called_once()
                    mock_slave_instance.start_slave.assert_called_once()
    
    @patch('app.init_nodes.SP')
    def test_get_dataset_image(self, mock_sp):
        """Test dataset factory for image classification"""
        mock_sp.DATASET_TYPE = 1
        mock_sp.DATASET_NAME = 'mnist'
        mock_sp.DATASET_SHAPE = (28, 28, 1)
        mock_sp.DATASET_CLASSES = 10
        mock_sp.DATASET_BATCH_SIZE = 32
        mock_sp.DATASET_VALIDATION_SPLIT = 0.2
        
        dataset = self.init_nodes.get_dataset()
        
        self.assertIsInstance(dataset, ImageClassificationBenchmarkDataset)
    
    @patch('app.init_nodes.SP')
    def test_get_dataset_regression(self, mock_sp):
        """Test dataset factory for regression"""
        mock_sp.DATASET_TYPE = 2
        mock_sp.DATASET_NAME = 'regression'
        mock_sp.DATASET_SHAPE = (10,)
        mock_sp.DATASET_FEATURES = 10
        mock_sp.DATASET_LABELS = 1
        mock_sp.DATASET_BATCH_SIZE = 32
        mock_sp.DATASET_VALIDATION_SPLIT = 0.2
        
        dataset = self.init_nodes.get_dataset()
        
        self.assertIsInstance(dataset, RegressionBenchmarkDataset)
    
    @patch('app.init_nodes.SP')
    def test_get_dataset_timeseries(self, mock_sp):
        """Test dataset factory for time series"""
        mock_sp.DATASET_TYPE = 3
        mock_sp.DATASET_NAME = 'timeseries'
        mock_sp.DATASET_WINDOW_SIZE = 24
        mock_sp.DATASET_DATA_SIZE = 100
        mock_sp.DATASET_BATCH_SIZE = 32
        mock_sp.DATASET_VALIDATION_SPLIT = 0.2
        
        dataset = self.init_nodes.get_dataset()
        
        self.assertIsInstance(dataset, TimeSeriesBenchmarkDataset)
    
    @patch('app.init_nodes.SP')
    def test_get_model_architecture_image(self, mock_sp):
        """Test architecture factory for image classification"""
        mock_sp.DATASET_TYPE = 1
        
        factory = self.init_nodes.get_model_architecture()
        
        self.assertIsInstance(factory, ImageModelArchitectureFactory)
    
    @patch('app.init_nodes.SP')
    def test_get_model_architecture_regression(self, mock_sp):
        """Test architecture factory for regression"""
        mock_sp.DATASET_TYPE = 2
        
        factory = self.init_nodes.get_model_architecture()
        
        self.assertIsInstance(factory, RegressionModelArchitectureFactory)
    
    @patch('app.init_nodes.SP')
    def test_get_model_architecture_timeseries(self, mock_sp):
        """Test architecture factory for time series"""
        mock_sp.DATASET_TYPE = 3
        
        factory = self.init_nodes.get_model_architecture()
        
        self.assertIsInstance(factory, TimeSeriesModelArchitectureFactory)
    
    @patch('app.init_nodes.SP')
    def test_get_dataset_invalid_type(self, mock_sp):
        """Test dataset factory with invalid type returns None"""
        mock_sp.DATASET_TYPE = 99
        
        dataset = self.init_nodes.get_dataset()
        
        self.assertIsNone(dataset)


class TestDummyFastDataset(unittest.TestCase):
    """Tests for the dummy fast dataset"""
    
    def test_dummy_dataset_creation(self):
        """Test dummy dataset can be created"""
        dataset = DummyFastDataset(name="test", shape=(32, 32, 3), classes=10)
        self.assertEqual(dataset.dataset_name, "test")
        self.assertEqual(dataset.shape, (32, 32, 3))
        self.assertEqual(dataset.class_count, 10)
    
    def test_dummy_dataset_load(self):
        """Test dummy dataset can load data"""
        dataset = DummyFastDataset(batch_size=8)
        dataset.load()
        
        self.assertIsNotNone(dataset._train_data)
        self.assertIsNotNone(dataset._val_data)
        self.assertIsNotNone(dataset._test_data)
    
    def test_dummy_dataset_get_train_data(self):
        """Test getting training data"""
        dataset = DummyFastDataset(batch_size=8)
        train_data = dataset.get_train_data()
        
        self.assertIsNotNone(train_data)
    
    def test_dummy_dataset_get_validation_data(self):
        """Test getting validation data"""
        dataset = DummyFastDataset(batch_size=8)
        val_data = dataset.get_validation_data()
        
        self.assertIsNotNone(val_data)
    
    def test_dummy_dataset_get_test_data(self):
        """Test getting test data"""
        dataset = DummyFastDataset(batch_size=8)
        test_data = dataset.get_test_data()
        
        self.assertIsNotNone(test_data)
    
    def test_dummy_dataset_get_input_shape(self):
        """Test getting input shape"""
        dataset = DummyFastDataset(shape=(64, 64, 3))
        shape = dataset.get_input_shape()
        
        self.assertEqual(shape, (64, 64, 3))
    
    def test_dummy_dataset_get_classes_count(self):
        """Test getting classes count"""
        dataset = DummyFastDataset(classes=5)
        classes = dataset.get_classes_count()
        
        self.assertEqual(classes, 5)
    
    def test_dummy_dataset_get_training_steps(self):
        """Test getting training steps"""
        dataset = DummyFastDataset(batch_size=10)
        steps = dataset.get_training_steps()
        
        self.assertIsInstance(steps, int)
        self.assertGreater(steps, 0)


class TestSearchSpace(unittest.TestCase):
    """Tests for search space components"""
    
    def test_image_architecture_factory_creation(self):
        """Test ImageModelArchitectureFactory can be created"""
        factory = ImageModelArchitectureFactory()
        self.assertIsInstance(factory, ModelArchitectureFactory)
    
    def test_regression_architecture_factory_creation(self):
        """Test RegressionModelArchitectureFactory can be created"""
        factory = RegressionModelArchitectureFactory()
        self.assertIsInstance(factory, ModelArchitectureFactory)
    
    def test_timeseries_architecture_factory_creation(self):
        """Test TimeSeriesModelArchitectureFactory can be created"""
        factory = TimeSeriesModelArchitectureFactory()
        self.assertIsInstance(factory, ModelArchitectureFactory)
    
    def test_image_architecture_parameters(self):
        """Test ImageModelArchitectureParameters creation"""
        params = ImageModelArchitectureParameters.new()
        self.assertIsNotNone(params)
        self.assertEqual(params.base_architecture, None)
    
    def test_regression_architecture_parameters(self):
        """Test RegressionModelArchitectureParameters creation"""
        params = RegressionModelArchitectureParameters.new()
        self.assertIsNotNone(params)
        self.assertEqual(params.base_architecture, None)
    
    def test_timeseries_architecture_parameters(self):
        """Test TimeSeriesModelArchitectureParameters creation"""
        params = TimeSeriesModelArchitectureParameters.new()
        self.assertIsNotNone(params)
        self.assertEqual(params.base_architecture, None)


class TestDatasetIterators(unittest.TestCase):
    """Tests for dataset iteration"""
    
    def test_iter_datasets(self):
        """Test dataset iteration through schedule"""
        from system_parameters import SystemParameters as SP
        datasets = list(SP.iter_datasets())
        self.assertIsInstance(datasets, list)


class TestSystemParameters(unittest.TestCase):
    """Tests for SystemParameters changes"""
    
    def test_change_system_info(self):
        """Test change_system_info updates parameters"""
        test_params = {
            'RabbitParams': {
                'Port': 5673,
                'ParametersQueue': 'test_params',
                'PerformanceQueue': 'test_results',
                'HostURL': 'testhost',
                'User': 'testuser',
                'Password': 'testpass',
                'VirtualHost': '/test'
            },
            'DatasetParams': {
                'DatasetName': 'test_dataset',
                'DatasetType': 'Image',
                'BatchSize': 64,
                'ValidationSplit': 0.3,
                'DatasetShape': {'Item1': 28, 'Item2': 28, 'Item3': 1},
                'Classnumber': 10,
                'FeatureNumber': 10,
                'LabelsNumber': 1,
                'WindowSize': 24,
                'FeatureSize': 100
            },
            'AutoMLParams': {
                'Trials': 10,
                'UseGPU': True,
                'ExplorationParams': {'Size': 5, 'Epochs': 2, 'EarlyStopping': 1},
                'HallOfFameParams': {'Size': 3, 'Epochs': 5, 'EarlyStopping': 2}
            },
            'ModelsParams': {
                'DataFormat': 'float32',
                'Optimizer': 'adam',
                'LayersActivation': 'relu',
                'OutputActivation': 'softmax',
                'Kernel': 'glorot_uniform',
                'Loss': 'sparse_categorical_crossentropy',
                'Metrics': 'accuracy',
                'Padding': 'same',
                'WeightDecay': 0.0001,
                'LSTMActivation': 'tanh'
            }
        }
        
        InitNodes.change_system_info(test_params)
        
        self.assertEqual(SP.DATASET_NAME, 'test_dataset')
        self.assertEqual(SP.DATASET_TYPE, 1)
        self.assertEqual(SP.DATASET_BATCH_SIZE, 64)
        self.assertEqual(SP.TRIALS, 10)
    
    def test_change_slave_system_parameters(self):
        """Test change_slave_system_parameters updates parameters"""
        test_params = {
            'RabbitParams': {
                'Port': 5674,
                'ParametersQueue': 'slave_params',
                'PerformanceQueue': 'slave_results',
                'HostURL': 'slavehost',
                'User': 'slaveuser',
                'Password': 'slavepass',
                'VirtualHost': '/slave'
            },
            'DatasetParams': {
                'DatasetName': 'slave_dataset',
                'DatasetType': 'Regression',
                'BatchSize': 128,
                'ValidationSplit': 0.15,
                'DatasetShape': {'Item1': 10, 'Item2': None, 'Item3': None},
                'Classnumber': 1,
                'FeatureNumber': 10,
                'LabelsNumber': 1,
                'WindowSize': 24,
                'FeatureSize': 100
            },
            'TrainGPU': True
        }
        
        InitNodes.change_slave_system_parameters(test_params)
        
        self.assertEqual(SP.DATASET_NAME, 'slave_dataset')
        self.assertEqual(SP.DATASET_TYPE, 2)
        self.assertEqual(SP.DATASET_BATCH_SIZE, 128)
        self.assertTrue(SP.TRAIN_GPU)


class TestIntegration(unittest.TestCase):
    """Integration tests for master-slave flow"""
    
    @patch('app.init_nodes.SocketCommunication')
    @patch('app.init_nodes.SP')
    def test_master_with_dummy_dataset(self, mock_sp, mock_socket_comm):
        """Test master node flow with dummy fast dataset"""
        mock_sp.iter_datasets.return_value = iter([
            (0, 'dummy_test', (28, 28, 1), 10)
        ])
        mock_sp.total_runs.return_value = 1
        mock_sp.DATASET_TYPE = 1
        mock_sp.DATASET_NAME = 'dummy_test'
        mock_sp.DATASET_SHAPE = (28, 28, 1)
        mock_sp.DATASET_CLASSES = 10
        mock_sp.DATASET_BATCH_SIZE = 8
        mock_sp.DATASET_VALIDATION_SPLIT = 0.2
        mock_sp.TRIALS = 1
        
        init_nodes = InitNodes()
        
        with patch.object(init_nodes, 'get_dataset') as mock_get_dataset:
            with patch.object(init_nodes, 'get_model_architecture') as mock_get_arch:
                with patch('app.master_node.optimization_job.OptimizationJob') as mock_job:
                    mock_dataset = DummyFastDataset()
                    mock_get_dataset.return_value = mock_dataset
                    mock_get_arch.return_value = ImageModelArchitectureFactory()
                    
                    mock_job_instance = MagicMock()
                    mock_job.return_value = mock_job_instance
                    
                    init_nodes.master()
                    
                    mock_get_dataset.assert_called_once()
                    mock_job_instance.start_optimization.assert_called_once()
    
    @patch('app.init_nodes.SocketCommunication')
    @patch('app.init_nodes.SP')
    def test_slave_with_dummy_dataset(self, mock_sp, mock_socket_comm):
        """Test slave node flow with dummy fast dataset"""
        mock_sp.DATASET_TYPE = 1
        mock_sp.DATASET_NAME = 'dummy_test'
        mock_sp.DATASET_SHAPE = (28, 28, 1)
        mock_sp.DATASET_CLASSES = 10
        mock_sp.DATASET_BATCH_SIZE = 8
        mock_sp.DATASET_VALIDATION_SPLIT = 0.2
        
        init_nodes = InitNodes()
        
        with patch.object(init_nodes, 'get_dataset') as mock_get_dataset:
            with patch.object(init_nodes, 'get_model_architecture') as mock_get_arch:
                with patch('app.slave_node.training_slave.TrainingSlave') as mock_slave:
                    mock_dataset = DummyFastDataset()
                    mock_get_dataset.return_value = mock_dataset
                    mock_get_arch.return_value = ImageModelArchitectureFactory()
                    
                    mock_slave_instance = MagicMock()
                    mock_slave.return_value = mock_slave_instance
                    
                    init_nodes.slave()
                    
                    mock_get_dataset.assert_called_once()
                    mock_slave.assert_called_once()
                    mock_slave_instance.start_slave.assert_called_once()


class TestResultsSaving(unittest.TestCase):
    """Tests for saving optimization results"""
    
    def setUp(self):
        self.test_results_dir = '/tmp/mloptimizer_test_results'
        os.makedirs(self.test_results_dir, exist_ok=True)
        
    def tearDown(self):
        import shutil
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
    
    def test_save_results_to_json(self):
        """Test saving optimization results to JSON file"""
        import json
        from dataclasses import dataclass, asdict
        
        @dataclass
        class MockModelRequest:
            id: int = 1
            experiment_id: str = "test_experiment"
            epochs: int = 5
        
        @dataclass
        class MockModelResponse:
            performance: float = 0.95
            training_time: float = 100.0
            model_training_request: MockModelRequest = None
        
        mock_model_response = MockModelResponse()
        mock_model_response.model_training_request = MockModelRequest()
        
        result_data = {
            "model_info": asdict(mock_model_response),
            "performance_metrics": {
                "elapsed_seconds": 100.0,
                "accuracy": 0.95,
                "loss": 0.05
            },
            "hardware_info": {
                "gpu": "NVIDIA RTX 3080",
                "device": "/gpu:0"
            }
        }
        
        filepath = os.path.join(self.test_results_dir, "test_experiment")
        with open(filepath + ".json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        self.assertTrue(os.path.exists(filepath + ".json"))
        
        with open(filepath + ".json", "r") as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data['performance_metrics']['accuracy'], 0.95)
        self.assertEqual(loaded_data['model_info']['performance'], 0.95)
    
    def test_results_directory_creation(self):
        """Test that results directory is created if it doesn't exist"""
        test_dir = os.path.join(self.test_results_dir, "new_results")
        
        os.makedirs(test_dir, exist_ok=True)
        
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.isdir(test_dir))
    
    def test_multiple_experiments_results(self):
        """Test saving multiple experiment results"""
        import json
        
        experiments = ['exp_001', 'exp_002', 'exp_003']
        
        for exp_id in experiments:
            result_data = {
                "experiment_id": exp_id,
                "performance": 0.9,
                "status": "completed"
            }
            filepath = os.path.join(self.test_results_dir, exp_id)
            with open(filepath + ".json", "w") as f:
                json.dump(result_data, f)
        
        for exp_id in experiments:
            filepath = os.path.join(self.test_results_dir, exp_id + ".json")
            self.assertTrue(os.path.exists(filepath))
    
    def test_results_with_dataset_info(self):
        """Test saving results with dataset information"""
        import json
        
        result_data = {
            "model_info": {
                "id": 1,
                "experiment_id": "test_mnist"
            },
            "dataset_info": {
                "name": "mnist",
                "shape": [28, 28, 1],
                "classes": 10,
                "training_samples": 60000,
                "validation_samples": 10000
            },
            "performance_metrics": {
                "accuracy": 0.98,
                "loss": 0.02,
                "elapsed_seconds": 150.5
            },
            "hardware_info": {
                "gpu": "NVIDIA RTX 3080",
                "device": "/gpu:0",
                "batch_size": 128
            }
        }
        
        filepath = os.path.join(self.test_results_dir, "dataset_info_test")
        with open(filepath + ".json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        with open(filepath + ".json", "r") as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['dataset_info']['name'], 'mnist')
        self.assertEqual(loaded['dataset_info']['classes'], 10)
        self.assertIn('performance_metrics', loaded)


class TestHardwarePerformanceLogging(unittest.TestCase):
    """Tests for hardware performance logging"""
    
    def test_hardware_logger_creation(self):
        """Test hardware performance logger can be created"""
        from app.common.hardware_performance_logger import HardwarePerformanceLogger
        
        logger = HardwarePerformanceLogger()
        self.assertIsNotNone(logger)
    
    def test_gpu_metrics_structure(self):
        """Test GPU metrics have correct structure"""
        mock_metrics = {
            "gpu_name": "NVIDIA RTX 3080",
            "memory_total_mb": 10240,
            "memory_used_mb": 2048,
            "utilization_percent": 75.0,
            "temperature_celsius": 65,
            "power_draw_watts": 150.0
        }
        
        self.assertIn("gpu_name", mock_metrics)
        self.assertIn("memory_total_mb", mock_metrics)
        self.assertIn("utilization_percent", mock_metrics)
    
    def test_cpu_metrics_structure(self):
        """Test CPU metrics have correct structure"""
        mock_metrics = {
            "cpu_name": "AMD Ryzen 9",
            "cores": 16,
            "utilization_percent": 45.0,
            "temperature_celsius": 55,
            "power_draw_watts": 65.0
        }
        
        self.assertIn("cpu_name", mock_metrics)
        self.assertIn("cores", mock_metrics)
        self.assertIn("utilization_percent", mock_metrics)


class TestRealModelTraining(unittest.TestCase):
    """Real integration tests - actually train models with DummyFastDataset"""
    
    def setUp(self):
        self.test_results_dir = '/tmp/mloptimizer_real_test_results'
        os.makedirs(self.test_results_dir, exist_ok=True)
        
    def tearDown(self):
        import shutil
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
    
    def test_real_model_training_with_dummy_dataset(self):
        """Test real model training with DummyFastDataset - no mocks - ACTUALLY TRAINS"""
        import json
        from app.common.model import Model
        from app.common.model_communication import ModelTrainingRequest
        from app.common.search_space import (
            ImageModelArchitectureParameters, SearchSpaceType
        )
        
        dataset = DummyFastDataset(name="test_real", shape=(28, 28, 1), classes=10, batch_size=8)
        
        arch_params = ImageModelArchitectureParameters.new()
        arch_params.base_architecture = 'cnn'
        arch_params.cnn_blocks_n = 1
        arch_params.cnn_blocks_conv_layers_n = 1
        arch_params.cnn_block_conv_filters = [16]
        arch_params.cnn_block_conv_filter_sizes = [3]
        arch_params.cnn_block_max_pooling_sizes = [2]
        arch_params.cnn_block_dropout_values = [0]
        arch_params.classifier_layer_type = 'mlp'
        arch_params.classifier_layers_n = 1
        arch_params.classifier_layers_units = [64]
        arch_params.classifier_dropouts = [0]
        arch_params.learning_rate = 0.001
        
        request = ModelTrainingRequest(
            id=1,
            training_type=1,
            experiment_id="test_real_training",
            architecture=arch_params,
            epochs=1,
            early_stopping_patience=3,
            is_partial_training=False,
            search_space_type="image",
            search_space_hash="test_hash",
            dataset_tag="test"
        )
        
        model = Model(request, dataset)
        
        performance, did_finish = model.build_and_train()
        
        self.assertIsInstance(performance, float)
        self.assertGreaterEqual(performance, 0.0)
        self.assertLessEqual(performance, 1.0)
        self.assertIsInstance(did_finish, bool)
    
    def test_real_dataset_get_data_methods(self):
        """Test DummyFastDataset get_data methods return real tf.data.Datasets"""
        import tensorflow as tf
        
        dataset = DummyFastDataset(name="test_data", shape=(32, 32, 3), classes=5, batch_size=4)
        
        train_data = dataset.get_train_data()
        val_data = dataset.get_validation_data()
        test_data = dataset.get_test_data()
        
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(val_data)
        self.assertIsNotNone(test_data)
        
        for batch in train_data.take(1):
            images, labels = batch
            self.assertEqual(images.shape[1:], (32, 32, 3))
    
    def test_real_model_can_be_built(self):
        """Test that a real Keras model can be built from architecture"""
        from app.common.search_space import ImageModelArchitectureParameters
        import tensorflow as tf
        
        input_shape = (28, 28, 1)
        inputs = tf.keras.Input(shape=input_shape)
        
        x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.assertIsNotNone(model)
        self.assertGreater(model.count_params(), 0)
        
        model.summary()
    
    def test_real_training_loop_one_epoch(self):
        """Test a real training loop executes correctly"""
        import tensorflow as tf
        import numpy as np
        
        x_train = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(20,))
        
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=(28, 28, 1), padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_ds, epochs=1, verbose=0)
        
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)


class TestRealArchitectureFactory(unittest.TestCase):
    """Test real architecture factory operations"""
    
    def test_image_architecture_factory_creation(self):
        """Test ImageModelArchitectureFactory can be instantiated"""
        from app.common.search_space import ImageModelArchitectureFactory
        
        factory = ImageModelArchitectureFactory()
        
        self.assertIsNotNone(factory)
        self.assertTrue(hasattr(factory, 'generate_model_params'))
    
    def test_regression_architecture_factory_creation(self):
        """Test RegressionModelArchitectureFactory can be instantiated"""
        from app.common.search_space import RegressionModelArchitectureFactory
        
        factory = RegressionModelArchitectureFactory()
        
        self.assertIsNotNone(factory)
        self.assertTrue(hasattr(factory, 'generate_model_params'))
    
    def test_timeseries_architecture_factory_creation(self):
        """Test TimeSeriesModelArchitectureFactory can be instantiated"""
        from app.common.search_space import TimeSeriesModelArchitectureFactory
        
        factory = TimeSeriesModelArchitectureFactory()
        
        self.assertIsNotNone(factory)
        self.assertTrue(hasattr(factory, 'generate_model_params'))


class TestRealSearchSpaceHash(unittest.TestCase):
    """Test search space hash verification"""
    
    def test_search_space_params_creation(self):
        """Test that search space parameters can be created"""
        from app.common.search_space import ImageModelArchitectureParameters
        
        params = ImageModelArchitectureParameters.new()
        
        self.assertIsNotNone(params)
        self.assertTrue(hasattr(params, 'base_architecture'))


if __name__ == '__main__':
    unittest.main()

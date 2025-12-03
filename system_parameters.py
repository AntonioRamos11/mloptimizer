import os
from dataclasses import dataclass
from typing import List

@dataclass
class SystemParameters:

    # ------------------------
    # CLOUD / LOCAL SELECTION
    # ------------------------
    # Set this to 1 in the cloud:
    #   export CLOUD_MODE=1
    # Set this to 0 for localhost:
    #   export CLOUD_MODE=0 (or unset - default)
    CLOUD_MODE: bool = bool(int(os.getenv("CLOUD_MODE", "0")))

    # ------------------------
    # LOCALHOST RABBITMQ
    # ------------------------
    LOCALHOST_HOST: str = "localhost"
    LOCALHOST_PORT: int = 5672
    LOCALHOST_MANAGEMENT_URL: str = "http://localhost:15672"
    
    # ------------------------
    # REMOTE RABBITMQ (ngrok / telebit)
    # ------------------------
    REMOTE_HOST_URL: str = "0.tcp.us-cal-1.ngrok.io"
    REMOTE_PORT: int = 19775
    REMOTE_MANAGEMENT_URL: str = "https://selfish-donkey-2.telebit.io"
    
    # ------------------------
    # ACTIVE CONFIGURATION (selected based on CLOUD_MODE)
    # ------------------------
    @classmethod
    def _get_active_host(cls) -> str:
        """Get active host based on CLOUD_MODE."""
        if os.getenv("INSTANCE_HOST_URL"):
            return os.getenv("INSTANCE_HOST_URL")
        return cls.REMOTE_HOST_URL if cls.CLOUD_MODE else cls.LOCALHOST_HOST
    
    @classmethod
    def _get_active_port(cls) -> int:
        """Get active port based on CLOUD_MODE."""
        if os.getenv("INSTANCE_PORT"):
            return int(os.getenv("INSTANCE_PORT"))
        return cls.REMOTE_PORT if cls.CLOUD_MODE else cls.LOCALHOST_PORT
    
    @classmethod
    def _get_active_management_url(cls) -> str:
        """Get active management URL based on CLOUD_MODE."""
        if os.getenv("INSTANCE_MANAGMENT_URL"):
            return os.getenv("INSTANCE_MANAGMENT_URL")
        return cls.REMOTE_MANAGEMENT_URL if cls.CLOUD_MODE else cls.LOCALHOST_MANAGEMENT_URL

    # Computed properties - use class methods to get current values
    @classmethod
    @property
    def INSTANCE_HOST_URL(cls) -> str:
        return cls._get_active_host()
    
    @classmethod
    @property
    def INSTANCE_PORT(cls) -> int:
        return cls._get_active_port()
    
    @classmethod
    @property
    def INSTANCE_MANAGMENT_URL(cls) -> str:
        return cls._get_active_management_url()

    INSTANCE_MODEL_PARAMETER_QUEUE: str = "parameters"
    INSTANCE_MODEL_PERFORMANCE_QUEUE: str = "results"
    INSTANCE_USER: str = os.getenv("RABBIT_USER", "guest")
    INSTANCE_PASSWORD: str = os.getenv("RABBIT_PASS", "guest")
    INSTANCE_VIRTUAL_HOST: str = "/"

    @classmethod
    def INSTANCE_CONNECTION(cls) -> List:
        """Get active connection parameters based on CLOUD_MODE."""
        return [
            cls._get_active_port(),
            cls.INSTANCE_MODEL_PARAMETER_QUEUE,
            cls.INSTANCE_MODEL_PERFORMANCE_QUEUE,
            cls._get_active_host(),
            cls.INSTANCE_USER,
            cls.INSTANCE_PASSWORD,
            cls.INSTANCE_VIRTUAL_HOST,
            cls._get_active_management_url(),
        ]

    DATASET_NAME: str = 'mnist'  # Example dataset

    # Define specific dataset configurations
    dataset_config = {
        'cifar10': {
            'shape': (32, 32, 3),
            'classes': 10   
        },
        'mnist': {
            'shape': (28, 28, 1),
            'classes': 10
        },
        'fashion_mnist': {
            'shape': (28, 28, 1),
            'classes': 10
        },
        'cifar100': {
            'shape': (32, 32, 3),
            'classes': 100
        },
        'grietas_baches': {
            'shape': (96, 96, 3),  # Reduced from 224x224 - saves MASSIVE memory!
            'classes': 2
        }
    }
    DATASET_SHAPE: tuple = (32, 32, 3)  # Adjust based on dataset shape
    
    DATASET_CLASSES: int = 10

    DATASET_SHAPE = dataset_config[DATASET_NAME]['shape']
    DATASET_CLASSES = dataset_config[DATASET_NAME]['classes']


    DATASET_SCHEDULE = [
        ('mnist', 3),
        ('cifar10', 3),
        ('fashion_mnist', 3),
    ]
    
    @classmethod
    def set_dataset(cls, name: str):
        """Switch active dataset and update dependent attributes."""
        if name not in cls.dataset_config:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(cls.dataset_config.keys())}")
        cls.DATASET_NAME = name
        cfg = cls.dataset_config[name]
        cls.DATASET_SHAPE = cfg['shape']
        cls.DATASET_CLASSES = cfg['classes']

    @classmethod
    def expand_schedule(cls):
        """Return flat list of dataset names expanded by repeat counts."""
        expanded = []
        for name, reps in cls.DATASET_SCHEDULE:
            if name not in cls.dataset_config:
                raise ValueError(f"Unknown dataset '{name}' in DATASET_SCHEDULE")
            expanded.extend([name]*reps)
        return expanded

    @classmethod
    def iter_datasets(cls):
        """Yield (index, dataset_name, shape, classes) for each scheduled occurrence."""
        for idx, name in enumerate(cls.expand_schedule()):
            cls.set_dataset(name)
            yield idx, name, cls.DATASET_SHAPE, cls.DATASET_CLASSES

    @classmethod
    def total_runs(cls) -> int:
        return sum(reps for _, reps in cls.DATASET_SCHEDULE)
    
    @classmethod
    def get_connection_info(cls) -> dict:
        """Get current connection configuration as a dictionary."""
        return {
            'mode': 'CLOUD' if cls.CLOUD_MODE else 'LOCALHOST',
            'host': cls._get_active_host(),
            'port': cls._get_active_port(),
            'management_url': cls._get_active_management_url(),
            'user': cls.INSTANCE_USER,
            'virtual_host': cls.INSTANCE_VIRTUAL_HOST
        }
    
    @classmethod
    def print_connection_info(cls):
        """Print current connection configuration."""
        info = cls.get_connection_info()
        print("=" * 60)
        print(f"  RabbitMQ Connection Mode: {info['mode']}")
        print("=" * 60)
        print(f"  Host: {info['host']}")
        print(f"  Port: {info['port']}")
        print(f"  Management URL: {info['management_url']}")
        print(f"  User: {info['user']}")
        print(f"  Virtual Host: {info['virtual_host']}")
        print("=" * 60)

    # Dataset parameters
    DATASET_TYPE: int = 1  # Image classification
    DATASET_BATCH_SIZE: int = 4  # Reduced from 8 - critical for memory!
    DATASET_VALIDATION_SPLIT: float = 0.2
    DATASET_FEATURES: int = 32  # Update for regression datasets
    DATASET_LABELS: int = 10
    DATASET_WINDOW_SIZE: int = 100
    DATASET_DATA_SIZE: int = 1
    
    # Custom dataset path (for grietas_baches and other local datasets)
    DATASET_PATH: str = './dataset_grietas_baches'

    # AutoML parameters
    TRAIN_GPU: bool = True
    TRIALS = 20
    EXPLORATION_SIZE: int = 10
    EXPLORATION_EPOCHS: int = 5  # Reduced from 10 - faster, less memory buildup
    EXPLORATION_EARLY_STOPPING_PATIENCE: int = 3
    HALL_OF_FAME_SIZE: int = 5
    HALL_OF_FAME_EPOCHS: int = 30  # Reduced from 150 - prevent long runs
    HOF_EARLY_STOPPING_PATIENCE: int = 5  # Reduced from 10

    # Model parameters
    DTYPE: str = 'float32'
    OPTIMIZER: str = 'adam'
    LAYERS_ACTIVATION_FUNCTION: str = 'relu'
    OUTPUT_ACTIVATION_FUNCTION: str = 'softmax'  # For classification
    KERNEL_INITIALIZER: str = 'he_uniform'
    LOSS_FUNCTION: str = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']
    PADDING: str = 'same'
    WEIGHT_DECAY = 1e-4

    # Time-series models
    LSTM_ACTIVATION_FUNCTION: str = 'tanh'
import os
from dataclasses import dataclass

@dataclass
class SystemParameters:

    # ------------------------
    # CLOUD / LOCAL SELECTION
    # ------------------------
    # Set this to 1 in the cloud:
    #   export CLOUD_MODE=1
    CLOUD_MODE: bool = bool(int(os.getenv("CLOUD_MODE", "0")))

    # ------------------------
    # REMOTE RABBITMQ (ngrok / telebit) - DEFAULT
    # ------------------------
    INSTANCE_HOST_URL: str = os.getenv("INSTANCE_HOST_URL", "0.tcp.us-cal-1.ngrok.io")
    INSTANCE_PORT: int = int(os.getenv("INSTANCE_PORT", "19775"))
    INSTANCE_MANAGMENT_URL: str = os.getenv("INSTANCE_MANAGMENT_URL", "https://selfish-donkey-2.telebit.io")

    INSTANCE_MODEL_PARAMETER_QUEUE: str = "parameters"
    INSTANCE_MODEL_PERFORMANCE_QUEUE: str = "results"
    INSTANCE_USER: str = os.getenv("RABBIT_USER", "guest")
    INSTANCE_PASSWORD: str = os.getenv("RABBIT_PASS", "guest")
    INSTANCE_VIRTUAL_HOST: str = "/"

    INSTANCE_CONNECTION = [
        INSTANCE_PORT,
        INSTANCE_MODEL_PARAMETER_QUEUE,
        INSTANCE_MODEL_PERFORMANCE_QUEUE,
        INSTANCE_HOST_URL,
        INSTANCE_USER,
        INSTANCE_PASSWORD,
        INSTANCE_VIRTUAL_HOST,
        INSTANCE_MANAGMENT_URL,
    ]
    #DATASET_NAME: str = 'mnist'  # Example datasetr
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

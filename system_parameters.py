class SystemParameters:
    # Rabbit MQ Connections
    #INSTANCE_PORT: int = 17121  # Port for RabbitMQ
    #INSTANCE_MANAGMENT_URL = "192.168.100.89"
    INSTANCE_PORT: int = 5672  # Port for RabbitMQ
    INSTANCE_MANAGMENT_URL = "localhost"
    INSTANCE_HOST_URL: str = 'localhost'
    INSTANCE_MODEL_PARAMETER_QUEUE: str = 'parameters'
    INSTANCE_MODEL_PERFORMANCE_QUEUE: str = 'results'
    #if not local
    INSTANCE_HOST_URL: str = '4.tcp.us-cal-1.ngrok.io'
    INSTANCE_MANAGMENT_URL = "https://quiet-husky-46.telebit.io"

    #INSTANCE_HOST_URL: str = 'serveo.net'
    INSTANCE_USER: str = 'guest'  # Default RabbitMQ user
    INSTANCE_PASSWORD: str = 'guest'  # Default password
    INSTANCE_VIRTUAL_HOST: str = '/'

    INSTANCE_CONNECTION = [
        INSTANCE_PORT, INSTANCE_MODEL_PARAMETER_QUEUE,
        INSTANCE_MODEL_PERFORMANCE_QUEUE, INSTANCE_HOST_URL,
        INSTANCE_USER, INSTANCE_PASSWORD, INSTANCE_VIRTUAL_HOST,INSTANCE_MANAGMENT_URL
    ]
    
    DATASET_NAME: str = 'mnist'  # Example datasetr
    #DATASET_NAME: str = 'cifar10'  # Example dataset

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
        }
    }
    DATASET_SHAPE: tuple = (32, 32, 3)  # Adjust based on dataset shape
    
    DATASET_CLASSES: int = 10

    DATASET_SHAPE = dataset_config[DATASET_NAME]['shape']
    DATASET_CLASSES = dataset_config[DATASET_NAME]['classes']


    # Dataset parameters
    DATASET_TYPE: int = 1  # Image classification
    DATASET_BATCH_SIZE: int = 32
    DATASET_VALIDATION_SPLIT: float = 0.2
    DATASET_FEATURES: int = 32  # Update for regression datasets
    DATASET_LABELS: int = 10
    DATASET_WINDOW_SIZE: int = 100
    DATASET_DATA_SIZE: int = 1

    

    # AutoML parameters
    TRAIN_GPU: bool = True
    TRIALS = 20
    EXPLORATION_SIZE: int = 10
    EXPLORATION_EPOCHS: int = 10
    EXPLORATION_EARLY_STOPPING_PATIENCE: int = 3
    HALL_OF_FAME_SIZE: int = 5
    HALL_OF_FAME_EPOCHS: int = 150
    HOF_EARLY_STOPPING_PATIENCE: int = 10

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

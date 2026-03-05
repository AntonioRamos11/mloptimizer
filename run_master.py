import os
# Force single GPU visibility BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

import logging
import sys

# Configure root logger FIRST before any imports
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

# Create handler that writes to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

# Suppress specific noisy loggers
for logger_name in ['aiormq', 'pika', 'aio_pika', 'asyncio', 'tensorflow', 'matplotlib', 
                    'urllib3', 'botocore', 'boto3', 's3transfer', 'google',
                    'optuna', 'aiohttp']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Set app loggers to INFO
for app_logger in ['app.init_nodes', 'app.master_node', 'app.common']:
    logging.getLogger(app_logger).setLevel(logging.INFO)

from app.init_nodes import *
from app.common.dataset import *
from app.common.search_space import *

InitNodes().master()
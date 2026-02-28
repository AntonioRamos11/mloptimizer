import os
# Force single GPU visibility BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

for logger_name in ['aiormq', 'pika', 'aio_pika', 'asyncio', 'tensorflow', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from app.init_nodes import *
from app.common.dataset import *
from app.common.search_space import *

InitNodes().master()   
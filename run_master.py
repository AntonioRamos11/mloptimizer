import os
# Force single GPU visibility BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

from app.init_nodes import *
from app.common.dataset import *
from app.common.search_space import *

InitNodes().master()   
from app.init_nodes import *
from app.common.dataset import *
from app.common.search_space import *
from clear_queues import main as clear_queues


clear_queues()
InitNodes().master()    
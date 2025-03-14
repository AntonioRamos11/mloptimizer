import asyncio
import time
import json
import os
import GPUtil
from dataclasses import asdict
import aio_pika
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.master_node.communication.master_rabbitmq_client import *
from app.master_node.communication.rabbitmq_monitor import *
from app.master_node.optimization_strategy import OptimizationStrategy, Action, Phase
from app.common.dataset import * 
from system_parameters import SystemParameters as SP
from app.common.socketCommunication import *
import platform
import subprocess
import os
import json

#add syspath of utils

def get_hardware_info():
    """
    Gather hardware information about the system
    
    Returns:
        dict: Dictionary containing hardware information
    """
    hardware_info = {}
    
    # CPU information
    hardware_info["cpu_model"] = platform.processor()
    hardware_info["cpu_cores"] = os.cpu_count()
    hardware_info["python_version"] = platform.python_version()
    hardware_info["system"] = platform.system()
    
    # RAM information
    try:
        import psutil
        ram = psutil.virtual_memory()
        hardware_info["ram_total"] = f"{ram.total / (1024**3):.2f} GB"
        hardware_info["ram_available"] = f"{ram.available / (1024**3):.2f} GB"
    except ImportError:
        hardware_info["ram_total"] = "Unknown (psutil not installed)"
    
    # GPU information
    try:
        # Try to get GPU info using NVIDIA tools
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader"], 
            universal_newlines=True
        )
        gpus = []
        for line in nvidia_output.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) >= 2:
                name, memory = parts[0], parts[1]
                driver = parts[2] if len(parts) > 2 else "Unknown"
                gpus.append({"model": name, "memory": memory, "driver": driver})
        
        hardware_info["gpu_count"] = len(gpus)
        hardware_info["gpus"] = gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If nvidia-smi isn't available or fails, try TensorFlow
        try:
            import tensorflow as tf
            physical_gpus = tf.config.list_physical_devices('GPU')
            hardware_info["gpu_count"] = len(physical_gpus)
            hardware_info["gpus"] = [{"device": str(gpu)} for gpu in physical_gpus]
            
            # Try to get more GPU details
            if len(physical_gpus) > 0:
                for i, gpu in enumerate(physical_gpus):
                    with tf.device(f'/GPU:{i}'):
                        gpu_name = tf.test.gpu_device_name()
                        if i < len(hardware_info["gpus"]):
                            hardware_info["gpus"][i]["name"] = gpu_name
        except:
            hardware_info["gpu_count"] = 0
            hardware_info["gpus"] = []
    
    return hardware_info
class OptimizationJob:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory):
		asyncio.set_event_loop(asyncio.new_event_loop())
		self.loop = asyncio.get_event_loop()
		self.dataset = dataset
		self.search_space: ModelArchitectureFactory = model_architecture_factory
		self.optimization_strategy = OptimizationStrategy(self.search_space, self.dataset, SP.EXPLORATION_SIZE, SP.HALL_OF_FAME_SIZE)
		#Creates a connection with a connection types as a parameter
		rabbit_connection_params = RabbitConnectionParams.new()
		self.rabbitmq_client = MasterRabbitMQClient(rabbit_connection_params, self.loop)
		self.rabbitmq_monitor = RabbitMQMonitor(rabbit_connection_params)

	def start_optimization(self, trials: int):
		self.start_time = time.time()
		self.loop.run_until_complete(self._run_optimization_startup())
		connection = self.loop.run_until_complete(self._run_optimization_loop(trials))
		try:
			self.loop.run_forever()
		finally:
			self.loop.run_until_complete(connection.close())

	async def _run_optimization_startup(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': '*** Running optimization startup ***'})
		await self.rabbitmq_client.prepare_queues()
		queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
		for i in range (0, queue_status.consumer_count + 1):
			await self.generate_model()

	async def _run_optimization_loop(self, trials: int) -> aio_pika.Connection:
		connection = await self.rabbitmq_client.listen_for_model_results(self.on_model_results)
		return connection

	async def on_model_results(self, response: dict):
		model_training_response = ModelTrainingResponse.from_dict(response)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received response'})
		cad = str(model_training_response.id) + '|' + str(model_training_response.performance) + '|' + str(model_training_response.finished_epochs)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		action: Action = self.optimization_strategy.report_model_response(model_training_response)
		SocketCommunication.decide_print_form(MSGType.FINISHED_MODEL, {'node': 1, 'msg': 'Finished a model', 'total': self.optimization_strategy.get_training_total()})
		print('Action:', action)
		if action == Action.GENERATE_MODEL:
			await self.generate_model()
		elif action == Action.WAIT:
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Wait for models'})
		elif action == Action.START_NEW_PHASE:
			queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
			SocketCommunication.decide_print_form(MSGType.CHANGE_PHASE, {'node': 1, 'msg': 'New phase, deep training'})
			for i in range(0, queue_status.consumer_count + 1):
				await self.generate_model()
		elif action == Action.FINISH:
			SocketCommunication.decide_print_form(MSGType.FINISHED_TRAINING, {'node': 1, 'msg': 'Finished training'})
			best_model = self.optimization_strategy.get_best_model()
			await self._log_results(best_model)
			model = Model(best_model.model_training_request, self.dataset)
			model.is_model_valid()
			self.loop.stop()

	async def generate_model(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Generating new model'})
		model_training_request: ModelTrainingRequest = self.optimization_strategy.recommend_model()
		model = Model(model_training_request, self.dataset)
		if not model.is_model_valid():
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Model is not valid'})
		else:
			await self._send_model_to_broker(model_training_request)
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Sent model to broker'})

	async def _send_model_to_broker(self, model_training_request: ModelTrainingRequest):
		model_training_request_dict = asdict(model_training_request)
		print("Model training request", model_training_request_dict)
		await self.rabbitmq_client.publish_model_params(model_training_request_dict)

	async def _log_results(self, best_model):
			# Simple approach using relative paths
			# This assumes the code is being run from the mloptimizer directory
			results_path = 'results'  # Relative path to mloptimizer/results
			
			# Create results directory if it doesn't exist
			os.makedirs(results_path, exist_ok=True)
			print(f"Saving results to: {results_path}")

			filename = best_model.model_training_request.experiment_id
			filename = os.path.join(results_path, filename)
			
			# Create a structured result object
			result_data = {
				"model_info": asdict(best_model),
				"dataset_ranges": None,
				"performance_metrics": {
					"elapsed_seconds": time.time() - self.start_time
				},
				"hardware_info": {}
			}
			
			print('Finished optimization')
			print('Best model: ')
			print(json.dumps(result_data["model_info"], indent=2))

			# Get dataset ranges
			try:
				self.dataset.load()
				ranges = self.dataset.get_ranges()
				result_data["dataset_ranges"] = ranges
				print('Information ranges from normalization')
				print(ranges)
			except Exception as e:
				print(f"Error getting dataset ranges: {e}")

			# Format elapsed time
			elapsed_seconds = result_data["performance_metrics"]["elapsed_seconds"]
			elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))
			result_data["performance_metrics"]["elapsed_time"] = elapsed_time
			
			time_text = f"Optimization took: {elapsed_time} (hh:mm:ss) {elapsed_seconds:.2f} (Seconds)"
			print(time_text)

			# Add hardware information
			try:
				info = get_hardware_info()
				result_data["hardware_info"] = info
			except Exception as e:
				print(f"Error collecting hardware information: {e}")

			# Save all results as a properly formatted JSON file
			with open(filename + ".json", "w") as f:
				json.dump(result_data, f, indent=2)

			print("\n ********************************************** \n")
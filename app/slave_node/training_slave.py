import asyncio
import logging
import time

import concurrent

import aio_pika
import json
from dataclasses import asdict
# import aio_pikas import asdict
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.slave_node.slave_rabbitmq_client import SlaveRabbitMQClient
from app.common.dataset import *
from system_parameters import SystemParameters as SP

from app.common.dataset_switcher import DatasetSwitchCallback

class TrainingSlave:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory, datasets_list=None):
		#asyncio.set_event_loop(asyncio.new_event_loop())
		#self.loop = asyncio.get_event_loop()
		self.loop = asyncio.get_event_loop()
		self.dataset = dataset
		rabbit_connection_params = RabbitConnectionParams.new()
		self.rabbitmq_client = SlaveRabbitMQClient(rabbit_connection_params, self.loop)
		model_architecture_factory = model_architecture_factory
		self.search_space_hash = model_architecture_factory.get_search_space().get_hash()
		cad = 'Hash ' + str(self.search_space_hash) 
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		self.model_type = model_architecture_factory.get_search_space().get_type()
		self.datasets_list = datasets_list if datasets_list else [dataset]

	def start_slave(self):
		#loop = asyncio.get_event_loop()
		#loop.run_until_complete(asyncio.wait(futures))
		#connection = asyncio.run(self._start_listening())
		#asyncio.new_event_loop(self._start_listening())
		#await connection = self._start_listening()
		connection = self.loop.run_until_complete(self._start_listening())
		print("Stop listening")
		try:
			self.loop.run_forever()
		finally:
			self.loop.run_until_complete(connection.close())

	@staticmethod
	def fake_blocking_training():
		# Method for testing broker connection timeout
		for i in range(0, 240):
			time.sleep(1)
			print(i)
		return 0.5

	@staticmethod
	def train_model(info_dict: dict) -> float:
		dataset = info_dict['dataset']
		model_training_request = info_dict['model_request']
		
		# Check if we have multiple datasets defined
		if hasattr(SP, 'DATASET_NAMES') and len(SP.DATASET_NAMES) > 1:
			# Load multiple datasets
			datasets_list = []
			
			# Log all datasets' shapes to help debug
			print(f"Training using {len(SP.DATASET_NAMES)} datasets:")
			for i, dataset_name in enumerate(SP.DATASET_NAMES):
				print(f"  Dataset {i+1}: {dataset_name}")
				
				# Create dataset
				if SP.DATASET_TYPE == 1:
					current_dataset = ImageClassificationBenchmarkDataset(
						dataset_name, 
						SP.dataset_config[dataset_name]['shape'],  # Use individual shape
						SP.dataset_config[dataset_name]['classes'],
						SP.DATASET_BATCH_SIZE, 
						SP.DATASET_VALIDATION_SPLIT
					)
				elif SP.DATASET_TYPE == 2:
					current_dataset = RegressionBenchmarkDataset(
						dataset_name, SP.DATASET_SHAPE, SP.DATASET_FEATURES,
						SP.DATASET_LABELS, SP.DATASET_BATCH_SIZE, SP.DATASET_VALIDATION_SPLIT
					)
				elif SP.DATASET_TYPE == 3:
					current_dataset = TimeSeriesBenchmarkDataset(
						dataset_name, SP.DATASET_WINDOW_SIZE, SP.DATASET_DATA_SIZE,
						SP.DATASET_BATCH_SIZE, SP.DATASET_VALIDATION_SPLIT
					)
				datasets_list.append(current_dataset)
			
			# Build model for multiple datasets
			model = Model(model_training_request, dataset)
			if SP.TRAIN_GPU:
				return model.build_and_train_multiple_datasets(datasets_list)
			else:
				# Implement CPU version if needed
				return model.build_and_train_multiple_datasets(datasets_list)
		else:
			# Original single dataset training
			dataset.load()
			model = Model(model_training_request, dataset)
			if SP.TRAIN_GPU:
				return model.build_and_train()
			else:
				return model.build_and_train_cpu()

	async def _start_listening(self) -> aio_pika.Connection:
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Worker started!"})
		return await self.rabbitmq_client.listen_for_model_params(self._on_model_params_received)

	async def _on_model_params_received(self, body_dict: dict):
		# Process incoming model parameters
		try:
			# Check if architecture is in the body_dict
			if 'architecture' in body_dict:
				# Extract architecture from the dictionary
				model_arch = body_dict['architecture']
				model_training_request = ModelTrainingRequest.from_dict(body_dict, model_arch)
			else:
				# If architecture is not present, handle the error
				SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, 
					{'node': 2, 'msg': 'Missing architecture in model request'})
				return
				
			# Display hardware information if present
			if 'hardware_info' in body_dict:
				hardware_info = body_dict['hardware_info']
				# Format hardware info nicely for display
				hw_info_str = json.dumps(hardware_info, indent=2)
				print(f"Master node hardware configuration:\n{hw_info_str}")
				
				# Log the GPU model and count specifically
				if 'gpu_count' in hardware_info and hardware_info['gpu_count'] > 0:
					gpu_info = f"Using {hardware_info['gpu_count']} GPUs: "
					gpu_info += ", ".join([gpu.get('model', 'Unknown') for gpu in hardware_info.get('gpus', [])])
					SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, 
						{'node': 2, 'msg': gpu_info})

			# Display dataset repetition information
			repetitions = getattr(SP, 'DATASET_REPETITIONS', 1)
			datasets = body_dict.get('dataset_names', [body_dict.get('dataset_tag', SP.DATASET_NAME)])
			dataset_msg = f"Training on {len(datasets)} datasets with {repetitions} repetitions each"
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, 
				{'node': 2, 'msg': dataset_msg})
			
			# Add local hardware info to the response
			from app.common.utils import get_hardware_info
			local_hardware = get_hardware_info()
			
			# Check if this request includes multiple datasets
			use_multiple_datasets = body_dict.get('use_multiple_datasets', False)
			
			# Create info dictionary with relevant data
			info_dict = {
				'model_request': model_training_request,
				'dataset': self.dataset
			}
			
			# Create a thread pool executor
			with concurrent.futures.ThreadPoolExecutor() as pool:
				# If using multiple datasets, use datasets_list
				if use_multiple_datasets and self.datasets_list and len(self.datasets_list) > 1:
					# Use the standard train_model method as it already supports multiple datasets
					training_val, did_finish_epochs = await self.loop.run_in_executor(
						pool, self.train_model, info_dict
					)
				else:
					# Call regular training with single dataset
					training_val, did_finish_epochs = await self.loop.run_in_executor(
						pool, self.train_model, info_dict
					)
				
			# Create and send response
			model_training_response = ModelTrainingResponse(
				id=model_training_request.id, 
				performance=training_val, 
				finished_epochs=did_finish_epochs
			)
			await self._send_performance_to_broker(model_training_response)
		
		except Exception as e:
			SocketCommunication.decide_print_form(MSGType.SLAVE_ERROR, 
				{'node': 2, 'msg': f'Error processing model request: {str(e)}'})
			import traceback
			print(traceback.format_exc())

	async def _send_performance_to_broker(self, model_training_response: ModelTrainingResponse):
		print(model_training_response)
		model_training_response_dict = asdict(model_training_response)
		print(model_training_response_dict)
		await self.rabbitmq_client.publish_model_performance(model_training_response_dict)


def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    logging.error(f"Caught exception: {msg}")
    logging.error(context["exception"])
    logging.info("Shutting down...")
    loop.stop()

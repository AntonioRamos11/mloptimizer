import asyncio
import time
import concurrent
import logging
from dataclasses import asdict
import aio_pika
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.slave_node.slave_rabbitmq_client import SlaveRabbitMQClient
from app.common.dataset import *
from system_parameters import SystemParameters as SP

class TrainingSlave:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory):
		#asyncio.set_event_loop(asyncio.new_event_loop())
		#self.loop = asyncio.get_event_loop()
		self.loop = asyncio.loop.run_until_complete(self.loop)
		self.dataset = dataset
		rabbit_connection_params = RabbitConnectionParams.new()
		self.rabbitmq_client = SlaveRabbitMQClient(rabbit_connection_params, self.loop)
		model_architecture_factory = model_architecture_factory
		self.search_space_hash = model_architecture_factory.get_search_space().get_hash()
		cad = 'Hash ' + str(self.search_space_hash) 
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		self.model_type = model_architecture_factory.get_search_space().get_type()

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
		"""if SP.DATASET_TYPE == 1:
			dataset = ImageClassificationBenchmarkDataset(SP.DATASET_NAME, SP.DATASET_SHAPE, SP.DATASET_CLASSES, SP.DATASET_BATCH_SIZE, SP.DATASET_VALIDATION_SPLIT)
		elif SP.DATASET_TYPE == 2:
			dataset = RegressionBenchmarkDataset(SP.DATASET_NAME, SP.DATASET_SHAPE, SP.DATASET_FEATURES, SP.DATASET_LABELS, SP.DATASET_BATCH_SIZE, SP.DATASET_VALIDATION_SPLIT)
		elif SP.DATASET_TYPE == 3:
			dataset = TimeSeriesBenchmarkDataset(SP.DATASET_NAME, SP.DATASET_WINDOW_SIZE, SP.DATASET_DATA_SIZE, SP.DATASET_BATCH_SIZE, SP.DATASET_VALIDATION_SPLIT)
		else:
			print("Please enter a valid dataset type")
			return"""
		dataset = info_dict['dataset']
		model_training_request = info_dict['model_request']
		dataset.load()
		model = Model(model_training_request, dataset)
		if SP.TRAIN_GPU:
			return model.build_and_train()
		else:
			return model.build_and_train_cpu()

	async def _start_listening(self) -> aio_pika.Connection:
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Worker started!"})
		return await self.rabbitmq_client.listen_for_model_params(self._on_model_params_received)

	async def _on_model_params_received(self, model_params):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Received model training request"})
		#print(model_params)
		self.model_type = int(model_params['training_type'])
		model_training_request = ModelTrainingRequest.from_dict(model_params, self.model_type)
		if not self.search_space_hash == model_training_request.search_space_hash:
			raise Exception("Search space of master is different to this worker's search space")
		info_dict = {
			'dataset': self.dataset,
			'model_request': model_training_request
		}
		with concurrent.futures.ProcessPoolExecutor() as pool:
			training_val, did_finish_epochs = await self.loop.run_in_executor(pool, self.train_model, info_dict)
		#training_val, did_finish_epochs = self.train_model(model_training_request)
		model_training_response = ModelTrainingResponse(id=model_training_request.id, performance=training_val, finished_epochs=did_finish_epochs)
		await self._send_performance_to_broker(model_training_response)

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

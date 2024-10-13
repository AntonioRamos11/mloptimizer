import asyncio
import time
import json
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
		filename = best_model.model_training_request.experiment_id
		#specifpath 
		filename = "/home/p0wden/Documents/IA/MLOptimizer/results/" + filename
		f = open(filename, "a")
		model_info_json = json.dumps(asdict(best_model))
		f.write(model_info_json)
		f.close()

		print('Finished optimization')
		print('Best model: ')
		print(model_info_json)

		self.dataset.load()
		ranges = self.dataset.get_ranges()
		print('Information ranges from normalization')
		print(ranges)

		f = open(filename, "a")
		ranges_json = json.dumps(ranges)
		f.write(ranges_json)
		f.close()

		elapsed_seconds = time.time() - self.start_time
		elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

		time_text = "\n Optimization took: " + str(elapsed_time) + " (hh:mm:ss) " + str(elapsed_seconds) + " (Seconds) "
		print(time_text)

		f = open(filename, "a")
		f.write(time_text)
		f.close()

		print("\n ********************************************** \n")
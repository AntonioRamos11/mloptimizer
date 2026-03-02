import aio_pika
import asyncio
import json

from app.common.base_rabbitmq_client import BaseRabbitMQClient

class MasterRabbitMQClient(BaseRabbitMQClient):
	
	def __init__(self, params, loop):
		super().__init__(params, loop)
		self._publisher_connection = None

	async def _get_publisher_connection(self) -> aio_pika.RobustConnection:
		"""Reuse a single persistent connection for all publishes."""
		if self._publisher_connection is None or self._publisher_connection.is_closed:
			self._publisher_connection = await self._create_connection()
		return self._publisher_connection

	async def publish_model_params(self, model_params: dict) -> aio_pika.Connection:
		connection = await self._get_publisher_connection()
		message_body_json = json.dumps(model_params).encode()
		await self._run_publish(connection, self.model_parameter_queue, message_body_json)
		return connection

	async def listen_for_model_results(self, callback) -> aio_pika.Connection:
		return await super().listen(self.model_performance_queue, callback, auto_close_connection=False)

	async def purge_queues(self):
		"""Purge all queues to clean stale messages from previous runs"""
		print("[MasterRabbitMQClient] Purging RabbitMQ queues...")
		try:
			connection = await asyncio.wait_for(
				aio_pika.connect_robust(
					f"amqp://{self.user}:{self.password}@{self.host_url}/{self.virtual_host}"
				),
				timeout=10.0
			)
			channel = await connection.channel()
			
			# Purge parameters queue
			try:
				await channel.queue_delete(self.model_parameter_queue)
				await channel.declare_queue(self.model_parameter_queue, durable=True)
				print(f"[MasterRabbitMQClient] Purged: {self.model_parameter_queue}")
			except Exception as e:
				print(f"[MasterRabbitMQClient] Skip {self.model_parameter_queue}: {e}")
			
			# Purge results queue
			try:
				await channel.queue_delete(self.model_performance_queue)
				await channel.declare_queue(self.model_performance_queue, durable=True)
				print(f"[MasterRabbitMQClient] Purged: {self.model_performance_queue}")
			except Exception as e:
				print(f"[MasterRabbitMQClient] Skip {self.model_performance_queue}: {e}")
			
			await connection.close()
			print("[MasterRabbitMQClient] Queue purge done")
		except asyncio.TimeoutError:
			print("[MasterRabbitMQClient] Queue purge TIMEOUT - continuing without purge")
		except Exception as e:
			print(f"[MasterRabbitMQClient] Queue purge ERROR: {e} - continuing without purge")

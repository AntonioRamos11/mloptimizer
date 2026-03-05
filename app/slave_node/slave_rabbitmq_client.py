import aio_pika
import json
from app.common.base_rabbitmq_client import BaseRabbitMQClient

class SlaveRabbitMQClient(BaseRabbitMQClient):
	def __init__(self, params, loop):
		super().__init__(params, loop)
		# Persistent publisher connection — reused for all result publishes
		# to avoid leaking robust connections that cause duplicate messages
		self._publisher_connection = None

	async def _get_publisher_connection(self) -> aio_pika.RobustConnection:
		"""Reuse a single connection for all result publishes."""
		if self._publisher_connection is None or self._publisher_connection.is_closed:
			self._publisher_connection = await self._create_connection()
		return self._publisher_connection

	async def publish_model_performance(self, model_params: dict):
		"""Publish training results using a persistent connection (no leak)."""
		connection = await self._get_publisher_connection()
		message_body_json = json.dumps(model_params).encode()
		await self._run_publish(connection, self.model_performance_queue, message_body_json)
		print('[X] Sent model performance')
		print(model_params)

	async def listen_for_model_params(self, callback) -> aio_pika.Connection:
		return await super().listen(self.model_parameter_queue, callback, auto_close_connection=False)
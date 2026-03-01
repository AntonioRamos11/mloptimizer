import aio_pika

from app.common.base_rabbitmq_client import BaseRabbitMQClient

class MasterRabbitMQClient(BaseRabbitMQClient):
	
	async def publish_model_params(self, model_params: dict) -> aio_pika.Connection:
		return await super().publish(self.model_parameter_queue, model_params, auto_close_connection=False)

	async def listen_for_model_results(self, callback) -> aio_pika.Connection:
		return await super().listen(self.model_performance_queue, callback, auto_close_connection=False)

	async def purge_queues(self):
		"""Purge all queues to clean stale messages from previous runs"""
		print("[MasterRabbitMQClient] Purging all RabbitMQ queues to remove stale messages...")
		try:
			# Create connection using parent's connection logic
			connection = await aio_pika.connect_robust(
				f"amqp://{self.user}:{self.password}@{self.host_url}/{self.virtual_host}"
			)
			channel = await connection.channel()
			
			# Purge parameters queue
			try:
				await channel.queue_delete(self.model_parameter_queue)
				await channel.declare_queue(self.model_parameter_queue, durable=True)
				print(f"[MasterRabbitMQClient] Purged queue: {self.model_parameter_queue}")
			except Exception as e:
				print(f"[MasterRabbitMQClient] Could not purge {self.model_parameter_queue}: {e}")
			
			# Purge results queue
			try:
				await channel.queue_delete(self.model_performance_queue)
				await channel.declare_queue(self.model_performance_queue, durable=True)
				print(f"[MasterRabbitMQClient] Purged queue: {self.model_performance_queue}")
			except Exception as e:
				print(f"[MasterRabbitMQClient] Could not purge {self.model_performance_queue}: {e}")
			
			await connection.close()
			print("[MasterRabbitMQClient] Queue purge completed")
		except Exception as e:
			print(f"[MasterRabbitMQClient] Error purging queues: {e}")

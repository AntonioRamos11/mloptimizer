import asyncio
import json
from dataclasses import astuple
import aio_pika
from aio_pika import IncomingMessage
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.socketCommunication import *
import logging
import socket
import traceback

#Base class for RabbitMQ opertaions

class BaseRabbitMQClient:

	#Class constructor.
	def __init__(self, params: RabbitConnectionParams, loop: asyncio.AbstractEventLoop):
		#All the information is stored in variables 
		#throught the transformation of the params to tuples.
		(
			self.port,
			self.model_parameter_queue,
			self.model_performance_queue,
			self.host_url,
			self.user,
			self.password,
			self.virtual_host,
			self.management_url,
		) = astuple(params)
		self.loop = loop

	#Async function that prepares the queue connection
	async def prepare_queues(self):
		logging.basicConfig(level=logging.DEBUG)
		logger = logging.getLogger("rabbitmq_connection")
		
		logger.info(f"=== CONNECTION PARAMETERS ===")
		logger.info(f"Host: {self.host_url}")
		logger.info(f"Port: {self.port}")
		logger.info(f"User: {self.user}")
		logger.info(f"Password: {'*' * len(self.password) if self.password else 'None'}")
		logger.info(f"VHost: {getattr(self, 'virtual_host', '/')}")
		
		# Validate host resolution
		try:
			logger.info(f"Resolving host: {self.host_url}")
			host_info = socket.getaddrinfo(self.host_url, self.port)
			logger.info(f"Host resolved: {host_info[0][4]}")
		except Exception as e:
			logger.error(f"Host resolution failed: {e}")
		
		# Test port connectivity
		try:
			logger.info(f"Testing TCP connection to {self.host_url}:{self.port}")
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.settimeout(5)
			s.connect((self.host_url, self.port))
			s.close()
			logger.info("TCP connection successful")
		except Exception as e:
			logger.error(f"TCP connection failed: {e}")
		
		# Try the actual connection with timeout
		try:
			logger.info("Attempting RabbitMQ connection...")
			connection_task = asyncio.create_task(self._create_connection())
			connection = await asyncio.wait_for(connection_task, timeout=10.0)
			
			logger.info("Connection established successfully!")
			logger.info(f"Connection info: {connection}")
			
			# Rest of your queue preparation code...
			# ...

			return connection
			
		except asyncio.TimeoutError:
			logger.error("Connection timed out after 10 seconds")
			raise
			
		except Exception as e:
			logger.error(f"Connection failed: {str(e)}")
			logger.error(f"Exception type: {type(e).__name__}")
			logger.error(f"Traceback: {traceback.format_exc()}")
			raise

	async def publish(self, queue_name: str, message_body: dict, auto_close_connection=True)->aio_pika.Connection:
		connection = await self._create_connection()
		message_body_json = json.dumps(message_body).encode()

		if auto_close_connection:
			async with connection:
				await self._run_publish(connection, queue_name, message_body_json)
		else:
			await self._run_publish(connection, queue_name, message_body_json)

		return connection

	async def listen(self, queue_name: str, callback, auto_close_connection=True)->aio_pika.Connection:
		connection = await self._create_connection()

		async def on_result_recieved(message: IncomingMessage):
			body_json = message.body.decode()
			body_dict = json.loads(body_json)
			await callback(body_dict)
			await message.ack()

		if auto_close_connection:
			async with connection:
				await self._run_listener(connection, queue_name, on_result_recieved)
		else:
			await self._run_listener(connection, queue_name, on_result_recieved)

		return connection

	#Static method thar runs a publisher to publish messages in the queues.
	@staticmethod
	async def _run_publish(connection, queue_name, message_body_json):
		routing_key = queue_name
		channel = await connection.channel()
		await channel.default_exchange.publish(
			message = aio_pika.Message(
				body = message_body_json,
				content_type = 'application/json',
				content_encoding = 'utf-8',
			),
			routing_key = routing_key,
		)

	#Static method that runs a listener for the incoming messages.
	@staticmethod
	async def _run_listener(connection, queue_name, callback):
		routing_key = queue_name
		channel: aio_pika.Channel = await connection.channel()
		await channel.set_qos(prefetch_count=1)
		queue: aio_pika.Queue = await channel.declare_queue(routing_key, durable=True)
		await queue.consume(callback, no_ack=False)
	async def _create_connection(self) -> aio_pika.RobustConnection:
		print("The URL trying to connect:")
		print("amqp://{}:{}@{}:{}/".format(self.user, self.password, self.host_url, self.port))
		
		return await aio_pika.connect_robust(
			"amqp://{}:{}@{}:{}/".format(
				self.user, 
				self.password, 
				self.host_url,
				self.port
			), 
			loop=self.loop
		)
		
		# Alternative implementation if needed:
		"""
		return await aio_pika.connect(
			host=self.host_url,
			port=self.port,
			virtualhost=self.virtual_host,
			login=self.user,
			password=self.password,
			loop=self.loop,
		)
		"""
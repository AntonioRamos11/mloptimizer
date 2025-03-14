from aiohttp import BasicAuth, ClientSession
from dataclasses import dataclass

import aiohttp
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.socketCommunication import *
import json
@dataclass
class QueueStatus:
	queue_name: str
	consumer_count: int
	message_count: int



class RabbitMQMonitor(object):

	def __init__(self, params: RabbitConnectionParams):
		self.cp = params
		self.auth = BasicAuth(login=self.cp.user, password=self.cp.password)

	async def get_queue_status(self):
		if self.cp.host_url == 'localhost':
				url = 'http://localhost:15672/api/queues/%2F/parameters'
		#if self.cp.host_url.startswith('http://localhost:15672/api/queues/%2F/parameters'):
		if(self.cp.host_url.startswith('192')):
			url = "http://" +self.cp.host_url + ":15672/api/queues/%2F/parameters"
		else:
			url = self.cp.managment_url
			# Use cp.model_parameter_queue or other appropriate attribute
		queue_name = self.cp.model_parameter_queue
		auth = aiohttp.BasicAuth(self.cp.user, self.cp.password)
		async with aiohttp.ClientSession(auth=auth) as session:
			try:
				async with session.get(url) as resp:
					if resp.status == 404:
						print(f"Queue '{queue_name}' not found. It might not exist yet.")
						# Return default values using only the parameters your QueueStatus accepts
						return QueueStatus(
							queue_name=queue_name,
							consumer_count=0,
							message_count=0
						)
					
					resp.raise_for_status()
					
					body = await resp.json()
					print(f"Queue API response: {body}")
					
					# Get values with defaults if keys don't exist
					consumer_count = body.get('consumers', 0)
					message_count = body.get('messages', 0)
					
					return QueueStatus(
						queue_name=queue_name,
						consumer_count=consumer_count,
						message_count=message_count
					)
			except aiohttp.ClientResponseError as e:
				print(f"API error: {e}")
				return QueueStatus(
					queue_name=queue_name,
					consumer_count=0,
					message_count=0,
				)
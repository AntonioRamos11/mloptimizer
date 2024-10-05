from aiohttp import BasicAuth, ClientSession
from dataclasses import dataclass
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

	async def get_queue_status(self) -> QueueStatus:
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Requesting queue status...'})
		async with ClientSession(auth=self.auth) as session:
			if self.cp.host_url == 'localhost':
				url = 'http://localhost:15672/api/queues/%2F/parameters'
			else:
				#http://192.168.1.84:15672/api/queues/%2F/parameters
				url = "http://{}:{}/api/queues/%2F/{}".format(self.cp.host_url, self.cp.port, self.cp.model_parameter_queue)
			print(url)
			async with session.get(url) as resp:
				print(resp.status)
				body = await resp.json()
				print(json.dumps(body, indent=4))
				consumer_count = body['consumers']
				message_count = body['messages']
				queue_name = body['name']
				queue_status = QueueStatus(
					queue_name=queue_name,
					consumer_count=consumer_count,
					message_count=message_count
				)
				SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received queue status'})
				cad = queue_status.queue_name + '|' + str(queue_status.consumer_count) + '|' + str(queue_status.message_count) 
				SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
				return queue_status
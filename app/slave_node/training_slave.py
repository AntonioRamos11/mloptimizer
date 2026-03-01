import asyncio
import time
import concurrent
import logging
import traceback
from datetime import datetime
from dataclasses import asdict
import aio_pika
import platform
import subprocess
import os
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.slave_node.slave_rabbitmq_client import SlaveRabbitMQClient
from app.common.dataset import *
from system_parameters import SystemParameters as SP

# Configure module logger
logger = logging.getLogger(__name__)

def get_hardware_info():
	"""
	Gather hardware information about the system
	
	Returns:
		dict: Dictionary containing hardware information
	"""
	hardware_info = {}
	
	hardware_info["cpu_model"] = platform.processor()
	hardware_info["cpu_cores"] = os.cpu_count()
	hardware_info["python_version"] = platform.python_version()
	hardware_info["system"] = platform.system()
	
	try:
		import psutil
		ram = psutil.virtual_memory()
		hardware_info["ram_total"] = f"{ram.total / (1024**3):.2f} GB"
		hardware_info["ram_available"] = f"{ram.available / (1024**3):.2f} GB"
	except ImportError:
		hardware_info["ram_total"] = "Unknown (psutil not installed)"
	
	try:
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
		try:
			import tensorflow as tf
			physical_gpus = tf.config.list_physical_devices('GPU')
			hardware_info["gpu_count"] = len(physical_gpus)
			hardware_info["gpus"] = [{"device": str(gpu)} for gpu in physical_gpus]
			
			if len(physical_gpus) > 0:
				for i, gpu in enumerate(physical_gpus):
					with tf.device(f'/GPU:{i}'):
						gpu_name = tf.test.gpu_device_name()
						if i < len(hardware_info["gpus"]):
							hardware_info["gpus"][i]["name"] = gpu_name
		except Exception as e:
			logger.error(f"Error getting GPU details via TensorFlow: {e}")
			hardware_info["gpu_count"] = 0
			hardware_info["gpus"] = []
	
	return hardware_info

# Global dataset cache for subprocess - loaded once per subprocess
_subprocess_dataset_cache = None
_subprocess_dataset_id = None

class TrainingSlave:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory):
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
		
		# Create persistent ProcessPoolExecutor (reuses same subprocess)
		logger.info("Creating persistent worker subprocess...")
		self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
		
		# Deduplication: track processed jobs to prevent duplicate processing
		self.processed_jobs = set()
		
		# Pre-load dataset in the subprocess ONCE
		logger.info("Pre-loading dataset in worker subprocess...")
		future = self.pool.submit(self._initialize_dataset_cache, self.dataset)
		result = future.result()  # Wait for dataset to load
		logger.info(f"✓ {result}")
		logger.info("Worker subprocess ready! Dataset cached and will be reused for all trainings.")

	def start_slave(self):
		"""Start the slave node with error handling"""
		try:
			logger.info("Starting slave node event loop...")
			connection = self.loop.run_until_complete(self._start_listening())
			logger.info("Slave node listening for tasks")
			
			try:
				self.loop.run_forever()
			finally:
				logger.info("Closing connection...")
				self.loop.run_until_complete(connection.close())
				logger.info("Connection closed")
				
		except Exception as e:
			logger.error("Error in start_slave:")
			logger.error(f"  Type: {type(e).__name__}")
			logger.error(f"  Message: {str(e)}")
			logger.error(f"  Traceback:\n{traceback.format_exc()}")
			raise

	@staticmethod
	def fake_blocking_training():
		# Method for testing broker connection timeout
		for i in range(0, 240):
			time.sleep(1)
			print(i)
		return 0.5

	@staticmethod
	def _initialize_dataset_cache(dataset):
		"""Initialize the dataset cache in the subprocess (called once at startup)"""
		global _subprocess_dataset_cache, _subprocess_dataset_id
		
		import os
		import logging
		logging.basicConfig(level=logging.INFO)
		subprocess_logger = logging.getLogger('dataset_cache')
		
		pid = os.getpid()
		dataset_id = f"{dataset.get_tag()}_{dataset.batch_size}_{dataset.validation_split_float}"
		
		subprocess_logger.info(f"[PID {pid}] Initializing dataset cache in worker subprocess...")
		subprocess_logger.info(f"[PID {pid}] Loading dataset: {dataset.get_tag()}")
		
		dataset.load()
		
		_subprocess_dataset_cache = dataset
		_subprocess_dataset_id = dataset_id
		
		subprocess_logger.info(f"[PID {pid}] ✓ Dataset loaded and cached!")
		subprocess_logger.info(f"[PID {pid}]   Cache ID: {dataset_id}")
		
		return f"Dataset cached in subprocess PID {pid}"

	@staticmethod
	def train_model(info_dict: dict) -> float:
		"""Train model with comprehensive error handling and logging"""
		global _subprocess_dataset_cache, _subprocess_dataset_id
		model_id = "unknown"
		
		# Configure logging in the subprocess
		import sys
		import os
		from pathlib import Path
		
		pid = os.getpid()
		
		# Create logs directory structure
		log_dir = Path('logs/slave/training')
		log_dir.mkdir(parents=True, exist_ok=True)
		
		subprocess_log_file = log_dir / f'subprocess_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
		subprocess_logger = logging.getLogger('subprocess_training')
		subprocess_handler = logging.FileHandler(subprocess_log_file)
		subprocess_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
		subprocess_logger.addHandler(subprocess_handler)
		subprocess_logger.setLevel(logging.INFO)
		
		try:
			dataset = info_dict['dataset']
			model_training_request = info_dict['model_request']
			model_id = model_training_request.id
			
			subprocess_logger.info(f"[PID {pid}] Subprocess: Starting training for model {model_id}")
			subprocess_logger.info(f"  Dataset: {dataset.get_tag()}")
			subprocess_logger.info(f"  Epochs: {model_training_request.epochs}")
			subprocess_logger.info(f"  Training type: {model_training_request.training_type}")
			
			# Use cached dataset (should already be loaded by _initialize_dataset_cache)
			dataset_id = f"{dataset.get_tag()}_{dataset.batch_size}_{dataset.validation_split_float}"
			
			if _subprocess_dataset_cache is None or _subprocess_dataset_id != dataset_id:
				# This should NOT happen if pre-loading worked!
				subprocess_logger.warning(f"[PID {pid}] ⚠️  Dataset NOT in cache! Loading now (this shouldn't happen)...")
				dataset.load()
				_subprocess_dataset_cache = dataset
				_subprocess_dataset_id = dataset_id
				subprocess_logger.info(f"[PID {pid}]   Dataset loaded and cached")
			else:
				subprocess_logger.info(f"[PID {pid}] ✓ Using CACHED dataset (already pre-loaded, no loading needed!)")
				dataset = _subprocess_dataset_cache
			
			subprocess_logger.info(f"  Training samples: {dataset.get_training_steps(use_augmentation=False)} steps")
			subprocess_logger.info(f"  Validation samples: {dataset.get_validation_steps()} steps")
			
			# Create and train model
			subprocess_logger.info("Subprocess: Creating model...")
			model = Model(model_training_request, dataset)
			
			subprocess_logger.info("Subprocess: Starting training...")
			start_time = time.time()
			
			if SP.TRAIN_GPU:
				subprocess_logger.info("Subprocess: Training with GPU")
				result = model.build_and_train()
			else:
				subprocess_logger.info("Subprocess: Training with CPU")
				result = model.build_and_train_cpu()
			
			elapsed_time = time.time() - start_time
			subprocess_logger.info(f"Subprocess: Training completed for model {model_id}")
			subprocess_logger.info(f"  Result: {result}")
			subprocess_logger.info(f"  Time elapsed: {elapsed_time:.2f}s")
			
			return result
			
		except Exception as e:
			subprocess_logger.error("=" * 60)
			subprocess_logger.error(f"SUBPROCESS ERROR during training of model {model_id}")
			subprocess_logger.error("=" * 60)
			subprocess_logger.error(f"Error type: {type(e).__name__}")
			subprocess_logger.error(f"Error message: {str(e)}")
			subprocess_logger.error("\nFull traceback:")
			subprocess_logger.error(traceback.format_exc())
			subprocess_logger.error("=" * 60)
			
			# Log additional context
			try:
				subprocess_logger.error("Additional context:")
				subprocess_logger.error(f"  Dataset: {info_dict.get('dataset', {}).get_tag() if 'dataset' in info_dict else 'N/A'}")
				subprocess_logger.error(f"  Model ID: {model_id}")
				if 'model_request' in info_dict:
					req = info_dict['model_request']
					subprocess_logger.error(f"  Epochs: {req.epochs}")
					subprocess_logger.error(f"  Architecture: {str(req.architecture)[:200]}")
			except Exception as ctx_error:
				subprocess_logger.error(f"  Could not log additional context: {ctx_error}")
			
			subprocess_logger.error(f"Log saved to: {subprocess_log_file}")
			subprocess_logger.error("=" * 60)
			
			# Print to stderr so parent process might see it
			print(f"SUBPROCESS ERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
			print(f"See log: {subprocess_log_file}", file=sys.stderr)
			
			# Re-raise the exception
			raise

	async def _start_listening(self) -> aio_pika.Connection:
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Worker started!"})
	 	
		return await self.rabbitmq_client.listen_for_model_params(self._on_model_params_received)

	async def _on_model_params_received(self, model_params):
		"""Handle received model training request with error handling"""
		model_id = model_params.get('id', 'unknown')
		
		# DEDUPLICATION: Skip if this job was already processed
		if model_id in self.processed_jobs:
			logger.warning(f"DUPLICATE: Job {model_id} already processed - ignoring")
			return
		
		# Mark job as processing
		self.processed_jobs.add(model_id)
		
		try:
			logger.info(f"Received model training request (ID: {model_id})")
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Received model training request"})
			
			# Parse model type
			self.model_type = int(model_params['training_type'])
			logger.info(f"  Training type: {self.model_type}")
			
			# Create training request object
			model_training_request = ModelTrainingRequest.from_dict(model_params, self.model_type)
			
			# Verify search space hash
			if not self.search_space_hash == model_training_request.search_space_hash:
				error_msg = f"Search space mismatch! Master: {model_training_request.search_space_hash}, Slave: {self.search_space_hash}"
				logger.error(error_msg)
				raise Exception(error_msg)
			
			logger.info(f"  Search space hash verified: {self.search_space_hash}")
			
			# Prepare training info
			info_dict = {
				'dataset': self.dataset,
				'model_request': model_training_request
			}
			
			# Execute training in persistent process pool
			logger.info(f"Submitting model {model_id} to process pool...")
			training_val, did_finish_epochs = await self.loop.run_in_executor(
				self.pool, 
				self.train_model, 
				info_dict
			)
			
			logger.info(f"Training completed for model {model_id}")
			logger.info(f"  Validation score: {training_val}")
			logger.info(f"  Finished all epochs: {did_finish_epochs}")
			
			# Send results back with hardware info
			hw_info = get_hardware_info()
			logger.info(f"Hardware info: {hw_info.get('gpu_count', 0)} GPUs detected")
			model_training_response = ModelTrainingResponse(
				id=model_training_request.id, 
				performance=training_val, 
				finished_epochs=did_finish_epochs,
				hardware_info=hw_info
			)
			await self._send_performance_to_broker(model_training_response)
			
			logger.info(f"Results sent to broker for model {model_id}")
			SocketCommunication.decide_print_form(
				MSGType.SLAVE_STATUS, 
				{'node': 2, 'msg': f'Model {model_id} completed successfully'}
			)
			
		except Exception as e:
			logger.error("=" * 60)
			logger.error(f"ERROR processing model {model_id}")
			logger.error("=" * 60)
			logger.error(f"Error type: {type(e).__name__}")
			logger.error(f"Error message: {str(e)}")
			logger.error("\nFull traceback:")
			logger.error(traceback.format_exc())
			logger.error("=" * 60)
			
			# Special handling for BrokenProcessPool
			if isinstance(e, concurrent.futures.process.BrokenProcessPool):
				logger.error("CRITICAL: Training subprocess crashed!")
				logger.error("This usually indicates:")
				logger.error("  1. Out of Memory (GPU or RAM)")
				logger.error("  2. Segmentation fault in TensorFlow/CUDA")
				logger.error("  3. Process killed by OS (OOM killer)")
				logger.error("")
				logger.error("Check subprocess logs: subprocess_training_*.log")
				logger.error("Monitor GPU memory: nvidia-smi")
				logger.error("Monitor system memory: free -h")
				logger.error("")
				logger.error("Possible solutions:")
				logger.error("  - Reduce DATASET_BATCH_SIZE in system_parameters.py")
				logger.error("  - Reduce image size (currently 224x224)")
				logger.error("  - Simplify model architecture")
				logger.error("  - Add more RAM/GPU memory")
			
			# Log the received parameters for debugging
			try:
				logger.error("Received parameters:")
				for key, value in model_params.items():
					if key == 'architecture':
						logger.error(f"  {key}: {str(value)[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
					else:
						logger.error(f"  {key}: {value}")
			except Exception as log_error:
				logger.error(f"Could not log parameters: {log_error}")
			
			logger.error("=" * 60)
			
			# Send error response back to master
			try:
				hw_info = get_hardware_info()
				error_response = ModelTrainingResponse(
					id=model_id,
					performance=-1.0,  # Negative score indicates error
					finished_epochs=False,
					hardware_info=hw_info
				)
				await self._send_performance_to_broker(error_response)
				logger.info("Error response sent to broker")
			except Exception as send_error:
				logger.error(f"Failed to send error response to broker: {send_error}")
				logger.error(traceback.format_exc())
			
			# Notify via socket - wrap in try-except to prevent secondary errors
			try:
				SocketCommunication.decide_print_form(
					MSGType.SLAVE_STATUS, 
					{'node': 2, 'msg': f'❌ ERROR: Model {model_id} failed: {str(e)[:100]}'}
				)
			except Exception as socket_error:
				logger.error(f"Failed to send socket notification: {socket_error}")
				# Just print to console as fallback
				print(f"❌ ERROR: Model {model_id} failed: {str(e)[:100]}")
			
			# Don't re-raise - we want the slave to continue processing other tasks
			logger.warning("Slave will continue processing other tasks")

	async def _send_performance_to_broker(self, model_training_response: ModelTrainingResponse):
		"""Send training results to broker with error handling"""
		try:
			logger.info(f"Sending performance to broker for model {model_training_response.id}")
			logger.debug(f"Response: {model_training_response}")
			
			model_training_response_dict = asdict(model_training_response)
			logger.debug(f"Response dict: {model_training_response_dict}")
			
			await self.rabbitmq_client.publish_model_performance(model_training_response_dict)
			logger.info("Performance sent successfully")
			
		except Exception as e:
			logger.error(f"Failed to send performance to broker: {e}")
			logger.error(traceback.format_exc())
			raise


def handle_exception(loop, context):
	"""Global exception handler for asyncio event loop"""
	# context["message"] will always be there; but context["exception"] may not
	msg = context.get("exception", context["message"])
	
	logger.error("=" * 60)
	logger.error("Asyncio event loop exception!")
	logger.error("=" * 60)
	logger.error(f"Message: {msg}")
	
	if "exception" in context:
		logger.error(f"Exception type: {type(context['exception']).__name__}")
		logger.error(f"Exception: {context['exception']}")
		
	# Log all context keys for debugging
	logger.error("\nFull context:")
	for key, value in context.items():
		if key != 'exception':  # Already logged above
			logger.error(f"  {key}: {value}")
	
	logger.error("=" * 60)
	logger.info("Shutting down event loop...")
	loop.stop()

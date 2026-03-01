import asyncio
import time
import json
import os
import GPUtil
from dataclasses import asdict
from typing import Set, Dict, Optional
import aio_pika
import logging
import traceback
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.master_node.communication.master_rabbitmq_client import *
from app.master_node.communication.rabbitmq_monitor import *
from app.master_node.optimization_strategy import OptimizationStrategy, Action, Phase
from app.master_node.state_manager import StateManager, create_state_from_strategy
from app.common.dataset import * 
from system_parameters import SystemParameters as SP
from app.common.socketCommunication import *
import platform
import subprocess
import os
import json

import sys
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"optimization_job_{int(time.time())}.log")

# Configure logging with absolute path
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ],
    force=True
)

def log_info(msg, obj=None):
    """Enhanced logging function that shows code location"""
    frame = sys._getframe(1)  # Get caller frame
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    function = frame.f_code.co_name
    
    # Format message with location info
    location_info = f"{filename}:{lineno} in {function}()"
    
    if obj is not None:
        if hasattr(obj, '__dict__'):
            # For complex objects, convert to dict if possible
            try:
                obj_str = json.dumps(obj.__dict__, default=str)[:150] + "..."
            except:
                obj_str = str(obj)[:150] + "..."
        else:
            obj_str = str(obj)[:150] + "..."
        full_msg = f"INFO [{location_info}] {msg} | {obj_str}"
    else:
        full_msg = f"INFO [{location_info}] {msg}"
    
    # Log the message
    logging.info(full_msg)
    return full_msg

def log_error(msg, exc=None, include_trace=False):
    """Enhanced error logging function"""
    frame = sys._getframe(1)  # Get caller frame
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    function = frame.f_code.co_name
    
    # Format message with location info
    location_info = f"{filename}:{lineno} in {function}()"
    full_msg = f"ERROR [{location_info}] {msg}"
    
    # Log the message
    logging.error(full_msg)
    
    # Log exception if provided
    if exc:
        logging.error(f"Exception: {str(exc)}")
    
    # Add stack trace if requested
    if include_trace:
        stack = ''.join(traceback.format_stack()[:-1])
        logging.error(f"Stack trace:\n{stack}")
    
    return full_msg

def get_hardware_info():
    """
    Gather hardware information about the system
    
    Returns:
        dict: Dictionary containing hardware information
    """
    hardware_info = {}
    
    # CPU information
    hardware_info["cpu_model"] = platform.processor()
    hardware_info["cpu_cores"] = os.cpu_count()
    hardware_info["python_version"] = platform.python_version()
    hardware_info["system"] = platform.system()
    
    # RAM information
    try:
        import psutil
        ram = psutil.virtual_memory()
        hardware_info["ram_total"] = f"{ram.total / (1024**3):.2f} GB"
        hardware_info["ram_available"] = f"{ram.available / (1024**3):.2f} GB"
    except ImportError:
        hardware_info["ram_total"] = "Unknown (psutil not installed)"
    
    # GPU information
    try:
        # Try to get GPU info using NVIDIA tools
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
        # If nvidia-smi isn't available or fails, try TensorFlow
        try:
            import tensorflow as tf
            physical_gpus = tf.config.list_physical_devices('GPU')
            hardware_info["gpu_count"] = len(physical_gpus)
            hardware_info["gpus"] = [{"device": str(gpu)} for gpu in physical_gpus]
            
            # Try to get more GPU details
            if len(physical_gpus) > 0:
                for i, gpu in enumerate(physical_gpus):
                    with tf.device(f'/GPU:{i}'):
                        gpu_name = tf.test.gpu_device_name()
                        if i < len(hardware_info["gpus"]):
                            hardware_info["gpus"][i]["name"] = gpu_name
        except Exception as e:
            log_error(f"Error getting GPU details via TensorFlow", e)
            hardware_info["gpu_count"] = 0
            hardware_info["gpus"] = []
    
    return hardware_info
    
class OptimizationJob:

    def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory):
        log_info(f"Initializing OptimizationJob with dataset: {dataset.get_tag()}")
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.loop = asyncio.get_event_loop()
            self.dataset = dataset
            self.search_space: ModelArchitectureFactory = model_architecture_factory
            
            # Initialize state manager for resume functionality
            self.state_manager = StateManager(results_dir='results_multi')
            resume_state = None
            self.is_resuming = False  # Track if we're resuming
            
            # Auto-resume check - if previous state exists, load it
            if self.state_manager.has_previous_state():
                log_info("Found previous optimization state - attempting to resume...")
                resume_state = self.state_manager.load_state()
                if resume_state:
                    log_info("Successfully loaded previous state - will resume from where left off")
                    self.is_resuming = True
                else:
                    log_info("Could not load previous state - starting fresh")
            else:
                log_info("No previous state found - starting fresh")
            
            log_info("Setting up optimization strategy", {
                "dataset": dataset.get_tag(), 
                "exploration_size": SP.EXPLORATION_SIZE, 
                "hall_of_fame_size": SP.HALL_OF_FAME_SIZE,
                "resuming": resume_state is not None
            })
            
            self.optimization_strategy = OptimizationStrategy(
                self.search_space, 
                self.dataset, 
                SP.EXPLORATION_SIZE, 
                SP.HALL_OF_FAME_SIZE,
                resume_from=resume_state
            )
            
            log_info("Setting up RabbitMQ clients")
            # Creates a connection with a connection types as a parameter
            rabbit_connection_params = RabbitConnectionParams.new()
            self.rabbitmq_client = MasterRabbitMQClient(rabbit_connection_params, self.loop)
            self.rabbitmq_monitor = RabbitMQMonitor(rabbit_connection_params)
            
            # Add state tracking for debugging
            self.state = {
                "models_generated": 0,
                "models_processed": 0,
                "current_phase": "initialization",
                "start_time": None,
                "last_action": None
            }
            
            # Track hardware from all workers
            self.all_worker_hardware = []
            
            # Async scheduler state variables
            self.inflight_jobs: Set[str] = set()
            self.completed_jobs: Set[str] = set()  # For deduplication
            self.max_jobs: int = 1
            self.worker_count: int = 0
            self.job_timeout: int = 4 * 3600  # 4 hours
            self.job_timestamps: Dict[str, float] = {}
            self._generate_lock = asyncio.Lock()
            self._startup_done = False  # Prevent multiple startup executions
            
            log_info("OptimizationJob initialized successfully")
        except Exception as e:
            log_error("Failed to initialize OptimizationJob", e, include_trace=True)
            raise

    async def maybe_generate(self, reason: str = "") -> bool:
        """Generate new job if there's capacity (with async lock and dynamic worker count)"""
        async with self._generate_lock:
            # Dynamically refresh worker count before each generation
            try:
                queue_status = await self.rabbitmq_monitor.get_queue_status()
                self.worker_count = max(1, queue_status.consumer_count)
                self.max_jobs = self.worker_count + 1
                log_info(f"Dynamic worker update: workers={self.worker_count}, max_jobs={self.max_jobs}")
            except Exception as e:
                log_info(f"Could not get queue status, using cached: workers={self.worker_count}, max_jobs={self.max_jobs}")
            
            # Protection against scheduler drift
            if len(self.inflight_jobs) > self.max_jobs + 2:
                log_error(f"SCHEDULER DRIFT: inflight={len(self.inflight_jobs)} > max_jobs+2={self.max_jobs + 2}")
            
            if len(self.inflight_jobs) >= self.max_jobs:
                log_info(f"Skip: inflight={len(self.inflight_jobs)}, max={self.max_jobs} ({reason})")
                return False
            
            job_id = await self.generate_model()
            if not job_id:
                log_error(f"generate_model failed, not adding to inflight")
                return False
            
            # Protection against duplicates
            if job_id in self.inflight_jobs:
                log_error(f"DUPLICATE JOB DETECTED: {job_id}")
            
            self.inflight_jobs.add(job_id)
            self.job_timestamps[job_id] = time.time()
            
            log_info(f"jobs: inflight={len(self.inflight_jobs)} max={self.max_jobs} workers={self.worker_count} ({reason})")
            return True

    async def fill_pipeline(self, reason: str = ""):
        """Fill the pipeline to keep GPUs saturated (robust loop)"""
        target = self.max_jobs
        
        log_info(f"fill_pipeline started: target={target}, current={len(self.inflight_jobs)}, reason={reason}")
        
        while len(self.inflight_jobs) < target:
            success = await self.maybe_generate(reason)
            
            if not success:
                await asyncio.sleep(0.5)  # Wait before retry
            else:
                await asyncio.sleep(0)  # Immediately try next
        
        log_info(f"fill_pipeline completed: inflight={len(self.inflight_jobs)}, target={target}")

    def start_optimization(self, trials: int):
        log_info(f"Starting optimization with {trials} trials")
        self.start_time = time.time()
        self.state["start_time"] = self.start_time
        
        try:
            log_info("Running optimization startup")
            self.loop.run_until_complete(self._run_optimization_startup())
            
            log_info("Starting main optimization loop")
            connection = self.loop.run_until_complete(self._run_optimization_loop(trials))
            
            try:
                log_info("Entering event loop")
                self.loop.run_forever()
            finally:
                log_info("Closing connection")
                self.loop.run_until_complete(connection.close())
                
        except Exception as e:
            log_error("Error in optimization process", e, include_trace=True)
            raise
        finally:
            log_info("Optimization process completed", {
                "total_time": time.time() - self.start_time,
                "models_processed": self.state["models_processed"],
                "models_generated": self.state["models_generated"]
            })

    async def _run_optimization_startup(self):
        # Prevent multiple startup executions
        if self._startup_done:
            log_info("Startup already completed, skipping")
            return
        self._startup_done = True
        
        log_info("Running optimization startup")
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': '*** Running optimization startup ***'})
        
        try:
            # Purge queues to remove stale messages from previous runs
            log_info("Purging RabbitMQ queues for clean start")
            await self.rabbitmq_client.purge_queues()
            
            log_info("Preparing RabbitMQ queues")
            await self.rabbitmq_client.prepare_queues()
            
            log_info("Getting queue status")
            queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
            log_info(f"Queue status retrieved", {
                "consumer_count": queue_status.consumer_count,
                "message_count": queue_status.message_count
            })
            
            # Initialize with queue status
            self.worker_count = max(1, queue_status.consumer_count)
            self.max_jobs = self.worker_count + 1
            log_info(f"Async scheduler init: workers={self.worker_count}, max_jobs={self.max_jobs}")
            
            # If resuming from previous state, DON'T generate new jobs - just wait for workers
            if self.is_resuming:
                log_info("RESUMING: Not generating new jobs - waiting for existing workers to complete")
                # Don't call fill_pipeline - just wait for results from existing jobs
                # The dedup logic will handle any stale results
            else:
                # Fresh start - generate initial jobs
                log_info("FRESH START: Generating initial jobs to fill pipeline")
                await self.fill_pipeline("startup")
                
            self.state["current_phase"] = "exploration"
            log_info("Optimization startup completed")
            
        except Exception as e:
            log_error("Error in optimization startup", e, include_trace=True)
            raise

    async def _run_optimization_loop(self, trials: int) -> aio_pika.Connection:
        log_info(f"Setting up model results listener for {trials} trials")
        try:
            connection = await self.rabbitmq_client.listen_for_model_results(self.on_model_results)
            log_info("Model results listener established")
            return connection
        except Exception as e:
            log_error("Failed to set up model results listener", e, include_trace=True)
            raise

    async def on_model_results(self, response: dict):
        try:
            # Get experiment_id from response
            response_experiment_id = response.get('experiment_id', '')
            
            # Check if this result is from current experiment
            current_experiment_id = getattr(self.optimization_strategy, 'experiment_id', '') if hasattr(self, 'optimization_strategy') else ''
            if response_experiment_id and current_experiment_id and response_experiment_id != current_experiment_id:
                log_info(f"Old experiment result ignored: got {response_experiment_id}, expected {current_experiment_id}")
                return  # Don't fill_pipeline - no job was released
            
            model_training_response = ModelTrainingResponse.from_dict(response)
            self.state["models_processed"] += 1
            
            # Use model_id as job_id (unique within experiment)
            job_id = str(model_training_response.id)
            
            # Check for duplicates first
            if job_id in self.completed_jobs:
                log_info(f"DUPLICATE: job {job_id} already processed - ignoring")
                return
            
            # Check if job is in our inflight list
            if job_id not in self.inflight_jobs:
                log_info(f"UNKNOWN JOB: job {job_id} not in inflight_jobs - ignoring. inflight={len(self.inflight_jobs)}")
                return  # Don't fill_pipeline - job was already processed or unknown
            
            # Valid result - process normally
            # Remove from inflight
            self.inflight_jobs.discard(job_id)
            self.job_timestamps.pop(job_id, None)
            
            # Add to completed jobs (dedup)
            self.completed_jobs.add(job_id)
            
            # Clean up completed_jobs if it gets too large
            if len(self.completed_jobs) > 10000:
                log_info(f"Cleaning up completed_jobs cache (size: {len(self.completed_jobs)})")
                self.completed_jobs.clear()
            
            log_info(f"Job completed: {job_id}, inflight={len(self.inflight_jobs)}")
            
            # Track unique hardware from workers (dedupe by static values only)
            if model_training_response.hardware_info:
                hw = model_training_response.hardware_info
                
                # Create static key from immutable hardware info
                gpu_models = tuple(sorted([g.get('model', '') for g in hw.get('gpus', [])]))
                static_key = (hw.get('cpu_cores', 0), hw.get('gpu_count', 0), gpu_models)
                
                # Check if we've seen this static hardware config before
                is_new = True
                for existing in self.all_worker_hardware:
                    existing_gpu_models = tuple(sorted([g.get('model', '') for g in existing.get('gpus', [])]))
                    existing_key = (existing.get('cpu_cores', 0), existing.get('gpu_count', 0), existing_gpu_models)
                    if static_key == existing_key:
                        is_new = False
                        break
                
                if is_new:
                    self.all_worker_hardware.append(hw)
                    log_info(f"New worker hardware detected", {"cpu_cores": hw.get('cpu_cores'), "gpu_count": hw.get('gpu_count'), "gpu_models": gpu_models})
            
            log_info(f"Received model training response", {
                "id": model_training_response.id,
                "performance": model_training_response.performance,
                "finished_epochs": model_training_response.finished_epochs
            })
            
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received response'})
            cad = str(model_training_response.id) + '|' + str(model_training_response.performance) + '|' + str(model_training_response.finished_epochs)
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
            
            # Process the response through the optimization strategy
            log_info(f"Reporting model response to optimization strategy")
            
            # Report to optimizer (after adding to completed_jobs)
            action: Action = self.optimization_strategy.report_model_response(model_training_response)
            
            # Fill pipeline to keep GPUs saturated (after reporting)
            await self.fill_pipeline("result")
            
            # Save state periodically for resume functionality
            try:
                state_data = create_state_from_strategy(self.optimization_strategy)
                self.state_manager.save_state(state_data)
            except Exception as e:
                log_info(f"Could not save state: {e}")
            
            # NEW: Save intermediate result for this model
            try:
                await self._save_intermediate_result(model_training_response)
            except Exception as e:
                log_info(f"Could not save intermediate result: {e}")
            
            # NEW: Log best model after each completion
            try:
                self._log_best_model_info()
            except Exception as e:
                log_info(f"Could not log best model info: {e}")
            
            self.state["last_action"] = str(action)
            log_info(f"Strategy returned action: {action}")
            
            SocketCommunication.decide_print_form(MSGType.FINISHED_MODEL, {
                'node': 1, 
                'msg': 'Finished a model', 
                'total': self.optimization_strategy.get_training_total()
            })
            
            # Perform action based on strategy recommendation
            # Note: fill_pipeline is called after job removal to keep GPUs saturated
            
            if action == Action.GENERATE_MODEL:
                log_info("Action: GENERATE_MODEL - Pipeline already filled by result handler")
                
            elif action == Action.WAIT:
                log_info("Action: WAIT - Waiting for more model results")
                SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Wait for models'})
                
            elif action == Action.START_NEW_PHASE:
                log_info("Action: START_NEW_PHASE - Transitioning to deep training phase")
                self.state["current_phase"] = "deep_training"
                
                SocketCommunication.decide_print_form(MSGType.CHANGE_PHASE, {'node': 1, 'msg': 'New phase, deep training'})
                    
            elif action == Action.FINISH:
                log_info("Action: FINISH - Optimization process completed")
                self.state["current_phase"] = "finished"
                
                SocketCommunication.decide_print_form(MSGType.FINISHED_TRAINING, {'node': 1, 'msg': 'Finished training'})
                
                # Get the best model
                best_model = self.optimization_strategy.get_best_model()
                log_info("Best model identified", {
                    "id": best_model.model_training_request.id,
                    "performance": best_model.performance,
                    "performance_2": best_model.performance_2
                })
                
                # Log results
                await self._log_results(best_model)
                
                # Validate the model
                log_info("Validating best model")
                model = Model(best_model.model_training_request, self.dataset)
                valid = model.is_model_valid()
                log_info(f"Model validation: {valid}")
                
                # Stop the event loop
                log_info("Stopping event loop")
                self.loop.stop()
                
        except Exception as e:
            log_error("Error processing model result", e, include_trace=True)

    async def generate_model(self, retry_count: int = 0) -> Optional[str]:
        """Generate and send a new model. Returns job_id on success, None on failure."""
        MAX_RETRIES = 5
        log_info("Generating new model")
        try:
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Generating new model'})
            
            # Get model from strategy
            model_training_request: ModelTrainingRequest = self.optimization_strategy.recommend_model()
            self.state["models_generated"] += 1
            
            log_info(f"Model recommended by strategy", {
                "id": model_training_request.id,
                "training_type": model_training_request.training_type,
                "epochs": model_training_request.epochs,
                "is_partial_training": model_training_request.is_partial_training,
                "architecture": model_training_request.architecture
            })
            
            # Check if architecture is None
            if model_training_request.architecture is None:
                log_error(f"Model {model_training_request.id} has None architecture! This should not happen.")
                SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Model architecture is None - regenerating'})
                # Try to generate another model instead of failing, but limit retries
                if retry_count < MAX_RETRIES:
                    log_error(f"Retry {retry_count + 1}/{MAX_RETRIES}")
                    return await self.generate_model(retry_count=retry_count + 1)
                else:
                    log_error(f"Max retries ({MAX_RETRIES}) exceeded for model generation!")
                    SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': f'Max retries exceeded'})
                    return None
            
            # Validate model
            model = Model(model_training_request, self.dataset)
            if not model.is_model_valid():
                log_error(f"Model {model_training_request.id} is not valid!")
                SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Model is not valid'})
                return None
                
            # Send model to broker
            await self._send_model_to_broker(model_training_request)
            log_info(f"Model {model_training_request.id} sent to broker")
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Sent model to broker'})
            
            # Return job_id for tracking (using model_id as unique identifier)
            return str(model_training_request.id)
            
        except Exception as e:
            log_error("Error generating model", e, include_trace=True)
            return None

    async def _send_model_to_broker(self, model_training_request: ModelTrainingRequest):
        log_info(f"Sending model {model_training_request.id} to broker")
        try:
            model_training_request_dict = asdict(model_training_request)
            
            # Log truncated version for clarity
            request_summary = {
                "id": model_training_request.id,
                "epochs": model_training_request.epochs,
                "is_partial_training": model_training_request.is_partial_training
            }
            log_info("Model training request details", request_summary)
            
            # Send to RabbitMQ
            await self.rabbitmq_client.publish_model_params(model_training_request_dict)
            log_info(f"Model {model_training_request.id} published to RabbitMQ")
            
        except Exception as e:
            log_error(f"Error sending model to broker", e, include_trace=True)
            raise

    async def _log_results(self, best_model):
        _send_slack_notification(self, best_model, time.time() - self.start_time, {})
        log_info(f"Logging results for best model {best_model.model_training_request.id}")
        #send mesasge to slack  
        
        try:
            # Simple approach using relative paths
            # This assumes the code is being run from the mloptimizer directory
            results_path = 'results_multi'  # Relative path to mloptimizer/results
            
            # Create results directory if it doesn't exist
            os.makedirs(results_path, exist_ok=True)
            log_info(f"Saving results to directory: {results_path}")

            # Create filename from experiment ID
            filename = best_model.model_training_request.experiment_id
            filepath = os.path.join(results_path, filename)
            log_info(f"Results file path: {filepath}.json")
            
            # Create a structured result object
            elapsed_seconds = time.time() - self.start_time
            result_data = {
                "model_info": asdict(best_model),
                "dataset_ranges": None,
                "performance_metrics": {
                    "elapsed_seconds": elapsed_seconds
                },
                "hardware_info": {}
            }
            
            log_info('Finished optimization')
            log_info('Best model: ', best_model.model_training_request.id)

            # Get dataset ranges
            try:
                self.dataset.load()
                ranges = self.dataset.get_ranges()
                result_data["dataset_ranges"] = ranges
                log_info('Dataset normalization ranges obtained')
            except Exception as e:
                log_error(f"Error getting dataset ranges", e)

            # Format elapsed time
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))
            result_data["performance_metrics"]["elapsed_time"] = elapsed_time
            
            log_info(f"Optimization completed in {elapsed_time} (hh:mm:ss) / {elapsed_seconds:.2f} seconds")

            # Add hardware information - use best model's worker hardware
            try:
                if best_model.hardware_info:
                    result_data["hardware_info"] = best_model.hardware_info
                    log_info("Using best model's worker hardware information")
                else:
                    info = get_hardware_info()
                    result_data["hardware_info"] = info
                    log_info("No worker hardware info, using master hardware")
            except Exception as e:
                log_error(f"Error collecting hardware information", e)
                result_data["hardware_info"] = {}

            # Add all workers hardware information
            result_data["all_workers_hardware"] = self.all_worker_hardware
            log_info(f"Tracked {len(self.all_worker_hardware)} unique worker hardware configurations")

            # Add optimization statistics
            result_data["optimization_stats"] = {
                "models_generated": self.state["models_generated"],
                "models_processed": self.state["models_processed"],
                "exploration_models": len(self.optimization_strategy.exploration_models_completed),
                "deep_training_models": len(self.optimization_strategy.deep_training_models_completed)
            }

            # Save all results as a properly formatted JSON file
            with open(filepath + ".json", "w") as f:
                json.dump(result_data, f, indent=2)
                
            log_info(f"Results successfully saved to {filepath}.json")
            
            # Upload to Google Drive if configured
            await self._upload_to_google_drive(filepath + ".json", result_data)
            
        except Exception as e:
            log_error(f"Error logging results", e, include_trace=True)

    async def _upload_to_google_drive(self, filepath: str, result_data: dict):
        """Upload result file to Google Drive if configured"""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            
            gdrive_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            gdrive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            
            if not gdrive_creds or not gdrive_folder_id:
                log_info("Google Drive not configured. Set GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_DRIVE_FOLDER_ID env vars.")
                return
            
            log_info("Uploading results to Google Drive...")
            
            # Create credentials from environment variable
            import json as json_module
            creds_dict = json_module.loads(gdrive_creds)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            
            # Build Drive service
            service = build('drive', 'v3', credentials=credentials)
            
            # Prepare file metadata
            filename = os.path.basename(filepath)
            dataset_name = result_data.get("model_info", {}).get("model_training_request", {}).get("experiment_id", "unknown")
            
            file_metadata = {
                'name': f"{dataset_name}_{filename}",
                'parents': [gdrive_folder_id]
            }
            
            # Upload file
            media = MediaFileUpload(filepath, resumable=True)
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            log_info(f"Results uploaded to Google Drive: {file.get('webViewLink')}")
            
        except Exception as e:
            log_error(f"Error uploading to Google Drive", e, include_trace=True)

    
async def _send_slack_notification(self, best_model, elapsed_time, result_data):
    """Send a notification to Slack with optimization results"""
    try:
        # Import at function level to avoid dependency issues
        import aiohttp
        
        # Get Slack webhook URL from environment or config
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/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")
        
        if not slack_webhook_url:
            log_info("Slack webhook URL not configured. Skipping notification.")
            return
            
        log_info("Preparing Slack notification")
        
        # Format hardware info for display
        gpu_info = ""
        if result_data["hardware_info"].get("gpus"):
            for gpu in result_data["hardware_info"]["gpus"]:
                gpu_info += f"• {gpu.get('model', 'Unknown GPU')}\n"
        else:
            gpu_info = "• No GPU detected\n"
            
        # Create a nicely formatted Slack message
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🎉 ML Optimization Complete: {best_model.model_training_request.experiment_id}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Dataset:*\n{self.dataset.get_tag()}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Runtime:*\n{elapsed_time}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Best Model ID:*\n{best_model.model_training_request.id}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Performance:*\n{best_model.performance_2:.4f}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Models Generated:*\n{result_data['optimization_stats']['models_generated']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Hardware:*\n{gpu_info}"
                        }
                    ]
                },
                {
                    "type": "divider"
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Results saved to: {os.path.abspath(f'results/{best_model.model_training_request.experiment_id}.json')}"
                        }
                    ]
                }
            ]
        }
        
        # Send the message asynchronously
        async with aiohttp.ClientSession() as session:
            log_info("Sending Slack notification")
            async with session.post(slack_webhook_url, json=message) as response:
                if response.status == 200:
                    log_info("Slack notification sent successfully")
                else:
                    error_text = await response.text()
                    log_error(f"Failed to send Slack notification. Status: {response.status}, Response: {error_text}")
    
    except ImportError:
        log_error("aiohttp package not installed. Cannot send Slack notification.")
    except Exception as e:
        log_error(f"Error sending Slack notification", e)

    async def _save_intermediate_result(self, model_training_response: ModelTrainingResponse):
        """
        Save individual model result after each completion.
        This allows tracking progress even before optimization finishes.
        """
        try:
            # Find the completed model in exploration or deep training
            completed_model = None
            
            # Search in exploration models
            for model in self.optimization_strategy.exploration_models_completed:
                if model.model_training_request.id == model_training_response.id:
                    completed_model = model
                    model_type = "exploration"
                    break
            
            # If not found, search in deep training models
            if completed_model is None:
                for model in self.optimization_strategy.deep_training_models_completed:
                    if model.model_training_request.id == model_training_response.id:
                        completed_model = model
                        model_type = "deep_training"
                        break
            
            if completed_model is None:
                log_info(f"Could not find completed model {model_training_response.id} to save")
                return
            
            # Validate architecture before saving
            if completed_model.model_training_request.architecture is None:
                log_error(f"Model {model_training_response.id} has no architecture - skipping save")
                return
            
            # Create result data (similar to _log_results but for single model)
            result_data = {
                "model_info": asdict(completed_model),
                "dataset_ranges": None,
                "performance_metrics": {
                    "model_id": model_training_response.id,
                    "model_type": model_type,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                },
                "hardware_info": model_training_response.hardware_info or {},
                "optimization_stats": {
                    "exploration_completed": len(self.optimization_strategy.exploration_models_completed),
                    "deep_training_completed": len(self.optimization_strategy.deep_training_models_completed),
                    "phase": self.optimization_strategy.phase.name
                }
            }
            
            # Save to results folder
            results_path = 'results'
            os.makedirs(results_path, exist_ok=True)
            
            filename = f"{self.optimization_strategy.experiment_id}_model_{model_training_response.id}.json"
            filepath = os.path.join(results_path, filename)
            
            with open(filepath, "w") as f:
                json.dump(result_data, f, indent=2, default=str)
            
            log_info(f"Intermediate result saved: {filepath}")
            
            # Also update a summary file with best models so far
            self._update_results_summary()
            
        except Exception as e:
            log_error(f"Error saving intermediate result", e)
    
    def _update_results_summary(self):
        """Update a summary file with all completed models"""
        try:
            results_path = 'results'
            summary_file = os.path.join(results_path, f"{self.optimization_strategy.experiment_id}_summary.json")
            
            # Gather all completed models
            all_models = []
            
            for model in self.optimization_strategy.exploration_models_completed:
                if model.model_training_request.architecture is not None:
                    all_models.append({
                        "id": model.model_training_request.id,
                        "type": "exploration",
                        "performance": model.performance,
                        "performance_2": model.performance_2,
                        "architecture": asdict(model.model_training_request.architecture)
                    })
            
            for model in self.optimization_strategy.deep_training_models_completed:
                if model.model_training_request.architecture is not None:
                    all_models.append({
                        "id": model.model_training_request.id,
                        "type": "deep_training",
                        "performance": model.performance,
                        "performance_2": model.performance_2,
                        "architecture": asdict(model.model_training_request.architecture)
                    })
            
            # Sort by performance
            all_models.sort(key=lambda x: x['performance'], reverse=True)
            
            summary = {
                "experiment_id": self.optimization_strategy.experiment_id,
                "total_models": len(all_models),
                "best_model": all_models[0] if all_models else None,
                "top_5_models": all_models[:5],
                "phase": self.optimization_strategy.phase.name,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            
            log_info(f"Results summary updated: {summary_file}")
            
        except Exception as e:
            log_error(f"Error updating results summary", e)
    
    def _log_best_model_info(self):
        """Log the current best model information after each completion"""
        try:
            # Find best exploration model
            best_exploration = None
            if self.optimization_strategy.exploration_models_completed:
                valid_models = [m for m in self.optimization_strategy.exploration_models_completed 
                               if m.model_training_request.architecture is not None]
                if valid_models:
                    best_exploration = max(valid_models, key=lambda m: m.performance)
            
            # Find best deep training model
            best_deep = None
            if self.optimization_strategy.deep_training_models_completed:
                valid_models = [m for m in self.optimization_strategy.deep_training_models_completed 
                               if m.model_training_request.architecture is not None]
                if valid_models:
                    best_deep = max(valid_models, key=lambda m: m.performance_2 if m.performance_2 > 0 else m.performance)
            
            log_info("=" * 50)
            log_info("CURRENT BEST MODELS")
            log_info("=" * 50)
            
            if best_exploration:
                arch = best_exploration.model_training_request.architecture
                log_info(f"Best Exploration: ID={best_exploration.model_training_request.id}, "
                        f"Performance={best_exploration.performance:.4f}, "
                        f"Base={arch.base_architecture}, Classifier={arch.classifier_layer_type}")
            
            if best_deep:
                arch = best_deep.model_training_request.architecture
                log_info(f"Best Deep Training: ID={best_deep.model_training_request.id}, "
                        f"Performance={best_deep.performance_2:.4f}, "
                        f"Base={arch.base_architecture}, Classifier={arch.classifier_layer_type}")
            
            log_info("=" * 50)
            
            # Send to socket for UI update
            best_msg = ""
            if best_exploration:
                best_msg = f"Best: #{best_exploration.model_training_request.id} ({best_exploration.performance:.4f})"
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': best_msg})
            
        except Exception as e:
            log_error(f"Error logging best model info", e)
from app.common.dataset import *
from app.common.search_space import ModelArchitectureFactory, ImageModelArchitectureFactory, RegressionModelArchitectureFactory, TimeSeriesModelArchitectureFactory
from system_parameters import SystemParameters as SP
from flask_socketio import SocketIO, send
from app.common.socketCommunication import *

class InitNodes:

	def master_socket(self, socketio: SocketIO):
		SocketCommunication.socket = socketio
		SocketCommunication.isSocket = True
		self.master()

	def slave_socket(self, socketio: SocketIO):
		SocketCommunication.socket = socketio
		SocketCommunication.isSocket = True
		self.slave()

	def master(self):
		model_architecture_factory: ModelArchitectureFactory = self.get_model_architecture()
		
		dataset: Dataset = self.get_dataset()
		print(model_architecture_factory)
		from app.master_node.optimization_job import OptimizationJob
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Initilizating master node'})
		optimization_job = OptimizationJob(dataset, model_architecture_factory)
		optimization_job.start_optimization(trials=SP.TRIALS)

	def slave(self):
		model_architecture_factory: ModelArchitectureFactory = self.get_model_architecture()
		dataset: Dataset = self.get_dataset()
		from app.slave_node.training_slave import TrainingSlave
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': 'Initilizating slave node'})
		training_slave = TrainingSlave(dataset, model_architecture_factory)
		training_slave.start_slave()

	def get_model_architecture(self) -> ModelArchitectureFactory:
		if SP.DATASET_TYPE == 1:
			return ImageModelArchitectureFactory()
		elif SP.DATASET_TYPE == 2:
			return RegressionModelArchitectureFactory()
		elif SP.DATASET_TYPE == 3:
			return TimeSeriesModelArchitectureFactory()
		else:
			print("Please enter a valid dataset type")
			return

	def get_dataset(self) -> Dataset:
		if SP.DATASET_TYPE == 1:
			return ImageClassificationBenchmarkDataset(
				SP.DATASET_NAME, 
				SP.DATASET_SHAPE, 
				SP.DATASET_CLASSES, 
				SP.DATASET_BATCH_SIZE, 
				SP.DATASET_VALIDATION_SPLIT)
		elif SP.DATASET_TYPE == 2:
			return RegressionBenchmarkDataset(
				SP.DATASET_NAME, 
				SP.DATASET_SHAPE, 
				SP.DATASET_FEATURES, 
				SP.DATASET_LABELS, 
				SP.DATASET_BATCH_SIZE, 
				SP.DATASET_VALIDATION_SPLIT)
		elif SP.DATASET_TYPE == 3:
			return TimeSeriesBenchmarkDataset(
				SP.DATASET_NAME, 
				SP.DATASET_WINDOW_SIZE, 
				SP.DATASET_DATA_SIZE, 
				SP.DATASET_BATCH_SIZE, 
				SP.DATASET_VALIDATION_SPLIT)
		else:
			
			print("Please enter a valid dataset type")
			return

	#Method for c# graphic interface
	@staticmethod
	def change_system_info(parameters: dict):
		#Rabbit MQ parameters
		SP.INSTANCE_PORT = parameters['RabbitParams']['Port']
		SP.INSTANCE_MODEL_PARAMETER_QUEUE = parameters['RabbitParams']['ParametersQueue']
		SP.INSTANCE_MODEL_PERFORMANCE_QUEUE = parameters['RabbitParams']['PerformanceQueue']
		SP.INSTANCE_HOST_URL = parameters['RabbitParams']['HostURL']
		SP.INSTANCE_USER = parameters['RabbitParams']['User']
		SP.INSTANCE_PASSWORD = parameters['RabbitParams']['Password']
		SP.INSTANCE_VIRTUAL_HOST = parameters['RabbitParams']['VirtualHost']
		#Dataset parameters
		SP.DATASET_NAME = parameters['DatasetParams']['DatasetName']
		if parameters['DatasetParams']['DatasetType'] == 'Image':
			SP.DATASET_TYPE = 1
		elif parameters['DatasetParams']['DatasetType'] == 'Regression':
			SP.DATASET_TYPE = 2
		elif parameters['DatasetParams']['DatasetType'] == 'Time Series':
			SP.DATASET_TYPE = 3
		SP.DATASET_BATCH_SIZE = parameters['DatasetParams']['BatchSize']
		SP.DATASET_VALIDATION_SPLIT = parameters['DatasetParams']['ValidationSplit']
		if parameters['DatasetParams']['DatasetShape']['Item2'] == None:
			SP.DATASET_SHAPE = (parameters['DatasetParams']['DatasetShape']['Item1'])
		elif parameters['DatasetParams']['DatasetShape']['Item3'] == None:
			SP.DATASET_SHAPE = (
				parameters['DatasetParams']['DatasetShape']['Item1'],
				parameters['DatasetParams']['DatasetShape']['Item2'])
		else:
			SP.DATASET_SHAPE = (
				parameters['DatasetParams']['DatasetShape']['Item1'],
				parameters['DatasetParams']['DatasetShape']['Item2'],
				parameters['DatasetParams']['DatasetShape']['Item3'])
		SP.DATASET_CLASSES = parameters['DatasetParams']['Classnumber']
		SP.DATASET_FEATURES = parameters['DatasetParams']['FeatureNumber']
		SP.DATASET_LABELS = parameters['DatasetParams']['LabelsNumber']
		SP.DATASET_WINDOW_SIZE = parameters['DatasetParams']['WindowSize']
		SP.DATASET_DATA_SIZE = parameters['DatasetParams']['FeatureSize']
		#AutoML parameters
		SP.TRAIN_GPU = parameters['AutoMLParams']['UseGPU']
		SP.TRIALS = parameters['AutoMLParams']['Trials']
		SP.EXPLORATION_SIZE = parameters['AutoMLParams']['ExplorationParams']['Size']
		SP.EXPLORATION_EPOCHS = parameters['AutoMLParams']['ExplorationParams']['Epochs']
		SP.EXPLORATION_EARLY_STOPPING_PATIENCE = parameters['AutoMLParams']['ExplorationParams']['EarlyStopping']
		SP.HALL_OF_FAME_SIZE = parameters['AutoMLParams']['HallOfFameParams']['Size']
		SP.HALL_OF_FAME_EPOCHS = parameters['AutoMLParams']['HallOfFameParams']['Epochs']
		SP.HOF_EARLY_STOPPING_PATIENCE = parameters['AutoMLParams']['HallOfFameParams']['EarlyStopping']
		#Model parameters
		SP.DTYPE = parameters['ModelsParams']['DataFormat']
		SP.OPTIMIZER = parameters['ModelsParams']['Optimizer']
		SP.LAYERS_ACTIVATION_FUNCTION = parameters['ModelsParams']['LayersActivation']
		SP.OUTPUT_ACTIVATION_FUNCTION = parameters['ModelsParams']['OutputActivation']
		SP.KERNEL_INITIALIZER = parameters['ModelsParams']['Kernel']
		SP.LOSS_FUNCTION = parameters['ModelsParams']['Loss']
		SP.METRICS = [parameters['ModelsParams']['Metrics']]
		SP.PADDING = parameters['ModelsParams']['Padding']
		SP.WEIGHT_DECAY = parameters['ModelsParams']['WeightDecay']
		SP.LSTM_ACTIVATION_FUNCTION = parameters['ModelsParams']['LSTMActivation']

	@staticmethod
	def change_slave_system_parameters(parameters: dict):
		print("Slave change: ", parameters)
		#Rabbit MQ parameters
		SP.INSTANCE_PORT = parameters['RabbitParams']['Port']
		SP.INSTANCE_MODEL_PARAMETER_QUEUE = parameters['RabbitParams']['ParametersQueue']
		SP.INSTANCE_MODEL_PERFORMANCE_QUEUE = parameters['RabbitParams']['PerformanceQueue']
		SP.INSTANCE_HOST_URL = parameters['RabbitParams']['HostURL']
		SP.INSTANCE_USER = parameters['RabbitParams']['User']
		SP.INSTANCE_PASSWORD = parameters['RabbitParams']['Password']
		SP.INSTANCE_VIRTUAL_HOST = parameters['RabbitParams']['VirtualHost']
		#Dataset parameters
		SP.DATASET_NAME = parameters['DatasetParams']['DatasetName']
		if parameters['DatasetParams']['DatasetType'] == 'Image':
			SP.DATASET_TYPE = 1
		elif parameters['DatasetParams']['DatasetType'] == 'Regression':
			SP.DATASET_TYPE = 2
		elif parameters['DatasetParams']['DatasetType'] == 'Time Series':
			SP.DATASET_TYPE = 3
		SP.DATASET_BATCH_SIZE = parameters['DatasetParams']['BatchSize']
		SP.DATASET_VALIDATION_SPLIT = parameters['DatasetParams']['ValidationSplit']
		if parameters['DatasetParams']['DatasetShape']['Item2'] == None:
			SP.DATASET_SHAPE = (parameters['DatasetParams']['DatasetShape']['Item1'])
		elif parameters['DatasetParams']['DatasetShape']['Item3'] == None:
			SP.DATASET_SHAPE = (
				parameters['DatasetParams']['DatasetShape']['Item1'],
				parameters['DatasetParams']['DatasetShape']['Item2'])
		else:
			SP.DATASET_SHAPE = (
				parameters['DatasetParams']['DatasetShape']['Item1'],
				parameters['DatasetParams']['DatasetShape']['Item2'],
				parameters['DatasetParams']['DatasetShape']['Item3'])
		SP.DATASET_CLASSES = parameters['DatasetParams']['Classnumber']
		SP.DATASET_FEATURES = parameters['DatasetParams']['FeatureNumber']
		SP.DATASET_LABELS = parameters['DatasetParams']['LabelsNumber']
		SP.DATASET_WINDOW_SIZE = parameters['DatasetParams']['WindowSize']
		SP.DATASET_DATA_SIZE = parameters['DatasetParams']['FeatureSize']

		SP.TRAIN_GPU = parameters['TrainGPU']
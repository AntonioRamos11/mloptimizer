from __future__ import absolute_import, division, print_function, unicode_literals
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import mixed_precision
from app.common.inception_module import InceptionV1ModuleBN
from app.common.search_space import *
from app.common.dataset import Dataset
from app.common.model_communication import *
from system_parameters import SystemParameters as SP
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.list_physical_devices('GPU')

class Model:
	def __init__(self, model_training_request: ModelTrainingRequest, dataset: Dataset):
		self.id = model_training_request.id
		self.experiment_id = model_training_request.experiment_id
		self.training_type = model_training_request.training_type
		self.search_space_type: SearchSpaceType = SearchSpaceType(model_training_request.search_space_type)
		self.model_params = model_training_request.architecture
		self.epochs = model_training_request.epochs
		self.early_stopping_patience = model_training_request.early_stopping_patience
		self.is_partial_training = model_training_request.is_partial_training
		self.model: tf.keras.Model
		self.dataset: Dataset = dataset

	def build_model(self, input_shape: tuple, class_count: int):
		if self.search_space_type == SearchSpaceType.IMAGE:
			if(gpus>1):
				return self.build_image_model(self.model_params, input_shape, class_count)
			else:
				return self.build_image_model(self.model_params, input_shape, class_count)
		elif self.search_space_type == SearchSpaceType.REGRESSION:
			return self.build_regression_model(self.model_params, input_shape, class_count)
		elif self.search_space_type == SearchSpaceType.TIME_SERIES:
			return self.build_time_series_model(self.model_params, input_shape, class_count)

	def build_and_train_cpu(self):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Training with CPU"})
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				self.build_and_train()
		except ValueError as e:
			logging.warning(e)

	def build_and_train(self) -> float:
		# Detect available GPUs
		num_gpus = len(gpus)
		
		# Choose appropriate distribution strategy
		if num_gpus >= 2:
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': f"Training with {num_gpus} GPUs using MirroredStrategy"})
			strategy = tf.distribute.MirroredStrategy()
		elif num_gpus == 1:
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Training with single GPU"})
			strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
		else:
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Training with CPU (no GPUs available)"})
			strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
		
		# Set augmentation and dataset parameters
		if self.search_space_type == SearchSpaceType.IMAGE:
			use_augmentation = not self.is_partial_training
		else:
			use_augmentation = False
			
		# Scale batch size based on number of GPUs if needed
		if num_gpus > 1:
			# This will be used when loading data
			original_batch_size = self.dataset.batch_size
			self.dataset.batch_size *= num_gpus
			SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, 
												  {'node': 2, 'msg': f"Scaling batch size from {original_batch_size} to {self.dataset.batch_size}"})
		
		# Get dataset
		input_shape = self.dataset.get_input_shape()
		class_count = self.dataset.get_classes_count()
		train = self.dataset.get_train_data(use_augmentation)
		training_steps = self.dataset.get_training_steps(use_augmentation)
		validation = self.dataset.get_validation_data()
		validation_steps = self.dataset.get_validation_steps()
		test = self.dataset.get_test_data()
		
		# Define learning rate scheduler
		def scheduler(epoch):
			if epoch < 10:
				return 0.001
			else:
				return float(0.001 * tf.math.exp(0.01 * (10 - epoch)).numpy())
		
		# Build and compile model within strategy scope
		with strategy.scope():
			model = self.build_model(input_shape, class_count)
			
			# Set up callbacks
			scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
			
			if self.search_space_type == SearchSpaceType.IMAGE:
				monitor_exploration_training = 'val_loss'
				monitor_full_training = 'val_accuracy'
			elif self.search_space_type == SearchSpaceType.TIME_SERIES:
				monitor_exploration_training = 'loss'
				monitor_full_training = 'loss'	
			else: 
				monitor_exploration_training = 'val_loss'
				monitor_full_training = 'val_loss'

			if self.is_partial_training:
				early_stopping = tf.keras.callbacks.EarlyStopping(
					monitor=monitor_exploration_training, 
					patience=self.early_stopping_patience, 
					verbose=1, 
					restore_best_weights=True
				)
			else:
				early_stopping = tf.keras.callbacks.EarlyStopping(
					monitor=monitor_full_training, 
					patience=self.early_stopping_patience, 
					verbose=1, 
					restore_best_weights=True
				)
		
		# Setup logging
		model_stage = "exp" if self.is_partial_training else "hof"
		log_dir = "logs/{}/{}-{}".format(self.experiment_id, model_stage, str(self.id))
		tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		callbacks = [early_stopping, tensorboard, scheduler_callback]
		
		# Log total model parameters
		total_weights = np.sum([np.prod(v.shape.as_list()) for v in model.variables])
		cad = f'Total weights {total_weights} using {num_gpus} GPU(s)'
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		
		# Train the model
		history = model.fit(
			train,
			epochs=self.epochs,
			steps_per_epoch=training_steps,
			callbacks=callbacks,
			validation_data=validation,
			validation_steps=validation_steps,
		)
		
		# Process results
		did_finish_epochs = self._did_finish_epochs(history, self.epochs)
		if self.search_space_type == SearchSpaceType.IMAGE:
			loss, training_val = model.evaluate(test, verbose=0)
			cad = 'Model accuracy ' + str(training_val)
		else:
			training_val = model.evaluate(test, verbose=0)
			cad = 'Model accuracy ' + str(training_val)
		
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		tf.keras.backend.clear_session()
		
		# If we scaled the batch size, restore it
		if num_gpus > 1:
			self.dataset.batch_size = original_batch_size
		
		return training_val, did_finish_epochs

	def build_and_train_multi_gpu(self) -> float:
		# Set up multi-GPU strategy
		strategy = tf.distribute.MirroredStrategy()
		print(f"Training using {strategy.num_replicas_in_sync} GPUs")
		
		# Scale batch size according to GPU count
		global_batch_size = SP.DATASET_BATCH_SIZE * strategy.num_replicas_in_sync
		
		# Build model within strategy scope
		with strategy.scope():
			input_shape = self.dataset.get_input_shape()
			class_count = self.dataset.get_classes_count()
			model = self.build_model(input_shape, class_count)
		
		# Continue with training using the strategy
		# ...rest of your code...

	def is_model_valid(self) -> bool:
		is_valid = True
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				input_shape = self.dataset.get_input_shape()
				class_count = self.dataset.get_classes_count()
				self.build_model(input_shape, class_count)
		except ValueError as e:
			logging.warning(e)
			is_valid = False
		tf.keras.backend.clear_session()
		return is_valid

	@staticmethod
	def _did_finish_epochs(history, requested_epochs: int) -> bool:
		h = history.history
		trained_epochs = len(h['loss'])
		return requested_epochs == trained_epochs

	def _add_cnn_architecture(self, model: keras.Model, model_parameters = ImageModelArchitectureParameters, activation='relu', padding='same', kernel_initializer='he_uniform'):
		cnn_layers_per_block = model_parameters.cnn_blocks_conv_layers_n
		weight_decay = SP.WEIGHT_DECAY
		for n in range(0, model_parameters.cnn_blocks_n):
			filters = model_parameters.cnn_block_conv_filters[n]
			filter_size = model_parameters.cnn_block_conv_filter_sizes[n]
			max_pooling_size = model_parameters.cnn_block_max_pooling_sizes[n]
			dropout_value = model_parameters.cnn_block_dropout_values[n]
			for m in range(0, cnn_layers_per_block):
				model.add(layers.Conv2D(filters, (filter_size, filter_size), padding=padding, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(weight_decay)))
				model.add(layers.BatchNormalization())
			model.add(keras.layers.MaxPooling2D(3, 2, padding=padding))
			model.add(keras.layers.Dropout(dropout_value))

	def _add_inception_architecture(self, model: keras.Model, model_parameters: ImageModelArchitectureParameters, activation='relu', padding='same'):
		for n in range(0, model_parameters.inception_stem_blocks_n):
			filters = model_parameters.inception_stem_block_conv_filters[n]
			conv_size = model_parameters.inception_stem_block_conv_filter_sizes[n]
			model.add(layers.Conv2D(filters, (conv_size, conv_size), padding='valid', activation=activation))
			model.add(layers.BatchNormalization())
			pool_size = model_parameters.inception_stem_block_max_pooling_sizes[n]
			model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

		for n in range(0, model_parameters.inception_blocks_n):
			conv1x1_filters = model_parameters.inception_modules_conv1x1_filters[n]
			conv3x3_reduce_filters = model_parameters.inception_modules_conv3x3_reduce_filters[n]
			conv3x3_filters = model_parameters.inception_modules_conv3x3_filters[n]
			conv5x5_reduce_filters = model_parameters.inception_modules_conv5x5_reduce_filters[n]
			conv5x5_filters = model_parameters.inception_modules_conv5x5_filters[n]
			pooling_conv_filters = model_parameters.inception_modules_pooling_conv_filters[n]
		for i in range(0, model_parameters.inception_modules_n):
			model.add(
				InceptionV1ModuleBN(
					conv1x1_filters,
					conv3x3_reduce_filters,
					conv3x3_filters,
					conv5x5_reduce_filters,
					conv5x5_filters,
					pooling_conv_filters,
				)
			)
		model.add(keras.layers.MaxPool2D(3, 2, padding=padding))

	def _add_mlp_architecture(self, model: keras.Model, model_parameters, class_count: int, kernel_initializer='normal', activation='relu'):
		for n in range(0, model_parameters.classifier_layers_n):
			units = model_parameters.classifier_layers_units[n]
			model.add(keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer))
			dropout = model_parameters.classifier_dropouts[n]
			model.add(keras.layers.Dropout(dropout))

	def _add_time_series_lstm_architecture(self, model: keras.Model, model_parameters: TimeSeriesModelArchitectureParameters, class_count: int, activation='tanh'):
		for n in range(model_parameters.lstm_layers_n-1):
			units = model_parameters.lstm_layers_units[n]
			model.add(keras.layers.LSTM(units, return_sequences=True, activation=activation))
		units = model_parameters.lstm_layers_units[model_parameters.lstm_layers_n-1]
		model.add(keras.layers.LSTM(units, activation=activation))

	def build_image_model(self, model_parameters: ImageModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		mixed_precision.Policy("mixed_float16")
		"""print('Compute dtype: %s' % policy.compute_dtype)
		print('Variable dtype: %s' % policy.variable_dtype)"""
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'cnn':
			self._add_cnn_architecture(model, model_parameters, SP.LAYERS_ACTIVATION_FUNCTION, SP.PADDING, SP.KERNEL_INITIALIZER)
		elif model_parameters.base_architecture == 'inception':
			self._add_inception_architecture(model, model_parameters, SP.LAYERS_ACTIVATION_FUNCTION, SP.PADDING)

		if model_parameters.classifier_layer_type == 'gap':
			model.add(keras.layers.Conv2D(class_count, (1,1), activation=SP.LAYERS_ACTIVATION_FUNCTION, kernel_initializer=SP.KERNEL_INITIALIZER))
			model.add(keras.layers.GlobalAveragePooling2D())
			model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		elif model_parameters.classifier_layer_type == 'mlp':
			model.add(keras.layers.Flatten())
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
			model.add(keras.layers.Dense(class_count))
			model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION, metrics=SP.METRICS)
		elapsed_seconds = int(round(time.time() * 1000)) - start_time
		print("Model building took", elapsed_seconds, "(miliseconds)")
		model.summary()
		return model

	def build_regression_model(self, model_parameters: RegressionModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'mlp':
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
		model.add(keras.layers.Dense(class_count))
		model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION)
		elapsed_seconds = int(round(time.time() * 1000)) - start_time
		print('Model building took', elapsed_seconds, '(miliseconds)')
		model.summary()
		return model

	def build_time_series_model(self, model_parameters: TimeSeriesModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		mixed_precision.set_policy("mixed_float16")
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'lstm':
			self._add_time_series_lstm_architecture(model, model_parameters, class_count, SP.LSTM_ACTIVATION_FUNCTION)
		if model_parameters.base_architecture == 'mlp':
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
		#All combination has the same final layers
		model.add(keras.layers.Dense(class_count))
		model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION)
		elapsed_seconds = int(round(time.time() * 1000))- start_time
		print("Model building took", elapsed_seconds, "(miliseconds)")
		model.summary()
		return model
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc

from tensorflow.keras import layers, regularizers
from tensorflow.keras import mixed_precision
from app.common.inception_module import InceptionV1ModuleBN
from app.common.search_space import *
from app.common.dataset import Dataset
from app.common.model_communication import *
from system_parameters import SystemParameters as SP
from app.common.hardware_performance_logger import HardwarePerformanceLogger
from app.common.hardware_performance_callback import HardwarePerformanceCallback
#physical_devices = tf.config.list_physical_devices('GPU')

#tf.config.experimental.set_memory_growth(physical_devices[0], True)


from tensorflow.config.experimental import VirtualDeviceConfiguration



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
		self.performance_logger = HardwarePerformanceLogger(tf_module=tf)

	def convert_to_tfrecord(self, data, filename):
		with tf.io.TFRecordWriter(filename) as writer:
			for x, y in data:
				feature = {
					'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
					'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
				}
				example = tf.train.Example(features=tf.train.Features(feature=feature))
				writer.write(example.SerializeToString())

	def _reset_gpu_memory(self):
		"""Aggressively clean up GPU memory"""
		try:
			# Clear TensorFlow session
			tf.keras.backend.clear_session(
				free_memory=True
			)

			
			# Force garbage collection
			gc.collect()
			
			# Give GPU time to release memory
			time.sleep(0.5)
			
			logging.info("GPU memory reset complete")
		except Exception as e:
			logging.warning(f"Error resetting GPU memory: {e}")

	def build_model(self, input_shape: tuple, class_count: int):
		if self.search_space_type == SearchSpaceType.IMAGE:
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
		 # Start timing

		self._reset_gpu_memory()
		self.performance_logger.start_timing()
		build_start_time = int(round(time.time() * 1000))
		 # Set memory growth to avoid allocating all GPU memory at once
		


		# Enable mixed precision globally
		tf.keras.mixed_precision.set_global_policy('mixed_float16')

		# Detect GPUs and configure strategy
		gpus = tf.config.list_physical_devices('GPU')
		num_gpus = len(gpus)
		
		if num_gpus >= 2:
			# Enable batch size scaling
			original_batch_size = self.dataset.batch_size
			self.dataset.batch_size *= num_gpus  # Critical for multi-GPU perf
			
			strategy = tf.distribute.MirroredStrategy(
				cross_device_ops=tf.distribute.NcclAllReduce(),  # Use NCCL for multi-GPU communication
				devices=[f"/gpu:{i}" for i in range(num_gpus)]
			)
		elif num_gpus == 1:
			strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
		else:
			strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
		
		# Set augmentation and dataset parameters
		if self.search_space_type == SearchSpaceType.IMAGE:
			use_augmentation = not self.is_partial_training
		else:
			use_augmentation = False



		# Get dataset
		input_shape = self.dataset.get_input_shape()
		class_count = self.dataset.get_classes_count()
		
		# Get raw datasets
		train = self.dataset.get_train_data(use_augmentation)
		validation = self.dataset.get_validation_data()
		test = self.dataset.get_test_data()
		
		# Optimize data pipeline for multi-GPU performance
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		
		# Apply optimizations to training data
		train = train.with_options(options)
		train = train.cache()  # Cache data after first epoch if it fits in memory
		train = train.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch while current is processing
		
		# Apply the same to validation data
		validation = validation.with_options(options)
		validation = validation.prefetch(tf.data.AUTOTUNE)
		
		# Apply the same to test data
		test = test.with_options(options)
		test = test.prefetch(tf.data.AUTOTUNE)
		
		# Add to Dataset preprocessing
		def preprocess_gpu(x, y):
			# Move preprocessing to GPU for operations like:
			x = tf.image.random_flip_left_right(x)
			x = tf.image.random_brightness(x, 0.1)
			return x, y
		
		# Apply to dataset
		train = train.map(preprocess_gpu, num_parallel_calls=tf.data.AUTOTUNE)
		
		training_steps = self.dataset.get_training_steps(use_augmentation)
		validation_steps = self.dataset.get_validation_steps()
		
		# Define learning rate scheduler
		def scheduler(epoch):
			if epoch < 10:
				return 0.001
			else:
				return float(0.001 * tf.math.exp(0.01 * (10 - epoch)).numpy())
		
		# Build and compile model within strategy scope
		with strategy.scope():
			model = self.build_model(input_shape, class_count)
			build_time_ms = int(round(time.time() * 1000)) - build_start_time
			
			tf.config.optimizer.set_jit(True)
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
		
		# Add performance logger callback
		performance_callback = HardwarePerformanceCallback(self.performance_logger)
		callbacks = [early_stopping, tensorboard, scheduler_callback, performance_callback]
		
		# Get model metrics before training
		model_metrics = self.performance_logger.get_model_metrics(
			model, 
			self.id, 
			self.experiment_id,
			self.search_space_type.name
		)
		
		# Train the model
		history = model.fit(
			train,
			epochs=self.epochs,
			steps_per_epoch=training_steps,
			callbacks=callbacks,
			validation_data=validation,
			validation_steps=validation_steps,
		)
		
		# Get optimizer information
		# Get optimizer information
		optimizer_name = SP.OPTIMIZER
		# Extract learning rate from the optimizer
		learning_rate = scheduler(self.epochs - 1)  # Get the learning rate for the last epoch
		
		# Get training metrics after training
		training_metrics = self.performance_logger.get_training_metrics(
			history,
			build_time_ms,
			self.dataset.batch_size,
			optimizer_name,
			learning_rate
		)
		
		# Save the hardware performance log
		log_file = self.performance_logger.save_log(model_metrics, training_metrics)
		print(f"Hardware performance log saved to: {log_file}")
		
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
		
		# Save the processed dataset
		tf.data.experimental.save(processed_dataset, "path/to/saved/dataset")

		# Load it later
		loaded_dataset = tf.data.experimental.load("path/to/saved/dataset")
		
		return training_val, did_finish_epochs

	
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

	def checkpoint_dataset(self, dataset, directory, name):
		"""Save a preprocessed dataset to disk to avoid repeating preprocessing"""
		import os
		
		# Create directory if it doesn't exist
		os.makedirs(directory, exist_ok=True)
		
		# Create a path that includes relevant model info
		path = os.path.join(directory, f"{name}_{self.search_space_type.name}_{self.id}.tf_dataset")
		
		# Save the dataset
		try:
			logging.info(f"Saving preprocessed dataset to {path}")
			tf.data.experimental.save(dataset, path)
			logging.info(f"Successfully saved dataset checkpoint")
			return path
		except Exception as e:
			logging.warning(f"Failed to save dataset checkpoint: {e}")
			return None

	def load_checkpointed_dataset(self, directory, name):
		"""Load a preprocessed dataset from disk if it exists"""
		import os
		
		# Create the expected path
		path = os.path.join(directory, f"{name}_{self.search_space_type.name}_{self.id}.tf_dataset")
		
		# Check if the dataset exists
		if os.path.exists(path):
			try:
				logging.info(f"Loading preprocessed dataset from {path}")
				
				# TensorFlow's dataset loading function
				dataset = tf.data.experimental.load(path)
				
				# Need to specify output types and shapes as they aren't saved
				element_spec = dataset.element_spec
				logging.info(f"Successfully loaded dataset with spec: {element_spec}")
				return dataset
			except Exception as e:
				logging.warning(f"Failed to load dataset checkpoint: {e}")
				return None
		else:
			logging.info(f"No dataset checkpoint found at {path}")
			return None
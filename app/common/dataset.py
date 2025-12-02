import abc
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import pandas as pd
import os
from sklearn import preprocessing
from app.common.preprocessing import *

#Abstract class
class Dataset (abc.ABC):
	@abc.abstractmethod
	def load(self):
		pass

	@abc.abstractclassmethod
	def get_train_data(self):
		pass

	@abc.abstractclassmethod
	def get_validation_data(self):
		pass

	@abc.abstractclassmethod
	def get_test_data(self):
		pass

	@abc.abstractclassmethod
	def get_training_steps(self) -> int:
		pass

	@abc.abstractclassmethod
	def get_validation_steps(self) -> int:
		pass

	@abc.abstractclassmethod
	def get_testing_steps(self) -> int:
		pass

	@abc.abstractclassmethod
	def get_input_shape(self) -> tuple:
		pass

	@abc.abstractclassmethod
	def get_tag(self) -> str:
		pass

class ImageClassificationBenchmarkDataset(Dataset):

	def __init__(self, dataset_name: str, shape:tuple, class_count=1, batch_size=128, validation_split=0.2):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.validation_split_float = validation_split
		self.shuffle_cache = self.batch_size * 2
		self.shape = shape
		self.class_count = class_count

	def load(self):
		try:
			train_split_float = float(1.0 - self.validation_split_float)
			val_split_percent = int(self.validation_split_float * 100)
			train_split_percent = int(train_split_float * 100)
			#Tensorflow dataset load
			self.train_original, self.info = tfds.load(self.dataset_name, with_info=True, as_supervised=True, split='train[:{}%]'.format(train_split_percent))
			train_augmented = tfds.load(self.dataset_name, as_supervised=True, split='train[:{}%]'.format(train_split_percent))
			train_augmented = train_augmented.map(self._augment)
			self.train = self.train_original.concatenate(train_augmented)
			self.validation = tfds.load(self.dataset_name, as_supervised=True, split='train[-{}%:]'.format(val_split_percent))
			self.train_split_count = self.info.splits['train'].num_examples * train_split_float
			self.validation_split_count = self.info.splits['train'].num_examples * self.validation_split_float
			self.test = tfds.load(self.dataset_name, as_supervised=True, split='test')
		except:
			#InitNodes.decide_print_form(MSGType.MASTER_ERROR, {'node': 1, 'msg': 'Somethings went wrong trying to load the dataset, please check the parameters and info'})
			print('Somethings went wrong trying to load the Image dataset, please check the parameters and info')
			raise

	def get_train_data(self, use_augmentation: False):
		train_data = None
		if use_augmentation:
			train_data = self.train.map(self._scale).cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		else:
			train_data = self.train_original.map(self._scale).cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		return train_data

	def get_validation_data(self):
		validation_data = self.validation.map(self._scale).cache().batch(self.batch_size)
		return validation_data

	def get_test_data(self):
		test_data = self.test.map(self._scale).cache().batch(self.batch_size)
		return test_data

	def get_training_steps(self, use_augmentation: False) -> int:
		if use_augmentation:
			return int(np.ceil(self.train_split_count/self.batch_size)) * 2
		else:
			return int(np.ceil(self.train_split_count/self.batch_size))

	def get_validation_steps(self) -> int:
		return int(np.ceil(self.validation_split_count/self.batch_size))

	def get_testing_steps(self):
		pass

	def get_input_shape(self) -> tuple:
		return self.shape

	def get_classes_count(self) -> int:
		return self.class_count

	def get_ranges(self):
		return ""

	def get_tag(self):
		return self.dataset_name

	@staticmethod
	def _scale(image, label):
		image = tf.cast(image, tf.float32)
		image /= 255
		return image, label

	@staticmethod
	def _augment(image, label):
		image = tf.image.random_flip_left_right(image)
		#RGB-only augmentations
		if image.shape[2] == 3:
			image = tf.image.random_hue(image, 0.08)
			image = tf.image.random_saturation(image, 0.6, 1.6)
		image = tf.image.random_brightness(image, 0.05)
		image = tf.image.random_contrast(image, 0.7, 1.3)
		return image, label

class RegressionBenchmarkDataset(Dataset):
	def __init__(self, dataset_name: str, shape:tuple, feature_size=1, n_labels=1, batch_size=128, validation_split=0.2):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.validation_split_float = validation_split
		self.shuffle_cache = self.batch_size * 2
		self.n_labels = n_labels
		self.shape = shape
		self.ranges = []
		self.feature_size = feature_size

	def load(self):
		train_split_float = float(1.0 - self.validation_split_float)
		val_split_percent = int(self.validation_split_float * 100)
		train_split_percent = int(train_split_float * 100)
		try:
			self.train_original, self.info = tfds.load(self.dataset_name, with_info=True, as_supervised=True, split='train[:{}%]'.format(train_split_percent))
			self.validation = tfds.load(self.dataset_name, as_supervised=True, split='train[-{}%:]').format(val_split_percent)
			self.train_split_count = self.info.splits['train'].num_examples * train_split_float
			self.validation_split_count = self.info.splits['train'].num_examples * self.validation_split_float
			self.test = tfds.load(self.dataset_name, as_supervised=True, split='test')
		except:
			try:
				route = '../mloptimizermodelgenerator/Datasets/Regression/'+self.dataset_name
				with open(route+'info.json') as jsonfile:
					info = json.load(jsonfile)
				self.train_split_count = int(info['splits']['train']*train_split_float)
				self.validation_split_count = int(info['splits']['train']*self.validation_split_float)
				all_data = np.array(pd.read_csv(route+'.csv'))
				all_data, self.ranges = normalization(all_data)
				self.train_original = all_data[:self.train_split_count]
				self.validation = all_data[self.train_split_count : self.train_split_count + self.validation_split_count]
				self.test = all_data[-info['splits']['test']:]
				self.train_original = self.numpy_data_to_tfdataset(self.train_original, self.n_labels)
				self.validation = self.numpy_data_to_tfdataset(self.validation, self.n_labels)
				self.test = self.numpy_data_to_tfdataset(self.test, self.n_labels)
			except:
				#InitNodes.decide_print_form(MSGType.MASTER_ERROR, {'node': 1, 'msg': 'Somethings went wrong trying to load the dataset, please check the parameters and info'})
				print('Somethings went wrong trying to load the Regression dataset, please check the parameters and info')
				raise

	def numpy_data_to_tfdataset(self, data, n_labels):
		labels = data[:, -n_labels:]
		samples = data[:, :-n_labels]
		dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
		return dataset

	def get_train_data(self, use_augmentation: False):
		#train_data = self.train_original.shuffle(self.shuffle_cache).cache().batch(self.batch_size).repeat()
		train_data = self.train_original.cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		return train_data

	def get_validation_data(self):
		validation_data = self.validation.cache().batch(self.batch_size)
		return validation_data

	def get_test_data(self):
		test_data = self.test.cache().batch(self.batch_size)
		return test_data

	def get_training_steps(self, use_augmentation: False) -> int:
		return int(np.ceil(self.train_split_count/self.batch_size))

	def get_validation_steps(self) -> int:
		return int(np.ceil(self.validation_split_count/self.batch_size))

	def get_testing_steps(self):
		pass

	def get_input_shape(self):
		return self.shape

	def get_classes_count(self):
		return self.n_labels

	def get_ranges(self):
		return self.ranges

	def get_tag(self):
		return self.dataset_name

class TimeSeriesBenchmarkDataset(Dataset):
	def __init__(self, dataset_name: str, window_size: int, data_size=1, batch_size=128, validation_split=0.2):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.validation_split_float = validation_split
		self.shuffle_cache = self.batch_size * 2
		self.window_size = window_size
		self.ranges = []
		self.data_size = data_size

	def load(self):
		train_split_float = float(1.0 - self.validation_split_float)
		val_split_percent = int(self.validation_split_float * 100)
		train_split_percent = int(train_split_float * 100)
		try:
			self.train_original, self.info = tfds.load(self.dataset_name, with_info=True, as_supervised=True, split='train[:{}%]'.format(train_split_percent))
			self.validation = tfds.load(self.dataset_name, as_supervised=True, split='train[-{}%:]').format(val_split_percent)
			self.train_split_count = self.info.splits['train'].num_examples * train_split_float
			self.validation_split_count = self.info.splits['train'].num_examples * self.validation_split_float
			self.test = tfds.load(self.dataset_name, as_supervised=True, split='test')
		except:
			try:
				ruta="../regressionmloptimizer/Datasets/TimeSeries/"+self.dataset_name
				with open(ruta+'info.json') as jsonfile:
					info = json.load(jsonfile)
				self.train_split_count = int(info['splits']['train']*train_split_float)
				self.validation_split_count = int(info['splits']['train']*self.validation_split_float)
				data_col = info['features']['date']+info['features']['value']-1
				all_data = np.array(pd.read_csv(ruta+'.csv'))
				all_data = all_data[:, data_col:]
				#Normalizar la información.
				all_data, self.ranges = normalization(all_data)
				self.train_original = all_data[:self.train_split_count]
				self.validation = all_data[self.train_split_count : self.train_split_count + self.validation_split_count]
				self.test = all_data[-info['splits']['test']:]
				self.train_original = self.time_series_partition(self.train_original, self.window_size)
				self.validation = self.time_series_partition(self.validation, self.window_size)
				self.test = self.time_series_partition(self.test, self.window_size)
			except:
				#InitNodes.decide_print_form(MSGType.MASTER_ERROR, {'node': 1, 'msg': 'Somethings went wrong trying to load the dataset, please check the parameters and info'})
				print('Somethings went wrong trying to load the time series dataset, please check the parameters and info')
				raise

	def time_series_partition(self, data, window_size):
		full_data = []
		for i in range(len(data) - window_size):
			x_part = data[i:i+window_size]
			y_part = data[i+window_size]
			full_data.append(np.append(x_part, y_part))
		full_data = np.array(full_data)
		full_data = np.asarray(full_data).astype('float64')
		dataset = self.numpy_data_to_tfdataset(full_data, 1)
		return dataset

	def numpy_data_to_tfdataset(self,data, n_labels):
		try:
			labels = data[:,-n_labels:]
		except:
			print("Error numpy to dataset",data)
			print(data.shape)
		samples = data[:,:-n_labels]
		samples = samples.reshape((samples.shape[0], 1, samples.shape[1]))
		dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
		return dataset

	def get_train_data(self, use_augmentation: False):
		train_data = self.train_original.cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		return train_data

	def get_validation_data(self):
		validation_data = self.validation.cache().batch(self.batch_size)
		return validation_data

	def get_test_data(self):
		test_data = self.test.cache().batch(self.batch_size)
		return test_data

	def get_training_steps(self, use_augmentation: False) -> int:
		return int(np.ceil(self.train_split_count / self.batch_size))

	def get_validation_steps(self) -> int:
		return int(np.ceil(self.validation_split_count / self.batch_size))

	def get_testing_steps(self):
		pass

	def get_input_shape(self) -> tuple:
		return (self.data_size, self.window_size)

	def get_classes_count(self):
		return self.data_size

	def get_ranges(self):
		return self.ranges

	def get_tag(self):
		return self.dataset_name


class GrietasBachesDataset(Dataset):
	"""
	Custom Dataset class for GRIETAS (cracks) and BACHES (potholes) that integrates with MLOptimizer.
	Loads images from local folder structure instead of tensorflow_datasets.
	"""
	
	def __init__(self, dataset_name: str, shape: tuple, class_count=2, batch_size=32, validation_split=0.2, dataset_path='./dataset_grietas_baches'):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.validation_split_float = validation_split
		self.shuffle_cache = self.batch_size * 2
		self.shape = shape
		self.class_count = class_count
		self.dataset_path = dataset_path
		
	def load(self):
		"""Load GRIETAS and BACHES dataset from folder structure"""
		try:
			print(f"Loading {self.dataset_name} dataset from {self.dataset_path}...")
			
			# Import here to avoid circular dependencies
			from load_grietas_baches_dataset import load_grietas_baches_dataset
			
			# Load the dataset
			(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
				load_grietas_baches_dataset(
					dataset_path=self.dataset_path,
					target_size=(self.shape[0], self.shape[1]),
					validation_split=self.validation_split_float,
					test_split=0.1
				)
			
			# Convert from CHW to HWC format (TensorFlow format)
			train_data = np.transpose(train_data, (0, 2, 3, 1))
			val_data = np.transpose(val_data, (0, 2, 3, 1))
			test_data = np.transpose(test_data, (0, 2, 3, 1))
			
			# Store counts
			self.train_split_count = len(train_data)
			self.validation_split_count = len(val_data)
			self.test_split_count = len(test_data)
			
			# Convert to TensorFlow datasets
			self.train_original = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
			self.validation = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
			self.test = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
			
			# Create augmented training data
			train_augmented = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
			train_augmented = train_augmented.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
			self.train = self.train_original.concatenate(train_augmented)
			
			print(f"Dataset loaded successfully:")
			print(f"  Training: {self.train_split_count} images")
			print(f"  Validation: {self.validation_split_count} images")
			print(f"  Test: {self.test_split_count} images")
			
		except Exception as e:
			print(f'ERROR: Failed to load GRIETAS and BACHES dataset: {e}')
			print(f'Make sure the dataset exists at: {self.dataset_path}')
			print('Expected structure:')
			print('  dataset_grietas_baches/')
			print('    ├── BACHES/')
			print('    │   └── *.png')
			print('    └── GRIETAS/')
			print('        └── *.png')
			raise
	
	def get_train_data(self, use_augmentation: bool = False):
		"""Return training data with optional augmentation"""
		if use_augmentation:
			train_data = self.train.cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		else:
			train_data = self.train_original.cache().shuffle(self.shuffle_cache).batch(self.batch_size).repeat()
		return train_data
	
	def get_validation_data(self):
		"""Return validation data"""
		validation_data = self.validation.cache().batch(self.batch_size)
		return validation_data
	
	def get_test_data(self):
		"""Return test data"""
		test_data = self.test.cache().batch(self.batch_size)
		return test_data
	
	def get_training_steps(self, use_augmentation: bool = False) -> int:
		"""Calculate training steps per epoch"""
		if use_augmentation:
			return int(np.ceil(self.train_split_count / self.batch_size)) * 2
		else:
			return int(np.ceil(self.train_split_count / self.batch_size))
	
	def get_validation_steps(self) -> int:
		"""Calculate validation steps"""
		return int(np.ceil(self.validation_split_count / self.batch_size))
	
	def get_testing_steps(self):
		"""Calculate testing steps"""
		return int(np.ceil(self.test_split_count / self.batch_size))
	
	def get_input_shape(self) -> tuple:
		"""Return input shape"""
		return self.shape
	
	def get_classes_count(self) -> int:
		"""Return number of classes"""
		return self.class_count
	
	def get_ranges(self):
		"""Return normalization ranges (not used for image data)"""
		return ""
	
	def get_tag(self):
		"""Return dataset name tag"""
		return self.dataset_name
	
	@staticmethod
	def _augment(image, label):
		"""Apply data augmentation to images"""
		# Random flip
		image = tf.image.random_flip_left_right(image)
		# Random brightness/contrast (useful for road images with varying lighting)
		image = tf.image.random_brightness(image, 0.2)
		image = tf.image.random_contrast(image, 0.8, 1.2)
		# Random hue/saturation for RGB images
		image = tf.image.random_hue(image, 0.08)
		image = tf.image.random_saturation(image, 0.6, 1.6)
		# Clip values to [0, 1]
		image = tf.clip_by_value(image, 0.0, 1.0)
		return image, label
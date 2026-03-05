from dataclasses import dataclass
from typing import Optional

from app.common.search_space import *

@dataclass(frozen=False)
class ModelTrainingRequest:
	id: int
	training_type: int
	experiment_id: str
	architecture: dataclass
	epochs: int
	early_stopping_patience: int
	is_partial_training: bool
	search_space_type: str
	search_space_hash: str
	dataset_tag: str

	@classmethod
	def from_dict(cls, body_dict, model_arch: int):
		_id = body_dict['id']
		_experiment_id = body_dict['experiment_id']
		_training_type = body_dict['training_type']
		model_arch = body_dict['training_type']
		if model_arch == 1:
			_architecture = ImageModelArchitectureParameters.from_dict(body_dict['architecture'])
		elif model_arch == 2:
			_architecture = RegressionModelArchitectureParameters.from_dict(body_dict['architecture'])
		elif model_arch == 3:
			_architecture = TimeSeriesModelArchitectureParameters.from_dict(body_dict['architecture'])
		_epochs = body_dict['epochs']
		_early_stopping_patience = body_dict['early_stopping_patience']
		_is_partial_training = body_dict['is_partial_training']
		_search_space_type = body_dict['search_space_type']
		_search_space_hash = body_dict['search_space_hash']
		_dataset_tag = body_dict['dataset_tag']
		return cls(_id, _training_type, _experiment_id, _architecture, _epochs, _early_stopping_patience, _is_partial_training, _search_space_type, _search_space_hash, _dataset_tag)

@dataclass
class ModelTrainingResponse:
	id: int
	performance: float
	finished_epochs: bool
	hardware_info: Optional[dict] = None

	@classmethod
	def from_dict(cls, body_dict):
		_id = body_dict['id']
		_performance = body_dict['performance']
		_finished_epochs = body_dict['finished_epochs']
		_hardware_info = body_dict.get('hardware_info', None)
		return cls(_id, _performance, _finished_epochs, _hardware_info)

@dataclass
class CompletedModel:
	model_training_request: ModelTrainingRequest
	performance: float = -1
	performance_2: float = -1
	hardware_info: Optional[dict] = None
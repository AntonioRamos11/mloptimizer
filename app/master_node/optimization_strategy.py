import datetime
from enum import Enum
from typing import List
import optuna
from optuna.samplers import TPESampler
from optuna.structs import TrialState
from app.common.model_communication import *
from app.common.search_space import *
from app.common.repeat_pruner import RepeatPruner
from app.common.dataset import Dataset
from system_parameters import SystemParameters as SP
from app.common.socketCommunication import *

from optuna.pruners import HyperbandPruner
class OptimizationStrategy(object):

	def __init__(self, model_architecture_factory: ModelArchitectureFactory, dataset: Dataset, exploration_trials: int, hall_of_fame_size: int):
		self.model_architecture_factory:ModelArchitectureFactory = model_architecture_factory
		self.dataset: Dataset = dataset
		self.storage = optuna.storages.InMemoryStorage()
		self.main_study: optuna.Study = optuna.create_study(study_name=dataset.get_tag(), storage=self.storage, load_if_exists=True, pruner=HyperbandPruner(),
														direction='maximize', sampler=TPESampler(n_ei_candidates=5000, n_startup_trials=30,))
		self.study_id = 0
		self.experiment_id = dataset.get_tag() + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		#Local class in the file
		self.phase: Phase = Phase.EXPLORATION 
		self.search_space_type = self.model_architecture_factory.get_search_space().get_type()
		self.search_space_hash = self.model_architecture_factory.get_search_space().get_hash()
		print('Hash', self.search_space_hash)
		self.exploration_trials = exploration_trials
		self.hall_of_fame_size = hall_of_fame_size
		#From model_comunicaiton
		self.exploration_models_requests: List[ModelTrainingRequest] = list()
		self.exploration_models_completed: List[CompletedModel] = list()
		self.hall_of_fame: List[CompletedModel] = list()
		self.deep_training_models_requests: List[ModelTrainingRequest] = list()
		self.deep_training_models_completed: List[CompletedModel] = list()

	def recommend_model(self) -> ModelTrainingRequest:
		if self.phase == Phase.EXPLORATION:
			return self._recommend_model_exploration()
		elif self.phase == Phase.DEEP_TRAINING:
			return self._recommend_model_hof()

	def _recommend_model_exploration(self) -> ModelTrainingRequest:
		trial = self._create_new_trial()
		params = self.model_architecture_factory.generate_model_params(trial, self.dataset.get_input_shape())
		cad = 'Generated trial ' + str(trial.number)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if trial.should_prune():
			self._on_trial_pruned(trial)
			cad = 'Prunned trial ' + str(trial.number)
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
			return self.recommend_model()
		epochs = SP.EXPLORATION_EPOCHS
		model_training_request = ModelTrainingRequest(
			id=trial.number,
			training_type=SP.DATASET_TYPE,
			experiment_id=self.experiment_id,
			architecture=params,
			epochs=epochs,
			early_stopping_patience=SP.EXPLORATION_EARLY_STOPPING_PATIENCE,
			is_partial_training=True,
			search_space_type=self.search_space_type.value,
			search_space_hash=self.search_space_hash,
			dataset_tag=self.dataset.get_tag()
		)
		self.exploration_models_requests.append(model_training_request)
		return model_training_request

	def _recommend_model_hof(self) -> ModelTrainingRequest:
		hof_model: CompletedModel = self.hall_of_fame.pop(0)
		model_training_request: ModelTrainingRequest = hof_model.model_training_request
		model_training_request.epochs = SP.HALL_OF_FAME_EPOCHS
		model_training_request.early_stopping_patience = SP.HOF_EARLY_STOPPING_PATIENCE
		model_training_request.is_partial_training = False
		self.deep_training_models_requests.append(model_training_request)
		return model_training_request

	def should_generate(self) -> bool:
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Should generate another model?'})
		if self.phase == Phase.EXPLORATION:
			return self._should_generate_exploration()
		elif self.phase == Phase.DEEP_TRAINING:
			return self._should_generate_hof()

	def _should_generate_exploration(self) -> bool:
		cad = 'Generated exploration models' + str(len(self.exploration_models_requests)) + ' / ' + str(self.exploration_trials)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		should_generate = False
		pending_to_generate = self.exploration_trials - len(self.exploration_models_requests)
		if pending_to_generate > 0:
			should_generate = True
		return should_generate

	def _should_generate_hof(self) -> bool:
		cad = 'Generated hall of fame models ' + str(len(self.deep_training_models_requests)) + ' / ' + str(self.hall_of_fame_size)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		should_generate = False
		if len(self.deep_training_models_requests) < self.hall_of_fame_size:
			should_generate = True
		return should_generate

	def should_wait(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Should wait?'})
		print('Should wait?')
		if self.phase == Phase.EXPLORATION:
			return self._should_wait_exploration()
		elif self.phase == Phase.DEEP_TRAINING:
			return self._should_generate_hof()

	def _should_wait_exploration(self) -> bool:
		cad = 'Received exploration models ' + str(len(self.exploration_models_completed)) + ' / ' + str(self.exploration_trials)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		should_wait = True
		if len(self.exploration_models_requests) == len(self.exploration_models_completed):
			should_wait = False
		return should_wait

	def _should_wait_hof(self) -> bool:
		cad = 'Received hall of fame models ' + str(len(self.deep_training_models_requests)) + ' / ' + str(self.hall_of_fame_size)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		should_wait = True
		if len(self.deep_training_models_completed) == self.hall_of_fame_size:
			should_wait = False
		return should_wait

	def get_training_total(self)-> int:
		if self.phase == Phase.EXPLORATION:
			return self.exploration_trials
		elif self.phase == Phase.DEEP_TRAINING:
			return self.hall_of_fame_size

	def is_finished(self):
		return not self._should_wait_exploration() and not self._should_wait_hof()

	def get_best_model(self):
		#If the value of the model is the max.
		if self.search_space_type == SearchSpaceType.IMAGE:
			return self.get_best_classification_model()
		#If the value of the model is the min.
		return self.get_best_regression_model()

	def get_best_classification_model(self):
		best_model = max(self.deep_training_models_completed, key=lambda completed_model: completed_model.performance_2)
		return best_model

	def get_best_regression_model(self):
		best_model = min(self.deep_training_models_completed, key=lambda completed_model: completed_model.performance_2)
		return best_model

	def get_best_exploration_classification_model(self):
		best_model = max(self.exploration_models_completed, key=lambda completed_model: completed_model.performance)
		return best_model

	def get_best_exploration_regression_model(self):
		best_model = min(self.exploration_models_completed, key=lambda completed_model: completed_model.performance)
		return best_model

	def _create_new_trial(self) -> optuna.Trial:
		trial_id = self.storage.create_new_trial(self.study_id)
		trial = optuna.Trial(self.main_study, trial_id)
		return trial

	def _on_trial_pruned(self, trial: optuna.Trial):
		self.storage.set_trial_state(trial.number, TrialState.PRUNED)

	def _build_hall_of_fame_classification(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Building Hall Of Fame for classification problem'})
		stored_completed_models = sorted(self.exploration_models_completed, key=lambda completed_model: completed_model.performance, reverse=True)
		self.hall_of_fame = stored_completed_models[0 : self.hall_of_fame_size]
		for model in self.hall_of_fame:
			print(model)

	def _build_hall_of_fame_regression(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Building Hall Of Fame for regression problem'})
		stored_completed_models = sorted(self.exploration_models_completed, key=lambda completed_model: completed_model.performance)
		self.hall_of_fame = stored_completed_models[0 : self.hall_of_fame_size]
		for model in self.hall_of_fame:
			print(model)

	def _register_completed_model(self, model_training_response: ModelTrainingResponse):
		model_training_request = next(
			request
			for request in self.exploration_models_requests
			if request.id == model_training_response.id
		)
		performance = model_training_response.performance
		completed_model = CompletedModel(model_training_request, performance)
		self.exploration_models_completed.append(completed_model)

	def report_model_response(self, model_training_response: ModelTrainingResponse):
		cad = 'Trial ' + str(model_training_response.id) + ' reported a score of ' + str(model_training_response.performance)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if self.search_space_type == SearchSpaceType.IMAGE:
			if self.phase == Phase.EXPLORATION:
				return self._report_model_response_exploration_classification(model_training_response)
			elif self.phase == Phase.DEEP_TRAINING:
				return self._report_model_response_hof_classification(model_training_response)

		if self.search_space_type == SearchSpaceType.REGRESSION or self.search_space_type == SearchSpaceType.TIME_SERIES:
			if self.phase == Phase.EXPLORATION:
				return self._report_model_response_exploration_regression(model_training_response)
			elif self.phase == Phase.DEEP_TRAINING:
				return self._report_model_response_hof_regression(model_training_response)

	def _report_model_response_exploration_classification(self, model_training_response: ModelTrainingResponse):
		# TODO: Assert that trial_id exists in Study
		performance = model_training_response.performance
		if model_training_response.finished_epochs is True:
			performance = performance * 1.03
		self.storage.set_trial_value(model_training_response.id, model_training_response.performance)
		self.storage.set_trial_state(model_training_response.id, TrialState.COMPLETE)
		self._register_completed_model(model_training_response)
		best_trial = self.get_best_exploration_classification_model()
		cad = 'Best exploration trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if self.should_generate():
			return Action.GENERATE_MODEL
		elif self.should_wait():
			return Action.WAIT
		elif not self._should_generate_exploration() and not self._should_wait_exploration():
			self._build_hall_of_fame_classification()
			self.phase = Phase.DEEP_TRAINING
			return Action.START_NEW_PHASE

	def _report_model_response_hof_classification(self, model_training_response: ModelTrainingResponse):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received HoF model response'})
		completed_model = next(
			model
			for model in self.exploration_models_completed
			if model.model_training_request.id == model_training_response.id
		)
		completed_model.performance_2 = model_training_response.performance
		self.deep_training_models_completed.append(completed_model)
		best_trial = self.get_best_classification_model()
		cad = 'Best HoF trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance_2)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if self.should_generate():
			return Action.GENERATE_MODEL
		elif self.should_wait():
			return Action.WAIT
		elif not self._should_generate_hof() and not self._should_wait_hof():
			return Action.FINISH

	def _report_model_response_exploration_regression(self, model_training_response: ModelTrainingResponse):
		loss = model_training_response.performance
		self.storage.set_trial_value(model_training_response.id, model_training_response.performance)
		self.storage.set_trial_state(model_training_response.id, TrialState.COMPLETE)
		self._register_completed_model(model_training_response)
		best_trial = self.get_best_exploration_regression_model()
		cad = 'Best exploration trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if self.should_generate():
			return Action.GENERATE_MODEL
		elif self.should_wait():
			return Action.WAIT
		elif not self._should_generate_exploration() and not self._should_wait_exploration():
			self._build_hall_of_fame_regression()
			self.phase = Phase.DEEP_TRAINING
			return Action.START_NEW_PHASE

	def _report_model_response_hof_regression(self, model_training_response: ModelTrainingResponse):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received HoF model response'})
		completed_model = next(
			model
			for model in self.exploration_models_completed
			if model.model_training_request.id == model_training_response.id
		)
		completed_model.performance_2 = model_training_response.performance
		self.deep_training_models_completed.append(completed_model)
		best_trial = self.get_best_regression_model()
		cad = 'Best HoF trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance_2)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		if self.should_generate():
			return Action.GENERATE_MODEL
		elif self.should_wait():
			return Action.WAIT
		elif (not self._should_generate_hof() and not self._should_wait_hof()):
			return Action.FINISH


class Phase(Enum):
	EXPLORATION = 1
	DEEP_TRAINING = 2

class Action(Enum):
	GENERATE_MODEL = 1
	WAIT = 2
	START_NEW_PHASE = 3
	FINISH = 4
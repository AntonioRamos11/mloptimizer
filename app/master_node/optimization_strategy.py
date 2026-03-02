import datetime
from enum import Enum
from typing import List
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from app.common.model_communication import *
from app.common.search_space import *
from app.common.repeat_pruner import RepeatPruner
from app.common.dataset import Dataset
from system_parameters import SystemParameters as SP
from app.common.socketCommunication import *
from app.master_node.state_manager import deserialize_completed_model
import logging
import traceback
import sys
import os
import json

# Configure proper logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler("debug_strategy.log"),
        logging.StreamHandler()
    ]
)

def debug_trace(msg, obj=None, include_trace=False):
    """Enhanced debugging function that shows code location"""
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
        full_msg = f"DEBUG [{location_info}] {msg} | {obj_str}"
    else:
        full_msg = f"DEBUG [{location_info}] {msg}"
    
    # Log the message
    logging.debug(full_msg)
    
    # Add stack trace if requested
    if include_trace:
        stack = ''.join(traceback.format_stack()[:-1])
        logging.debug(f"Stack trace:\n{stack}")
    
    return full_msg

class OptimizationStrategy(object):

    def __init__(self, model_architecture_factory: ModelArchitectureFactory, dataset: Dataset, exploration_trials: int, hall_of_fame_size: int, resume_from: dict = None):
        debug_trace("Initializing OptimizationStrategy", 
                   {"exploration_trials": exploration_trials, "hall_of_fame_size": hall_of_fame_size, "resuming": resume_from is not None})
        self.model_architecture_factory:ModelArchitectureFactory = model_architecture_factory
        self.dataset: Dataset = dataset
        self.storage = optuna.storages.InMemoryStorage()

        # RepeatPruner
        try:
            self.main_study: optuna.Study = optuna.create_study(
                study_name=dataset.get_tag(), 
                storage=self.storage, 
                load_if_exists=True, 
                pruner=RepeatPruner(),
                direction='maximize', 
                sampler=TPESampler(n_ei_candidates=5000, n_startup_trials=30)
            )
            debug_trace("Created Optuna study", {"study_name": dataset.get_tag()})
        except Exception as e:
            debug_trace(f"Error creating Optuna study: {str(e)}", include_trace=True)
            raise
            
        self.study_id = 0
        
        # Check if resuming from previous state
        if resume_from:
            self.experiment_id = resume_from.get('experiment_id', dataset.get_tag() + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            
            # Handle both old format ("Phase.EXPLORATION") and new format ("EXPLORATION")
            phase_str = resume_from.get('phase', 'EXPLORATION')
            if '.' in phase_str:
                phase_str = phase_str.split('.')[-1]  # Extract "EXPLORATION" from "Phase.EXPLORATION"
            self.phase = Phase[phase_str]
            
            debug_trace("Resuming from previous state", {
                "phase": str(self.phase),
                "exploration_completed": resume_from.get('exploration_completed', 0),
                "deep_training_completed": resume_from.get('deep_training_completed', 0)
            })
        else:
            self.experiment_id = dataset.get_tag() + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.phase = Phase.EXPLORATION 
        
        self.search_space_type = self.model_architecture_factory.get_search_space().get_type()
        self.search_space_hash = self.model_architecture_factory.get_search_space().get_hash()
        debug_trace("Search space configuration", {
            "type": str(self.search_space_type),
            "hash": self.search_space_hash
        })
        
        self.exploration_trials = exploration_trials
        self.hall_of_fame_size = hall_of_fame_size
        
        # From model_communication
        self.exploration_models_requests: List[ModelTrainingRequest] = list()
        self.exploration_models_completed: List[CompletedModel] = list()
        self.hall_of_fame: List[CompletedModel] = list()
        self.deep_training_models_requests: List[ModelTrainingRequest] = list()
        self.deep_training_models_completed: List[CompletedModel] = list()
        
        # Restore from saved state if resuming
        if resume_from:
            self._restore_from_state(resume_from)
        
        # Create state tracking for debugging
        self.debug_state = {
            "phase_transitions": [],
            "models_generated": 0,
            "models_completed": 0,
            "failures": 0,
            "last_action": None
        }
        
        debug_trace("OptimizationStrategy initialized successfully")

    def _restore_from_state(self, state: dict):
        """Restore optimization state from saved state dictionary"""
        debug_trace("Restoring optimization state from saved data")
        
        # Restore exploration models
        exploration_models = state.get('exploration_models_completed', [])
        for model_data in exploration_models:
            model = deserialize_completed_model(model_data)
            if model:
                self.exploration_models_completed.append(model)
                self.exploration_models_requests.append(model.model_training_request)
                self._restore_optuna_trial(model)
                debug_trace(f"Restored exploration model: {model.model_training_request.id}")
        
        # Restore deep training models
        deep_models = state.get('deep_training_models_completed', [])
        for model_data in deep_models:
            model = deserialize_completed_model(model_data)
            if model:
                self.deep_training_models_completed.append(model)
                self.deep_training_models_requests.append(model.model_training_request)
                debug_trace(f"Restored deep training model: {model.model_training_request.id}")
        
        # Rebuild hall of fame from best models
        if self.exploration_models_completed:
            sorted_models = sorted(self.exploration_models_completed, key=lambda m: m.performance, reverse=True)
            self.hall_of_fame = sorted_models[:self.hall_of_fame_size]
            debug_trace(f"Rebuilt hall of fame with {len(self.hall_of_fame)} models")
        
        debug_trace("State restoration complete", {
            "exploration_restored": len(self.exploration_models_completed),
            "deep_training_restored": len(self.deep_training_models_completed),
            "hof_size": len(self.hall_of_fame)
        })

    def _restore_optuna_trial(self, model: CompletedModel):
        """Restore an Optuna trial from a completed model"""
        try:
            trial_id = self.storage.create_new_trial(self.study_id)
            trial = optuna.Trial(self.main_study, trial_id)
            
            # Set trial params from model architecture
            arch = model.model_training_request.architecture
            if arch and hasattr(arch, '__dict__'):
                for key, value in arch.__dict__.items():
                    if not key.startswith('_'):
                        try:
                            trial.params[key] = value
                        except:
                            pass
            
            # Set trial value (performance) - use trial's report method
            try:
                trial.report(model.performance, step=0)
                trial.should_prune()
            except Exception:
                pass
            
            # Try to set trial state via storage (older Optuna API)
            try:
                self.storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
            except AttributeError:
                # Newer Optuna versions don't have this method directly on storage
                # Try study method
                try:
                    self.main_study._storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
                except:
                    pass
            
            debug_trace(f"Restored Optuna trial {trial_id} with performance {model.performance}")
        except Exception as e:
            debug_trace(f"Error restoring Optuna trial: {e}")

    def recommend_model(self) -> ModelTrainingRequest:
        debug_trace(f"Recommending model in phase: {self.phase}")
        try:
            if self.phase == Phase.EXPLORATION:
                return self._recommend_model_exploration()
            elif self.phase == Phase.DEEP_TRAINING:
                return self._recommend_model_hof()
        except Exception as e:
            debug_trace(f"Error in recommend_model: {str(e)}", include_trace=True)
            raise

    def _recommend_model_exploration(self) -> ModelTrainingRequest:
        debug_trace("Recommending exploration model")
        try:
            trial = self._create_new_trial()
            debug_trace(f"Created new trial: {trial.number}")
            
            params = self.model_architecture_factory.generate_model_params(trial, self.dataset.get_input_shape())
            
            # Debug: Log the generated params
            debug_trace("Generated model parameters", {
                "trial": trial.number, 
                "params": str(params)[:200] if params else "None",
                "base_architecture": params.base_architecture if params and hasattr(params, 'base_architecture') else "MISSING"
            })
            
            # Check if params or base_architecture is None
            if params is None:
                debug_trace("ERROR: generate_model_params returned None!", include_trace=True)
                raise ValueError("generate_model_params returned None")
            if not hasattr(params, 'base_architecture') or params.base_architecture is None:
                debug_trace("ERROR: params.base_architecture is None!", include_trace=True)
                raise ValueError("base_architecture is None in generated params")
            
            cad = 'Generated trial ' + str(trial.number)
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
            
            if trial.should_prune():
                debug_trace(f"Trial {trial.number} pruned immediately")
                self._on_trial_pruned(trial)
                cad = 'Pruned trial ' + str(trial.number)
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
            
            debug_trace(f"Created model training request", {
                "id": model_training_request.id,
                "epochs": model_training_request.epochs,
                "is_partial_training": model_training_request.is_partial_training
            })
            
            self.exploration_models_requests.append(model_training_request)
            self.debug_state["models_generated"] += 1
            
            return model_training_request
        except Exception as e:
            debug_trace(f"Error in _recommend_model_exploration: {str(e)}", include_trace=True)
            self.debug_state["failures"] += 1
            raise

    def _recommend_model_hof(self) -> ModelTrainingRequest:
        debug_trace("Recommending HoF model", {"hall_of_fame_size": len(self.hall_of_fame)})
        try:
            if not self.hall_of_fame:
                debug_trace("WARNING: Hall of Fame is empty!", include_trace=True)
                # Return a fallback model if possible
                if self.exploration_models_completed:
                    best_model = max(self.exploration_models_completed, 
                        key=lambda m: m.performance)
                    debug_trace("Using best exploration model as fallback")
                    self.hall_of_fame = [best_model]
                else:
                    debug_trace("No exploration models completed yet - falling back to exploration phase")
                    # Fall back to exploration instead of crashing
                    self.phase = Phase.EXPLORATION
                    return self._recommend_model_exploration()
            
            # Check if the HoF model has valid architecture
            hof_model: CompletedModel = self.hall_of_fame[0]  # Peek at first model
            if hof_model.model_training_request.architecture is None:
                debug_trace("HoF model has None architecture - falling back to exploration", {
                    "model_id": hof_model.model_training_request.id
                })
                # Remove bad model from HoF
                self.hall_of_fame.pop(0)
                # Try again with remaining models or fall back to exploration
                if self.hall_of_fame:
                    return self._recommend_model_hof()
                else:
                    self.phase = Phase.EXPLORATION
                    return self._recommend_model_exploration()
            
            # Now pop and use the model
            hof_model = self.hall_of_fame.pop(0)
            debug_trace(f"Popped HoF model", {"id": hof_model.model_training_request.id, "performance": hof_model.performance})
            
            model_training_request: ModelTrainingRequest = hof_model.model_training_request
            model_training_request.epochs = SP.HALL_OF_FAME_EPOCHS
            model_training_request.early_stopping_patience = SP.HOF_EARLY_STOPPING_PATIENCE
            model_training_request.is_partial_training = False
            
            debug_trace(f"Modified HOF training request", {
                "id": model_training_request.id,
                "epochs": model_training_request.epochs,
                "early_stopping_patience": model_training_request.early_stopping_patience
            })
            
            self.deep_training_models_requests.append(model_training_request)
            self.debug_state["models_generated"] += 1
            
            return model_training_request
            
        except Exception as e:
            debug_trace(f"Error in _recommend_model_hof: {str(e)}", include_trace=True)
            self.debug_state["failures"] += 1
            raise

    def should_generate(self) -> bool:
        debug_trace(f"Checking should_generate in phase: {self.phase}")
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Should generate another model?'})
        
        try:
            if self.phase == Phase.EXPLORATION:
                return self._should_generate_exploration()
            elif self.phase == Phase.DEEP_TRAINING:
                return self._should_generate_hof()
        except Exception as e:
            debug_trace(f"Error in should_generate: {str(e)}", include_trace=True)
            return False  # Safe default

    def _should_generate_exploration(self) -> bool:
        pending_to_generate = self.exploration_trials - len(self.exploration_models_requests)
        should_generate = pending_to_generate > 0
        
        debug_trace(f"Should generate exploration?", {
            "requests": len(self.exploration_models_requests),
            "target": self.exploration_trials,
            "pending": pending_to_generate,
            "result": should_generate
        })
        
        cad = 'Generated exploration models ' + str(len(self.exploration_models_requests)) + ' / ' + str(self.exploration_trials)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        
        return should_generate

    def _should_generate_hof(self) -> bool:
        should_generate = len(self.deep_training_models_requests) < self.hall_of_fame_size
        
        debug_trace(f"Should generate HoF?", {
            "requests": len(self.deep_training_models_requests),
            "target": self.hall_of_fame_size,
            "result": should_generate
        })
        
        cad = 'Generated hall of fame models ' + str(len(self.deep_training_models_requests)) + ' / ' + str(self.hall_of_fame_size)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        
        return should_generate

    def should_wait(self):
        debug_trace(f"Checking should_wait in phase: {self.phase}")
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Should wait?'})
        
        try:
            if self.phase == Phase.EXPLORATION:
                return self._should_wait_exploration()
            elif self.phase == Phase.DEEP_TRAINING:
                # BUG FIX: This should call _should_wait_hof, not _should_generate_hof
                debug_trace("BUGFIX: Called _should_wait_hof() instead of _should_generate_hof()")
                return self._should_wait_hof()
        except Exception as e:
            debug_trace(f"Error in should_wait: {str(e)}", include_trace=True)
            return True  # Safe default is to wait

    def _should_wait_exploration(self) -> bool:
        should_wait = len(self.exploration_models_requests) != len(self.exploration_models_completed)
        
        debug_trace(f"Should wait for exploration?", {
            "requests": len(self.exploration_models_requests),
            "completed": len(self.exploration_models_completed),
            "result": should_wait
        })
        
        cad = 'Received exploration models ' + str(len(self.exploration_models_completed)) + ' / ' + str(self.exploration_trials)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        
        return should_wait

    def _should_wait_hof(self) -> bool:
        should_wait = len(self.deep_training_models_completed) != self.hall_of_fame_size
        
        debug_trace(f"Should wait for HoF?", {
            "completed": len(self.deep_training_models_completed),
            "target": self.hall_of_fame_size,
            "result": should_wait
        })
        
        cad = 'Received hall of fame models ' + str(len(self.deep_training_models_completed)) + ' / ' + str(self.hall_of_fame_size)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        
        return should_wait

    def report_model_response(self, model_training_response: ModelTrainingResponse):
        debug_trace(f"Processing model response", {
            "id": model_training_response.id,
            "performance": model_training_response.performance,
            "finished_epochs": model_training_response.finished_epochs
        })
        
        cad = 'Trial ' + str(model_training_response.id) + ' reported a score of ' + str(model_training_response.performance)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        
        try:
            action = None
            # Debug the current state
            debug_trace("Current state before processing response", {
                "phase": self.phase,
                "search_space_type": self.search_space_type,
                "exploration_completed": len(self.exploration_models_completed),
                "deep_training_completed": len(self.deep_training_models_completed)
            })
            
            if self.search_space_type == SearchSpaceType.IMAGE:
                if self.phase == Phase.EXPLORATION:
                    debug_trace("Handling exploration classification response")
                    action = self._report_model_response_exploration_classification(model_training_response)
                elif self.phase == Phase.DEEP_TRAINING:
                    debug_trace("Handling HoF classification response")
                    action = self._report_model_response_hof_classification(model_training_response)

            if self.search_space_type == SearchSpaceType.REGRESSION or self.search_space_type == SearchSpaceType.TIME_SERIES:
                if self.phase == Phase.EXPLORATION:
                    debug_trace("Handling exploration regression response")
                    action = self._report_model_response_exploration_regression(model_training_response)
                elif self.phase == Phase.DEEP_TRAINING:
                    debug_trace("Handling HoF regression response") 
                    action = self._report_model_response_hof_regression(model_training_response)
                    
            self.debug_state["last_action"] = action
            self.debug_state["models_completed"] += 1
            
            debug_trace(f"Action decided: {action}")
            return action
            
        except Exception as e:
            debug_trace(f"ERROR in report_model_response: {str(e)}", include_trace=True)
            self.debug_state["failures"] += 1
            # Return generate model as a fallback
            return Action.GENERATE_MODEL

    def _report_model_response_exploration_classification(self, model_training_response: ModelTrainingResponse):
        debug_trace("Processing exploration classification response", {
            "trial_id": model_training_response.id,
            "performance": model_training_response.performance
        })
        
        try:
            # Check if trial exists in storage
            if self.storage.get_trial(model_training_response.id) is None:
                debug_trace(f"WARNING: Trial {model_training_response.id} not found in storage", include_trace=True)
                return Action.GENERATE_MODEL
                
            # Validate performance value
            if model_training_response.performance is None:
                debug_trace(f"ERROR: Trial {model_training_response.id} reported None performance", include_trace=True)
                return Action.GENERATE_MODEL
                
            # Adjust performance for finished epochs
            performance = model_training_response.performance
            if model_training_response.finished_epochs is True:
                performance = performance * 1.03
                debug_trace(f"Adjusted performance: {performance} (1.03x bonus for finished epochs)")
                
            # Set trial results in storage
            try:
                self.storage.set_trial_state_values(model_training_response.id, TrialState.COMPLETE, (performance,))
                debug_trace(f"Trial {model_training_response.id} marked as complete with value {performance}")
            except Exception as storage_error:
                debug_trace(f"ERROR updating trial in storage: {str(storage_error)}", include_trace=True)
            
            # Register the completed model - with validation
            try:
                self._register_completed_model(model_training_response)
                debug_trace(f"Registered completed model", {"total_completed": len(self.exploration_models_completed)})
            except Exception as register_error:
                debug_trace(f"ERROR registering completed model: {str(register_error)}", include_trace=True)
                # Add the model to completed models directly as fallback
                for request in self.exploration_models_requests:
                    if request.id == model_training_response.id:
                        completed_model = CompletedModel(
                            request, 
                            performance,
                            hardware_info=model_training_response.hardware_info
                        )
                        self.exploration_models_completed.append(completed_model)
                        debug_trace("Added model via fallback method")
                        break
            
            # Safety check: ensure we have completed models before finding the best
            if not self.exploration_models_completed:
                debug_trace("ERROR: No completed models available to determine best model", include_trace=True)
                return Action.GENERATE_MODEL
                
            # Get best model
            try:
                best_trial = self.get_best_exploration_classification_model()
                debug_trace("Best exploration trial", {
                    "id": best_trial.model_training_request.id, 
                    "score": best_trial.performance
                })
                
                cad = f'Best exploration trial so far is # {best_trial.model_training_request.id} with a score of {best_trial.performance}'
                SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
            except Exception as best_model_error:
                debug_trace(f"ERROR finding best model: {str(best_model_error)}", include_trace=True)
            
            # Check conditions and decide action
            try:
                should_gen = self.should_generate()
                should_wait = self.should_wait()
                debug_trace(f"Decision state", {
                    "should_generate": should_gen, 
                    "should_wait": should_wait,
                    "exploration_generated": len(self.exploration_models_requests),
                    "exploration_completed": len(self.exploration_models_completed)
                })
                
                if should_gen:
                    debug_trace("Decided: GENERATE_MODEL")
                    return Action.GENERATE_MODEL
                elif should_wait:
                    debug_trace("Decided: WAIT")
                    return Action.WAIT
                elif not self._should_generate_exploration() and not self._should_wait_exploration():
                    debug_trace("Building Hall of Fame and transitioning to DEEP_TRAINING phase")
                    
                    # Build hall of fame with error handling
                    try:
                        self._build_hall_of_fame_classification()
                    except Exception as hof_error:
                        debug_trace(f"ERROR building hall of fame: {str(hof_error)}", include_trace=True)
                        # Fallback: manually build hall of fame with top models
                        if len(self.exploration_models_completed) > 0:
                            try:
                                debug_trace("Using fallback hall of fame building")
                                stored_models = sorted(
                                    self.exploration_models_completed, 
                                    key=lambda m: float(m.performance) if m.performance is not None else float('-inf'), 
                                    reverse=True
                                )
                                self.hall_of_fame = stored_models[:self.hall_of_fame_size]
                            except Exception as fallback_error:
                                debug_trace(f"ERROR in fallback hall of fame: {str(fallback_error)}", include_trace=True)
                    
                    # Log phase transition
                    self.debug_state["phase_transitions"].append({
                        "time": str(datetime.datetime.now()),
                        "from": "EXPLORATION",
                        "to": "DEEP_TRAINING",
                        "models_completed": len(self.exploration_models_completed),
                        "best_model": best_trial.model_training_request.id if 'best_trial' in locals() else None,
                        "best_performance": best_trial.performance if 'best_trial' in locals() else None
                    })
                    
                    self.phase = Phase.DEEP_TRAINING
                    debug_trace("Decided: START_NEW_PHASE")
                    return Action.START_NEW_PHASE
            except Exception as decision_error:
                debug_trace(f"ERROR making decision: {str(decision_error)}", include_trace=True)
                return Action.WAIT  # Safe default
                    
            # This should never happen, but add as a fallback
            debug_trace("WARNING: No clear action determined, defaulting to WAIT")
            return Action.WAIT
            
        except Exception as e:
            # Catch-all error handler with detailed information
            debug_trace(f"ERROR in exploration classification handler: {str(e)}", include_trace=True)
            return Action.WAIT  # Safe default

    def _report_model_response_hof_classification(self, model_training_response: ModelTrainingResponse):
        debug_trace("Processing HoF classification response", {
            "trial_id": model_training_response.id,
            "performance": model_training_response.performance
        })
        
        try:
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received HoF model response'})
            
            # Check if already processed (duplicate message)
            already_processed = any(
                model.model_training_request.id == model_training_response.id 
                for model in self.deep_training_models_completed
            )
            if already_processed:
                debug_trace(f"Duplicate HoF response for model {model_training_response.id} - ignoring")
                if self.should_generate():
                    return Action.GENERATE_MODEL
                elif self.should_wait():
                    return Action.WAIT
                elif not self._should_generate_hof() and not self._should_wait_hof():
                    return Action.FINISH
                return Action.WAIT
            
            # Find the model in exploration_models_completed
            completed_model = next(
                (model for model in self.exploration_models_completed 
                 if model.model_training_request.id == model_training_response.id),
                None
            )
            
            if completed_model is None:
                debug_trace(f"WARNING: Model {model_training_response.id} not found in exploration_models_completed", include_trace=True)
                debug_trace(f"Available exploration models: {[m.model_training_request.id for m in self.exploration_models_completed]}")
                debug_trace(f"Deep training completed: {[m.model_training_request.id for m in self.deep_training_models_completed]}")
                return Action.WAIT
            
            completed_model.performance_2 = model_training_response.performance
            self.deep_training_models_completed.append(completed_model)
            
            # Check if we have any completed HoF models before calling get_best
            if not self.deep_training_models_completed:
                debug_trace("ERROR: No completed HoF models yet", include_trace=True)
                return Action.WAIT
            
            best_trial = self.get_best_classification_model()
            debug_trace("Best HoF trial", {
                "id": best_trial.model_training_request.id, 
                "score": best_trial.performance_2
            })
            cad = 'Best HoF trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance_2)
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
            if self.should_generate():
                return Action.GENERATE_MODEL
            elif self.should_wait():
                return Action.WAIT
            elif not self._should_generate_hof() and not self._should_wait_hof():
                return Action.FINISH
            return Action.WAIT
        except Exception as e:
            debug_trace(f"ERROR in HoF classification handler: {str(e)}", include_trace=True)
            return Action.WAIT  # Safe default

    def _report_model_response_exploration_regression(self, model_training_response: ModelTrainingResponse):
        debug_trace("Processing exploration regression response", {
            "trial_id": model_training_response.id,
            "performance": model_training_response.performance
        })
        
        try:
            loss = model_training_response.performance
            self.storage.set_trial_state_values(model_training_response.id, TrialState.COMPLETE, (model_training_response.performance,))
            self._register_completed_model(model_training_response)
            best_trial = self.get_best_exploration_regression_model()
            debug_trace("Best exploration trial", {
                "id": best_trial.model_training_request.id, 
                "score": best_trial.performance
            })
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
        except Exception as e:
            debug_trace(f"ERROR in exploration regression handler: {str(e)}", include_trace=True)
            return Action.WAIT  # Safe default

    def _report_model_response_hof_regression(self, model_training_response: ModelTrainingResponse):
        debug_trace("Processing HoF regression response", {
            "trial_id": model_training_response.id,
            "performance": model_training_response.performance
        })
        
        try:
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received HoF model response'})
            
            # Check if already processed (duplicate message)
            already_processed = any(
                model.model_training_request.id == model_training_response.id 
                for model in self.deep_training_models_completed
            )
            if already_processed:
                debug_trace(f"Duplicate HoF response for model {model_training_response.id} - ignoring")
                if self.should_generate():
                    return Action.GENERATE_MODEL
                elif self.should_wait():
                    return Action.WAIT
                elif not self._should_generate_hof() and not self._should_wait_hof():
                    return Action.FINISH
                return Action.WAIT
            
            # Find the model in exploration_models_completed
            completed_model = next(
                (model for model in self.exploration_models_completed 
                 if model.model_training_request.id == model_training_response.id),
                None
            )
            
            if completed_model is None:
                debug_trace(f"WARNING: Model {model_training_response.id} not found in exploration_models_completed", include_trace=True)
                debug_trace(f"Available exploration models: {[m.model_training_request.id for m in self.exploration_models_completed]}")
                debug_trace(f"Deep training completed: {[m.model_training_request.id for m in self.deep_training_models_completed]}")
                return Action.WAIT
            
            completed_model.performance_2 = model_training_response.performance
            self.deep_training_models_completed.append(completed_model)
            
            # Check if we have any completed HoF models before calling get_best
            if not self.deep_training_models_completed:
                debug_trace("ERROR: No completed HoF models yet", include_trace=True)
                return Action.WAIT
            
            best_trial = self.get_best_regression_model()
            debug_trace("Best HoF trial", {
                "id": best_trial.model_training_request.id, 
                "score": best_trial.performance_2
            })
            cad = 'Best HoF trial so far is # ' + str(best_trial.model_training_request.id) + ' with a score of ' + str(best_trial.performance_2)
            SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
            if self.should_generate():
                return Action.GENERATE_MODEL
            elif self.should_wait():
                return Action.WAIT
            elif (not self._should_generate_hof() and not self._should_wait_hof()):
                return Action.FINISH
            return Action.WAIT
        except Exception as e:
            debug_trace(f"ERROR in HoF regression handler: {str(e)}", include_trace=True)
            return Action.WAIT  # Safe default

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
        """Get best exploration model with improved error handling"""
        if not self.exploration_models_completed:
            debug_trace("ERROR: No exploration models completed yet", include_trace=True)
            raise ValueError("No exploration models completed")
            
        try:
            # Filter out models with None performance
            valid_models = [m for m in self.exploration_models_completed if m.performance is not None]
            
            if not valid_models:
                debug_trace("ERROR: No models with valid performance values", include_trace=True)
                raise ValueError("No models with valid performance")
                
            best_model = max(valid_models, key=lambda completed_model: completed_model.performance)
            return best_model
            
        except Exception as e:
            debug_trace(f"ERROR finding best exploration model: {str(e)}", include_trace=True)
            raise

    def get_best_exploration_regression_model(self):
        best_model = min(self.exploration_models_completed, key=lambda completed_model: completed_model.performance)
        return best_model

    def _create_new_trial(self) -> optuna.Trial:
        trial_id = self.storage.create_new_trial(self.study_id)
        trial = optuna.Trial(self.main_study, trial_id)
        return trial

    def _on_trial_pruned(self, trial: optuna.Trial):
        try:
            self.storage.set_trial_state(trial.number, optuna.trial.TrialState.PRUNED)
        except AttributeError:
            # Newer Optuna versions may not have this method
            try:
                self.main_study._storage.set_trial_state(trial.number, optuna.trial.TrialState.PRUNED)
            except:
                pass

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
        """Register a completed model with improved error handling"""
        try:
            # Find matching request
            matching_request = None
            for request in self.exploration_models_requests:
                if request.id == model_training_response.id:
                    matching_request = request
                    break
                    
            if matching_request is None:
                debug_trace(f"ERROR: No matching request found for model {model_training_response.id}", include_trace=True)
                raise ValueError(f"No matching request for model {model_training_response.id}")
                
            performance = model_training_response.performance
            if performance is None:
                debug_trace(f"WARNING: Model {model_training_response.id} has None performance, using 0 as fallback")
                performance = 0.0
                
            completed_model = CompletedModel(
                matching_request, 
                performance,
                hardware_info=model_training_response.hardware_info
            )
            self.exploration_models_completed.append(completed_model)
            debug_trace(f"Successfully registered model {model_training_response.id} with performance {performance}")
            
        except Exception as e:
            debug_trace(f"ERROR in _register_completed_model: {str(e)}", include_trace=True)
            raise


class Phase(Enum):
    EXPLORATION = 1
    DEEP_TRAINING = 2

class Action(Enum):
    GENERATE_MODEL = 1
    WAIT = 2
    START_NEW_PHASE = 3
    FINISH = 4
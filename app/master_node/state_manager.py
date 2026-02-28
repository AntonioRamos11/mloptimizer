import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from app.common.model_communication import CompletedModel, ModelTrainingRequest
from app.common.search_space import ImageModelArchitectureParameters, RegressionModelArchitectureParameters, TimeSeriesModelArchitectureParameters


class StateManager:
    """Manages optimization state persistence for resume functionality"""
    
    def __init__(self, results_dir: str = 'results_multi'):
        self.results_dir = results_dir
        
    def get_latest_state_file(self) -> Optional[str]:
        """Find the most recent state file for auto-resume"""
        if not os.path.exists(self.results_dir):
            return None
            
        state_files = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('_state.json'):
                filepath = os.path.join(self.results_dir, filename)
                state_files.append((filepath, os.path.getmtime(filepath)))
        
        if not state_files:
            return None
            
        # Return most recent state file
        state_files.sort(key=lambda x: x[1], reverse=True)
        return state_files[0][0]
    
    def load_state(self, filepath: str = None) -> Optional[Dict[str, Any]]:
        """Load optimization state from JSON file"""
        if filepath is None:
            filepath = self.get_latest_state_file()
            
        if filepath is None or not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            print(f"[StateManager] Loaded state from: {filepath}")
            print(f"[StateManager]   Phase: {state.get('phase', 'UNKNOWN')}")
            print(f"[StateManager]   Exploration completed: {state.get('exploration_completed', 0)}")
            print(f"[StateManager]   Deep training completed: {state.get('deep_training_completed', 0)}")
            return state
        except Exception as e:
            print(f"[StateManager] Error loading state: {e}")
            return None
    
    def save_state(self, state: Dict[str, Any], experiment_id: str = None) -> str:
        """Save optimization state to JSON file"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        if experiment_id is None:
            experiment_id = state.get('experiment_id', 'unknown')
            
        filename = f"{experiment_id}_state.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            print(f"[StateManager] Saved state to: {filepath}")
            return filepath
        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")
            return None
    
    def has_previous_state(self) -> bool:
        """Check if there's a previous state to resume from"""
        state_file = self.get_latest_state_file()
        return state_file is not None
    
    def delete_state(self, experiment_id: str = None) -> bool:
        """Delete state file (call when starting fresh)"""
        if experiment_id:
            filepath = os.path.join(self.results_dir, f"{experiment_id}_state.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        return False


def deserialize_completed_model(data: Dict[str, Any]) -> Optional[CompletedModel]:
    """Deserialize a CompletedModel from JSON data"""
    try:
        mtr_data = data.get('model_training_request', {})
        
        search_space_type = mtr_data.get('search_space_type', 1)
        arch_data = mtr_data.get('architecture')
        
        if search_space_type == 1:
            architecture = ImageModelArchitectureParameters.from_dict(arch_data) if arch_data else None
        elif search_space_type == 2:
            architecture = RegressionModelArchitectureParameters.from_dict(arch_data) if arch_data else None
        elif search_space_type == 3:
            architecture = TimeSeriesModelArchitectureParameters.from_dict(arch_data) if arch_data else None
        else:
            architecture = None
        
        model_training_request = ModelTrainingRequest(
            id=mtr_data.get('id', 0),
            training_type=mtr_data.get('training_type', 1),
            experiment_id=mtr_data.get('experiment_id', ''),
            architecture=architecture,
            epochs=mtr_data.get('epochs', 10),
            early_stopping_patience=mtr_data.get('early_stopping_patience', 5),
            is_partial_training=mtr_data.get('is_partial_training', True),
            search_space_type=mtr_data.get('search_space_type', '1'),
            search_space_hash=mtr_data.get('search_space_hash', ''),
            dataset_tag=mtr_data.get('dataset_tag', '')
        )
        
        return CompletedModel(
            model_training_request=model_training_request,
            performance=data.get('performance', -1),
            performance_2=data.get('performance_2', -1),
            hardware_info=data.get('hardware_info')
        )
    except Exception as e:
        print(f"[StateManager] Error deserializing model: {e}")
        return None


def serialize_completed_model(model) -> Dict[str, Any]:
    """Serialize a CompletedModel for JSON storage"""
    return {
        'model_training_request': {
            'id': model.model_training_request.id,
            'training_type': model.model_training_request.training_type,
            'experiment_id': model.model_training_request.experiment_id,
            'epochs': model.model_training_request.epochs,
            'early_stopping_patience': model.model_training_request.early_stopping_patience,
            'is_partial_training': model.model_training_request.is_partial_training,
            'search_space_type': model.model_training_request.search_space_type,
            'search_space_hash': model.model_training_request.search_space_hash,
            'dataset_tag': model.model_training_request.dataset_tag,
            'architecture': str(model.model_training_request.architecture) if model.model_training_request.architecture else None
        },
        'performance': model.performance,
        'performance_2': getattr(model, 'performance_2', None),
        'hardware_info': getattr(model, 'hardware_info', None)
    }


def create_state_from_strategy(strategy) -> Dict[str, Any]:
    """Create state dictionary from OptimizationStrategy"""
    state = {
        'experiment_id': strategy.experiment_id,
        'phase': str(strategy.phase),
        'search_space_type': str(strategy.search_space_type),
        'search_space_hash': strategy.search_space_hash,
        'exploration_trials': strategy.exploration_trials,
        'hall_of_fame_size': strategy.hall_of_fame_size,
        'exploration_completed': len(strategy.exploration_models_completed),
        'exploration_requested': len(strategy.exploration_models_requests),
        'deep_training_completed': len(strategy.deep_training_models_completed),
        'deep_training_requested': len(strategy.deep_training_models_requests),
        'hall_of_fame_size_current': len(strategy.hall_of_fame),
        'timestamp': datetime.now().isoformat()
    }
    
    # Serialize completed models
    state['exploration_models_completed'] = [
        serialize_completed_model(m) for m in strategy.exploration_models_completed
    ]
    
    state['deep_training_models_completed'] = [
        serialize_completed_model(m) for m in strategy.deep_training_models_completed
    ]
    
    # Save best model info
    if strategy.exploration_models_completed:
        best = max(strategy.exploration_models_completed, key=lambda m: m.performance)
        state['best_exploration_model_id'] = best.model_training_request.id
        state['best_exploration_performance'] = best.performance
    
    if strategy.deep_training_models_completed:
        try:
            best_hof = max(strategy.deep_training_models_completed, key=lambda m: m.performance_2)
            state['best_hof_model_id'] = best_hof.model_training_request.id
            state['best_hof_performance'] = best_hof.performance_2
        except:
            pass
    
    return state

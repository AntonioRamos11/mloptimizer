#!/usr/bin/env python3

import os
import sys
import asyncio
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler("test_run.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Simplified test parameters
class TestParameters:
    DATASET_NAME = "synthetic"
    DATASET_BATCH_SIZE = 16
    EXPLORATION_SIZE = 3
    HALL_OF_FAME_SIZE = 2
    EXPLORATION_EPOCHS = 2
    HALL_OF_FAME_EPOCHS = 3
    EXPLORATION_EARLY_STOPPING_PATIENCE = 1
    HOF_EARLY_STOPPING_PATIENCE = 1
    DATASET_TYPE = "classification"
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "adam"

import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import necessary components
from app.common.search_space import SearchSpaceType
from app.common.model_factory import ModelArchitectureFactory, ModelParameters
from app.datasets.synthetic_dataset import SyntheticDataset

# Mock classes for OptimizationStrategy
class Phase:
    EXPLORATION = 1
    DEEP_TRAINING = 2

class Action:
    GENERATE_MODEL = 1
    WAIT = 2
    START_NEW_PHASE = 3
    FINISH = 4

# Mock classes for model training requests
@dataclass
class ModelTrainingRequest:
    id: int
    architecture: ModelParameters
    training_type: str = "classification"
    experiment_id: str = "test_experiment"
    epochs: int = 5
    early_stopping_patience: int = 2
    is_partial_training: bool = True
    search_space_type: str = "image"
    search_space_hash: str = "1234"
    dataset_tag: str = "synthetic"

@dataclass
class ModelTrainingResponse:
    id: int
    performance: float
    finished_epochs: bool = True

@dataclass
class CompletedModel:
    model_training_request: ModelTrainingRequest
    performance: float
    performance_2: float = 0.0

class MockOptimizationStrategy:
    """Mock implementation of OptimizationStrategy for testing"""
    
    def __init__(self, model_architecture_factory, dataset, exploration_trials, hall_of_fame_size):
        self.model_architecture_factory = model_architecture_factory
        self.dataset = dataset
        self.exploration_trials = exploration_trials
        self.hall_of_fame_size = hall_of_fame_size
        self.phase = Phase.EXPLORATION
        self.search_space_type = self.model_architecture_factory.get_search_space().get_type()
        self.search_space_hash = str(self.model_architecture_factory.get_search_space().get_hash())
        
        # Storage for models
        self.exploration_models_requests = []
        self.exploration_models_completed = []
        self.deep_training_models_requests = []
        self.deep_training_models_completed = []
        self.hall_of_fame = []
        
        # Mock trial counter
        self.trial_counter = 0
        
    def recommend_model(self):
        """Generate a new model recommendation"""
        if self.phase == Phase.EXPLORATION:
            return self._recommend_model_exploration()
        else:
            return self._recommend_model_hof()
            
    def _recommend_model_exploration(self):
        """Create an exploration model request"""
        self.trial_counter += 1
        trial_id = self.trial_counter
        
        # Create a mock trial object for parameter generation
        class MockTrial:
            def suggest_int(self, name, low, high, step=1):
                return np.random.randint(low, high+1)
                
            def suggest_float(self, name, low, high, step=None):
                return np.random.uniform(low, high)
        
        mock_trial = MockTrial()
        params = self.model_architecture_factory.generate_model_params(mock_trial, self.dataset.get_input_shape())
        
        # Create model request
        request = ModelTrainingRequest(
            id=trial_id,
            training_type=TestParameters.DATASET_TYPE,
            architecture=params,
            epochs=TestParameters.EXPLORATION_EPOCHS,
            early_stopping_patience=TestParameters.EXPLORATION_EARLY_STOPPING_PATIENCE,
            is_partial_training=True,
            search_space_type=self.search_space_type.value,
            search_space_hash=self.search_space_hash,
            dataset_tag=self.dataset.get_tag()
        )
        
        self.exploration_models_requests.append(request)
        return request
        
    def _recommend_model_hof(self):
        """Create a hall of fame model request"""
        if not self.hall_of_fame:
            raise ValueError("Hall of fame is empty")
            
        model = self.hall_of_fame.pop(0)
        request = model.model_training_request
        request.epochs = TestParameters.HALL_OF_FAME_EPOCHS
        request.early_stopping_patience = TestParameters.HOF_EARLY_STOPPING_PATIENCE
        request.is_partial_training = False
        
        self.deep_training_models_requests.append(request)
        return request
    
    def report_model_response(self, response):
        """Process a model training response"""
        if self.phase == Phase.EXPLORATION:
            self._register_completed_model(response)
            
            # Check if we should transition to deep training
            if len(self.exploration_models_completed) >= self.exploration_trials:
                self._build_hall_of_fame()
                self.phase = Phase.DEEP_TRAINING
                return Action.START_NEW_PHASE
            elif len(self.exploration_models_requests) < self.exploration_trials:
                return Action.GENERATE_MODEL
            else:
                return Action.WAIT
        else:
            # Deep training phase
            self._register_completed_hof_model(response)
            
            if len(self.deep_training_models_completed) >= self.hall_of_fame_size:
                return Action.FINISH
            elif len(self.deep_training_models_requests) < self.hall_of_fame_size:
                return Action.GENERATE_MODEL
            else:
                return Action.WAIT
    
    def _register_completed_model(self, response):
        """Register a completed model"""
        # Find the corresponding request
        request = next((req for req in self.exploration_models_requests if req.id == response.id), None)
        if not request:
            raise ValueError(f"No request found for response id {response.id}")
            
        # Create completed model
        model = CompletedModel(request, response.performance)
        self.exploration_models_completed.append(model)
        
    def _register_completed_hof_model(self, response):
        """Register a completed HoF model"""
        # Find the corresponding request
        request = next((req for req in self.deep_training_models_requests if req.id == response.id), None)
        if not request:
            raise ValueError(f"No request found for response id {response.id}")
            
        # Find the original model in completed models
        model = next((model for model in self.exploration_models_completed if model.model_training_request.id == response.id), None)
        if not model:
            raise ValueError(f"No exploration model found for id {response.id}")
            
        # Update performance and add to deep training completed
        model.performance_2 = response.performance
        self.deep_training_models_completed.append(model)
    
    def _build_hall_of_fame(self):
        """Build the hall of fame from exploration models"""
        # Sort by performance (assuming higher is better)
        sorted_models = sorted(self.exploration_models_completed, 
                               key=lambda model: model.performance, 
                               reverse=True)
        
        # Take top models for hall of fame
        self.hall_of_fame = sorted_models[:self.hall_of_fame_size]
    
    def get_best_model(self):
        """Get the best model from deep training"""
        if not self.deep_training_models_completed:
            raise ValueError("No completed deep training models")
            
        return max(self.deep_training_models_completed, 
                   key=lambda model: model.performance_2)

class TestOptimizationJob:
    """Mock OptimizationJob for testing"""
    
    def __init__(self, dataset, model_architecture_factory, exploration_trials, hall_of_fame_size):
        self.dataset = dataset
        self.model_architecture_factory = model_architecture_factory
        self.exploration_trials = exploration_trials
        self.hall_of_fame_size = hall_of_fame_size
        self.logger = logging.getLogger(__name__)
        self.strategy = MockOptimizationStrategy(
            model_architecture_factory,
            dataset,
            exploration_trials,
            hall_of_fame_size
        )
        
    async def run(self):
        """Run the optimization loop"""
        self.logger.info("Starting optimization run")
        
        # Initial models
        for _ in range(min(2, self.exploration_trials)):
            await self._generate_model()
        
        # Main optimization loop
        while True:
            # Mock training - in real case this would come from workers
            for request in self.strategy.exploration_models_requests:
                if request.id not in [m.model_training_request.id for m in self.strategy.exploration_models_completed]:                    # Simulate some work
                    await asyncio.sleep(0.2)
                    
                    # Mock response with random performance
                    performance = np.random.uniform(0.6, 0.95)
                    response = ModelTrainingResponse(
                        id=request.id,
                        performance=performance,
                        finished_epochs=True
                    )
                    
                    # Process response
                    action = self.strategy.report_model_response(response)
                    self.logger.info(f"Model {request.id} completed with performance {performance:.4f}, action: {action}")
                    
                    if action == Action.GENERATE_MODEL:
                        await self._generate_model()
                    elif action == Action.START_NEW_PHASE:
                        self.logger.info("Starting deep training phase")
                        await self._generate_model()  # Generate first HoF model
                    elif action == Action.FINISH:
                        best_model = self.strategy.get_best_model()
                        self.logger.info(f"Optimization finished. Best model: {best_model.model_training_request.id} with performance {best_model.performance_2:.4f}")
                        return best_model
            
            # If we're in deep training phase, process HoF models
            if self.strategy.phase == Phase.DEEP_TRAINING:
                for request in self.strategy.deep_training_models_requests:
                    if request.id not in [m.model_training_request.id for m in self.strategy.deep_training_models_completed]:
                        # Simulate some work
                        await asyncio.sleep(0.3)
                        
                        # Mock response with random performance (slightly better than exploration)
                        performance = np.random.uniform(0.7, 0.98)
                        response = ModelTrainingResponse(
                            id=request.id,
                            performance=performance,
                            finished_epochs=True
                        )
                        
                        # Process response
                        action = self.strategy.report_model_response(response)
                        self.logger.info(f"HoF Model {request.id} completed with performance {performance:.4f}, action: {action}")
                        
                        if action == Action.GENERATE_MODEL:
                            await self._generate_model()
                        elif action == Action.FINISH:
                            best_model = self.strategy.get_best_model()
                            self.logger.info(f"Optimization finished. Best model: {best_model.model_training_request.id} with performance {best_model.performance_2:.4f}")
                            return best_model
            
            # Short delay to avoid tight loop
            await asyncio.sleep(0.1)
    
    async def _generate_model(self):
        """Generate a new model"""
        request = self.strategy.recommend_model()
        self.logger.info(f"Generated model {request.id}")
        return request

async def run_test():
    """Run a test optimization"""
    # Create dataset
    dataset = SyntheticDataset()
    logging.info(f"Created synthetic dataset with shape {dataset.get_input_shape()} and {dataset.get_classes_count()} classes")
    
    # Create model factory
    factory = ModelArchitectureFactory.create(SearchSpaceType.IMAGE)
    logging.info(f"Created model factory with search space type: {SearchSpaceType.IMAGE}")
    
    # Create optimization job
    job = TestOptimizationJob(
        dataset=dataset,
        model_architecture_factory=factory,
        exploration_trials=TestParameters.EXPLORATION_SIZE,
        hall_of_fame_size=TestParameters.HALL_OF_FAME_SIZE
    )
    
    # Run optimization
    start_time = datetime.now()
    try:
        best_model = await job.run()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logging.info(f"Test completed in {duration:.2f} seconds")
        logging.info(f"Best model: {best_model.model_training_request.id} with performance {best_model.performance_2:.4f}")
        
        return best_model, duration
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    print("Starting test optimization...")
    
    # Create necessary directories
    os.makedirs("app/common", exist_ok=True)
    os.makedirs("app/datasets", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run test
    best_model, duration = asyncio.run(run_test())
    
    print("\n" + "="*50)
    print(f"TEST COMPLETE - Duration: {duration:.2f} seconds")
    print(f"Best model ID: {best_model.model_training_request.id}")
    print(f"Best model performance: {best_model.performance_2:.4f}")
    print("="*50)
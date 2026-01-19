# backend/hyperopt_runner.py
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Any
import itertools
import random

from services.trainer import start_training
from services.ws_broker import publish

class HyperparameterOptimizer:
    def __init__(self, run_id: str, base_config: Dict, search_space: Dict):
        self.run_id = run_id
        self.base_config = base_config
        self.search_space = search_space
        self.results = []
        self.best_config = None
        self.best_score = float('-inf')
        self.current_trial = 0
        self.total_trials = 0
        
    def grid_search(self, max_trials: int = 20):
        """Perform grid search over hyperparameter space"""
        # Generate all combinations
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit trials if too many combinations
        if len(all_combinations) > max_trials:
            all_combinations = random.sample(all_combinations, max_trials)
            
        self.total_trials = len(all_combinations)
        
        publish(self.run_id, {
            "event": "hyperopt_started",
            "total_trials": self.total_trials,
            "search_type": "grid_search"
        })
        
        for i, combination in enumerate(all_combinations):
            self.current_trial = i + 1
            
            # Create config for this trial
            trial_config = self.base_config.copy()
            for param_name, value in zip(param_names, combination):
                trial_config[param_name] = value
                
            # Run training
            trial_id = f"{self.run_id}_trial_{i+1}"
            score = self._run_trial(trial_id, trial_config)
            
            # Update best if better
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                
            # Emit progress
            publish(self.run_id, {
                "event": "hyperopt_progress",
                "trial": self.current_trial,
                "total_trials": self.total_trials,
                "current_score": score,
                "best_score": self.best_score,
                "best_config": self.best_config
            })
            
        publish(self.run_id, {
            "event": "hyperopt_completed",
            "best_score": self.best_score,
            "best_config": self.best_config,
            "total_trials": self.total_trials
        })
        
        return self.best_config, self.best_score
    
    def random_search(self, max_trials: int = 20):
        """Perform random search over hyperparameter space"""
        self.total_trials = max_trials
        
        publish(self.run_id, {
            "event": "hyperopt_started", 
            "total_trials": self.total_trials,
            "search_type": "random_search"
        })
        
        for i in range(max_trials):
            self.current_trial = i + 1
            
            # Sample random configuration
            trial_config = self.base_config.copy()
            for param_name, values in self.search_space.items():
                trial_config[param_name] = random.choice(values)
                
            # Run training
            trial_id = f"{self.run_id}_trial_{i+1}"
            score = self._run_trial(trial_id, trial_config)
            
            # Update best if better
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                
            # Emit progress
            publish(self.run_id, {
                "event": "hyperopt_progress",
                "trial": self.current_trial,
                "total_trials": self.total_trials,
                "current_score": score,
                "best_score": self.best_score,
                "best_config": self.best_config
            })
            
        publish(self.run_id, {
            "event": "hyperopt_completed",
            "best_score": self.best_score,
            "best_config": self.best_config,
            "total_trials": self.total_trials
        })
        
        return self.best_config, self.best_score
    
    def _run_trial(self, trial_id: str, config: Dict) -> float:
        """Run a single training trial and return validation score"""
        try:
            # Run training (simplified version)
            _train_task(trial_id, config)
            
            # Load results
            trial_dir = Path("runs") / trial_id
            metrics_file = trial_dir / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                # Get best validation accuracy as score
                val_accs = [m.get("val_accuracy", 0) for m in metrics if m.get("val_accuracy")]
                score = max(val_accs) if val_accs else 0
                
                # Store result
                self.results.append({
                    "trial_id": trial_id,
                    "config": config,
                    "score": score,
                    "metrics": metrics[-1] if metrics else {}
                })
                
                return score
            else:
                return 0
                
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")
            return 0

def start_hyperparameter_optimization(run_id: str, config: Dict):
    """Start hyperparameter optimization in background thread"""
    def _hyperopt_task():
        try:
            base_config = config["base_config"]
            search_space = config["search_space"]
            search_type = config.get("search_type", "random")
            max_trials = config.get("max_trials", 10)
            
            optimizer = HyperparameterOptimizer(run_id, base_config, search_space)
            
            if search_type == "grid":
                optimizer.grid_search(max_trials)
            else:
                optimizer.random_search(max_trials)
                
        except Exception as e:
            publish(run_id, {
                "event": "hyperopt_error",
                "error": str(e)
            })
    
    thread = threading.Thread(target=_hyperopt_task, daemon=True)
    thread.start()

# Predefined search spaces for common scenarios
SEARCH_SPACES = {
    "classification_basic": {
        "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
        "batch_size": [8, 16, 32],
        "weight_decay": [0.0, 0.01, 0.1],
        "epochs": [3, 5, 8]
    },
    "classification_advanced": {
        "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4],
        "batch_size": [4, 8, 16, 32, 64],
        "weight_decay": [0.0, 0.001, 0.01, 0.1, 0.2],
        "epochs": [3, 5, 8, 10, 15],
        "warmup_steps": [0, 100, 500, 1000]
    },
    "small_dataset": {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "batch_size": [4, 8, 16],
        "weight_decay": [0.01, 0.1],
        "epochs": [5, 8, 10, 15]
    },
    "large_dataset": {
        "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
        "batch_size": [16, 32, 64, 128],
        "weight_decay": [0.0, 0.01],
        "epochs": [2, 3, 5]
    }
}
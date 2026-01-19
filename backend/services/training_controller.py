"""
Training Control Manager - Handles pause/resume and live parameter updates
"""
import threading
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrainingController:
    """
    Manages training state and live parameter updates
    """
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        
        # Threading control
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start in running state
        
        # State
        self.is_paused = False
        self.trainer = None  # Will be set by trainer
        
        logger.info(f"[{job_id}] TrainingController initialized")
    
    def pause(self):
        """Pause training"""
        if not self.is_paused:
            self.pause_event.clear()
            self.is_paused = True
            logger.info(f"[{self.job_id}] Training PAUSED")
            return {"status": "paused"}
        return {"status": "already_paused"}
    
    def resume(self):
        """Resume training"""
        if self.is_paused:
            self.pause_event.set()
            self.is_paused = False
            logger.info(f"[{self.job_id}] Training RESUMED")
            return {"status": "resumed"}
        return {"status": "already_running"}
    
    def wait_if_paused(self):
        """
        Block execution if paused. Call this in training loop.
        """
        self.pause_event.wait()
    
    def update_hyperparameters(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hot-swap hyperparameters in the running trainer
        
        Args:
            updates: Dictionary of parameter updates
                e.g., {"learning_rate": 5e-5, "weight_decay": 0.02}
        
        Returns:
            Status dictionary with applied changes
        """
        if not self.trainer:
            return {"error": "Trainer not initialized"}
        
        applied = {}
        errors = []
        
        try:
            # Access optimizer param groups
            for param_group in self.trainer.optimizer.param_groups:
                
                # Update learning rate
                if 'learning_rate' in updates:
                    new_lr = float(updates['learning_rate'])
                    old_lr = param_group['lr']
                    param_group['lr'] = new_lr
                    applied['learning_rate'] = {'old': old_lr, 'new': new_lr}
                    logger.info(f"[{self.job_id}] LR updated: {old_lr} → {new_lr}")
                
                # Update weight decay
                if 'weight_decay' in updates:
                    new_wd = float(updates['weight_decay'])
                    old_wd = param_group.get('weight_decay', 0.0)
                    param_group['weight_decay'] = new_wd
                    applied['weight_decay'] = {'old': old_wd, 'new': new_wd}
                    logger.info(f"[{self.job_id}] Weight decay updated: {old_wd} → {new_wd}")
            
            # Note: Other params like dropout, batch_size require trainer restart
            unsupported = set(updates.keys()) - {'learning_rate', 'weight_decay'}
            if unsupported:
                errors.append(f"Unsupported hot-swap params: {list(unsupported)}")
                logger.warning(f"[{self.job_id}] Cannot hot-swap: {unsupported}")
            
            return {
                "status": "success" if applied else "no_changes",
                "applied": applied,
                "errors": errors if errors else None
            }
            
        except Exception as e:
            logger.error(f"[{self.job_id}] Failed to update hyperparameters: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training control status"""
        return {
            "is_paused": self.is_paused,
            "can_update_params": self.trainer is not None
        }


# Global registry of active controllers
_active_controllers: Dict[str, TrainingController] = {}


def get_controller(job_id: str) -> Optional[TrainingController]:
    """Get training controller for a job"""
    return _active_controllers.get(job_id)


def create_controller(job_id: str) -> TrainingController:
    """Create and register a new training controller"""
    controller = TrainingController(job_id)
    _active_controllers[job_id] = controller
    return controller


def remove_controller(job_id: str):
    """Remove controller when training completes"""
    if job_id in _active_controllers:
        del _active_controllers[job_id]
        logger.info(f"[{job_id}] Controller removed")

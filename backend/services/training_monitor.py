"""
AI Training Monitor - Real-time metric analysis and insight generation
Detects overfitting, divergence, and stagnation during training
"""
import math
import time
from typing import List, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitors training metrics in real-time and generates AI insights
    """
    
    def __init__(self, socket_manager, job_id: str, window_size: int = 10):
        """
        Args:
            socket_manager: WebSocket manager for broadcasting insights
            job_id: Training job identifier
            window_size: Number of recent steps to analyze
        """
        self.socket_manager = socket_manager
        self.job_id = job_id
        self.window_size = window_size
        
        # Metric history
        self.train_losses: deque = deque(maxlen=window_size)
        self.val_losses: deque = deque(maxlen=window_size)
        self.learning_rates: deque = deque(maxlen=window_size)
        self.grad_norms: deque = deque(maxlen=window_size)
        
        # Detection state
        self.last_insight_time = {}
        self.insight_cooldown = 60  # Seconds between similar insights
        self.stagnation_counter = 0
        
    async def analyze_step(
        self,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None
    ):
        """
        Analyze a single training step and send insights if needed
        """
        # Update history
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        
        # Run detectors
        await self._detect_divergence(step, train_loss)
        await self._detect_overfitting(step)
        await self._detect_stagnation(step)
        await self._detect_gradient_issues(step, grad_norm)
    
    async def _detect_divergence(self, step: int, loss: Optional[float]):
        """
        Detect if model is diverging (NaN, Inf, or exploding loss)
        """
        if loss is None:
            return
        
        # Check for NaN or Inf
        if math.isnan(loss) or math.isinf(loss):
            await self._send_insight(
                'error',
                '🚨 Model Diverging: NaN or Inf detected!',
                'The model has diverged. Training cannot continue.',
                'Restart with lower learning rate (try 1e-6)'
            )
            return
        
        # Check for exploding loss
        if loss > 10.0:
            await self._send_insight(
                'error',
                f'🚨 Exploding Loss: {loss:.2f}',
                'Loss is abnormally high and increasing',
                'Reduce learning rate by 50% or enable gradient clipping'
            )
            return
        
        # Check for rapid increase
        if len(self.train_losses) >= 3:
            recent_losses = list(self.train_losses)[-3:]
            if all(recent_losses[i] < recent_losses[i+1] * 0.8 for i in range(len(recent_losses)-1)):
                await self._send_insight(
                    'warning',
                    '⚠️ Loss increasing rapidly',
                    'Training loss has grown significantly in the last few steps',
                    'Consider reducing learning rate or checking data quality'
                )
    
    async def _detect_overfitting(self, step: int):
        """
        Detect overfitting: train loss decreasing but val loss increasing
        """
        if len(self.train_losses) < 5 or len(self.val_losses) < 5:
            return
        
        train_slope = self._calculate_slope(list(self.train_losses))
        val_slope = self._calculate_slope(list(self.val_losses))
        
        # Overfitting: train↓ but val↑
        if train_slope < -0.01 and val_slope > 0.01:
            await self._send_insight(
                'warning',
                '📉 Overfitting Detected',
                'Training loss is decreasing while validation loss is increasing',
                'Increase dropout, weight_decay, or add more training data'
            )
    
    async def _detect_stagnation(self, step: int):
        """
        Detect if learning has plateaued
        """
        if len(self.train_losses) < 5:
            self.stagnation_counter = 0
            return
        
        # Calculate recent loss changes
        recent_losses = list(self.train_losses)[-5:]
        changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
        avg_change = sum(changes) / len(changes)
        
        if avg_change < 0.001:
            self.stagnation_counter += 1
            
            if self.stagnation_counter >= 5:  # 5 consecutive stagnant steps
                await self._send_insight(
                    'suggestion',
                    '📊 Learning Plateau Detected',
                    f'Loss has barely changed (Δ={avg_change:.6f}) for {self.stagnation_counter} steps',
                    'Try adjusting the learning rate scheduler or increasing warmup steps'
                )
                self.stagnation_counter = 0  # Reset after sending
        else:
            self.stagnation_counter = 0
    
    async def _detect_gradient_issues(self, step: int, grad_norm: Optional[float]):
        """
        Detect gradient-related issues
        """
        if grad_norm is None:
            return
        
        self.grad_norms.append(grad_norm)
        
        # Vanishing gradients
        if grad_norm < 1e-6:
            await self._send_insight(
                'warning',
                '🔻 Vanishing Gradients',
                f'Gradient norm is extremely small: {grad_norm:.2e}',
                'Check learning rate or model architecture. Consider gradient clipping.'
            )
        
        # Exploding gradients
        if grad_norm > 100.0:
            await self._send_insight(
                'warning',
                '🔺 Exploding Gradients',
                f'Gradient norm is very large: {grad_norm:.2f}',
                'Enable gradient clipping (max_grad_norm=1.0)'
            )
    
    def _calculate_slope(self, values: List[float]) -> float:
        """
        Calculate the slope of a list of values using linear regression
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # Calculate slope using least squares
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    async def _send_insight(
        self,
        level: str,
        message: str,
        details: str,
        action: Optional[str] = None
    ):
        """
        Send an insight to the frontend via WebSocket
        Args:
            level: 'info', 'warning', 'suggestion', 'error'
            message: Main insight message
            details: Additional context
            action: Suggested action
        """
        # Cooldown check to avoid spamming
        insight_key = f"{level}:{message}"
        current_time = time.time()
        
        if insight_key in self.last_insight_time:
            time_since_last = current_time - self.last_insight_time[insight_key]
            if time_since_last < self.insight_cooldown:
                return  # Skip this insight
        
        self.last_insight_time[insight_key] = current_time
        
        # Send insight
        insight = {
            'type': 'insight',
            'level': level,
            'message': message,
            'details': details,
            'action': action,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        try:
            await self.socket_manager.broadcast_json(self.job_id, insight)
            logger.info(f"[{self.job_id}] AI Insight ({level}): {message}")
        except Exception as e:
            logger.error(f"Failed to send insight: {e}")
    
    async def send_hardware_metrics(
        self,
        gpu_util: float = 0.0,
        gpu_vram_used: float = 0.0,
        gpu_vram_total: float = 16.0,
        cpu_percent: float = 0.0,
        ram_used: float = 0.0,
        ram_total: float = 16.0
    ):
        """
        Send hardware metrics to frontend
        """
        hardware = {
            'type': 'hardware',
            'gpu_util': gpu_util,
            'gpu_vram_used': gpu_vram_used,
            'gpu_vram_total': gpu_vram_total,
            'cpu_percent': cpu_percent,
            'ram_used': ram_used,
            'ram_total': ram_total
        }
        
        try:
            await self.socket_manager.broadcast_json(self.job_id, hardware)
        except Exception as e:
            logger.error(f"Failed to send hardware metrics: {e}")

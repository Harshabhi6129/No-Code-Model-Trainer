"""
AI Sportscaster - Natural language training commentary powered by Gemini
Generates encouraging, technical status updates during training
"""
import os
import time
from typing import List, Dict, Optional
from collections import deque
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


class AICommentator:
    """
    Generates natural language commentary on training progress using Gemini
    """
    
    def __init__(self, socket_manager, job_id: str, trigger_interval: int = 20):
        """
        Args:
            socket_manager: WebSocket manager for broadcasting
            job_id: Training job identifier
            trigger_interval: Generate commentary every N steps
        """
        self.socket_manager = socket_manager
        self.job_id = job_id
        self.trigger_interval = trigger_interval
        self.step_counter = 0
        
        # Metric history for trend analysis
        self.loss_history: deque = deque(maxlen=5)
        self.accuracy_history: deque = deque(maxlen=5)
        self.lr_history: deque = deque(maxlen=5)
        
        # Last commentary time
        self.last_commentary_time = 0
        self.commentary_cooldown = 30  # Seconds between commentary
        
        # Initialize Gemini model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.enabled = True
            logger.info(f"[{job_id}] AI Commentator initialized with Gemini")
        except Exception as e:
            logger.warning(f"[{job_id}] Failed to initialize Gemini: {e}")
            self.enabled = False
    
    async def process_step(
        self,
        step: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch: Optional[float] = None
    ):
        """
        Process a training step and potentially generate commentary
        """
        if not self.enabled:
            return
        
        # Update history
        if loss is not None:
            self.loss_history.append(loss)
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        if learning_rate is not None:
            self.lr_history.append(learning_rate)
        
        self.step_counter += 1
        
        # Check if we should generate commentary
        if self.step_counter % self.trigger_interval != 0:
            return
        
        # Cooldown check
        current_time = time.time()
        if current_time - self.last_commentary_time < self.commentary_cooldown:
            return
        
        self.last_commentary_time = current_time
        
        # Generate commentary
        await self._generate_commentary(step, epoch)
    
    async def _generate_commentary(self, step: int, epoch: Optional[float]):
        """
        Generate natural language commentary using Gemini
        """
        try:
            # Analyze trends
            analysis = self._analyze_trends()
            
            # Build prompt
            prompt = self._build_prompt(step, epoch, analysis)
            
            # Generate with Gemini
            response = await self._call_gemini(prompt)
            
            if response:
                # Send as AI insight
                await self._send_commentary(response, analysis['sentiment'])
                
        except Exception as e:
            logger.error(f"[{self.job_id}] Failed to generate commentary: {e}")
    
    def _analyze_trends(self) -> Dict:
        """
        Analyze metric trends to inform commentary
        """
        analysis = {
            'loss_trend': 'stable',
            'loss_change': 0.0,
            'current_loss': None,
            'current_accuracy': None,
            'sentiment': 'info'
        }
        
        # Loss trend
        if len(self.loss_history) >= 3:
            recent = list(self.loss_history)
            analysis['current_loss'] = recent[-1]
            
            # Calculate change
            initial = recent[0]
            final = recent[-1]
            change_pct = ((final - initial) / initial) * 100 if initial > 0 else 0
            analysis['loss_change'] = change_pct
            
            # Classify trend
            if change_pct < -10:
                analysis['loss_trend'] = 'dropping fast'
                analysis['sentiment'] = 'info'
            elif change_pct < -2:
                analysis['loss_trend'] = 'decreasing steadily'
                analysis['sentiment'] = 'info'
            elif abs(change_pct) < 2:
                analysis['loss_trend'] = 'plateauing'
                analysis['sentiment'] = 'suggestion'
            else:
                analysis['loss_trend'] = 'increasing'
                analysis['sentiment'] = 'warning'
        
        # Accuracy
        if len(self.accuracy_history) > 0:
            analysis['current_accuracy'] = self.accuracy_history[-1]
        
        return analysis
    
    def _build_prompt(self, step: int, epoch: Optional[float], analysis: Dict) -> str:
        """
        Build Gemini prompt for commentary generation
        """
        loss_info = ""
        if analysis['current_loss'] is not None:
            loss_info = f"Current loss: {analysis['current_loss']:.4f} (Trend: {analysis['loss_trend']}, {analysis['loss_change']:.1f}% change)"
        
        acc_info = ""
        if analysis['current_accuracy'] is not None:
            acc_info = f"Accuracy: {analysis['current_accuracy']:.2%}"
        
        epoch_info = f"Epoch {epoch:.1f}" if epoch else "Training in progress"
        
        prompt = f"""You are an AI Flight Engineer monitoring a machine learning training mission.
        
Current Status:
- Step: {step}
- {epoch_info}
- {loss_info}
- {acc_info}

Task: Provide a single, concise sentence (max 15 words) that:
1. Acknowledges the current progress
2. Is encouraging but technically accurate
3. Uses space/flight metaphors (optional)
4. Sounds like a mission control update

Examples:
- "Loss trajectory optimal, continuing descent on schedule"
- "Training velocity nominal, metrics within expected parameters"
- "Slight turbulence detected, but model maintaining stable course"

Your update (ONE sentence only):"""
        
        return prompt
    
    async def _call_gemini(self, prompt: str) -> Optional[str]:
        """
        Call Gemini API for commentary generation
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.8,
                    'max_output_tokens': 50,
                }
            )
            
            if response.text:
                # Clean up response
                commentary = response.text.strip()
                # Remove quotes if present
                commentary = commentary.strip('"\'')
                return commentary
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None
    
    async def _send_commentary(self, message: str, sentiment: str = 'info'):
        """
        Send commentary to frontend via WebSocket
        """
        insight = {
            'type': 'insight',
            'level': sentiment,
            'message': f"🎙️ {message}",
            'details': 'AI Flight Engineer Status Update',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        try:
            await self.socket_manager.broadcast_json(self.job_id, insight)
            logger.info(f"[{self.job_id}] AI Commentary: {message}")
        except Exception as e:
            logger.error(f"Failed to send commentary: {e}")
    
    async def send_milestone_commentary(self, milestone: str):
        """
        Send commentary for major milestones (epoch complete, training done, etc.)
        """
        if not self.enabled:
            return
        
        try:
            prompt = f"""You are an AI Flight Engineer. The training mission just reached this milestone:
"{milestone}"

Provide a ONE sentence celebratory/acknowledgment message (max 12 words), using space/flight terminology.

Examples:
- "Epoch complete, all systems nominal ✓"
- "Training sequence finalized, mission success achieved"
- "Checkpoint reached, trajectory remains optimal"

Your message:"""
            
            response = await self._call_gemini(prompt)
            if response:
                await self._send_commentary(response, 'info')
                
        except Exception as e:
            logger.error(f"Failed to send milestone commentary: {e}")

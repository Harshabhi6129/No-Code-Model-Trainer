# backend/resource_monitor.py
import psutil
import threading
import time
from typing import Dict, Optional
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from services.ws_broker import publish

class ResourceMonitor:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self._get_resource_stats()
                publish(self.run_id, {
                    "type": "resource_update",
                    "resources": stats
                })
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(5)
                
    def _get_resource_stats(self) -> Dict:
        """Get current resource usage statistics"""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }
        
        # GPU stats if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    stats.update({
                        "gpu_utilization": gpu.load * 100,
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal,
                        "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "gpu_temperature": gpu.temperature
                    })
            except Exception:
                pass
                
        return stats

# Global monitors
resource_monitors = {}

def start_resource_monitoring(run_id: str):
    """Start resource monitoring for a training run"""
    if run_id not in resource_monitors:
        monitor = ResourceMonitor(run_id)
        resource_monitors[run_id] = monitor
        monitor.start_monitoring()
        
def stop_resource_monitoring(run_id: str):
    """Stop resource monitoring for a training run"""
    if run_id in resource_monitors:
        resource_monitors[run_id].stop_monitoring()
        del resource_monitors[run_id]
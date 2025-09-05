"""Performance monitoring utilities for CUDA operations."""

import time
import logging
import psutil
import torch
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system and GPU performance during operations."""
    
    def __init__(self):
        self.metrics = {}
    
    @contextmanager
    def monitor_generation(self, operation_name: str = "generation"):
        """Context manager to monitor generation performance."""
        start_time = time.time()
        start_memory = self._get_memory_info()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_info()
            
            duration = end_time - start_time
            memory_delta = self._calculate_memory_delta(start_memory, end_memory)
            
            self._log_performance_metrics(operation_name, duration, memory_delta)
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        info = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / 1e9,
            "ram_percent": psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
                "gpu_utilization": self._get_gpu_utilization(),
            })
        
        return info
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
    
    def _calculate_memory_delta(self, start: Dict, end: Dict) -> Dict[str, float]:
        """Calculate memory usage delta."""
        delta = {}
        for key in start:
            if key in end:
                delta[f"{key}_delta"] = end[key] - start[key]
        return delta
    
    def _log_performance_metrics(self, operation: str, duration: float, memory_delta: Dict):
        """Log performance metrics."""
        logger.info(f"Performance [{operation}]: {duration:.2f}s")
        
        if "gpu_memory_allocated_delta" in memory_delta:
            gpu_delta = memory_delta["gpu_memory_allocated_delta"]
            logger.info(f"GPU memory delta: {gpu_delta:.2f}GB")
        
        if memory_delta.get("ram_used_gb_delta", 0) > 0.1:
            ram_delta = memory_delta["ram_used_gb_delta"]
            logger.info(f"RAM delta: {ram_delta:.2f}GB")


# Global monitor instance
performance_monitor = PerformanceMonitor()

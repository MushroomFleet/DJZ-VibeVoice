"""Advanced memory optimization with dynamic allocation strategies."""

import torch
import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import gc

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class MemoryProfile:
    """Memory usage profile for optimization."""
    gpu_total_gb: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    cpu_total_gb: float
    cpu_used_gb: float
    swap_total_gb: float
    swap_used_gb: float


class AdaptiveMemoryManager:
    """Adaptive memory manager with dynamic optimization strategies."""
    
    def __init__(self, strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE):
        self.strategy = strategy
        self.monitoring_thread = None
        self.should_stop = threading.Event()
        self.memory_history = []
        self.max_history = 100
        
        # Memory thresholds (percentage) - optimized for RTX 4090 24GB
        self.gpu_warning_threshold = 85.0
        self.gpu_critical_threshold = 95.0
        self.cpu_warning_threshold = 90.0
        
        # Optimization callbacks
        self.optimization_callbacks: List[Callable] = []
        
        # Statistics
        self.optimization_count = 0
        self.last_optimization_time = 0
        
    def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.should_stop.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,)
        )
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread."""
        if self.monitoring_thread:
            self.should_stop.set()
            self.monitoring_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while not self.should_stop.is_set():
            try:
                profile = self._get_memory_profile()
                self._update_history(profile)
                self._check_optimization_needed(profile)
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)
    
    def _get_memory_profile(self) -> MemoryProfile:
        """Get current memory usage profile."""
        # GPU memory
        gpu_total = gpu_allocated = gpu_reserved = 0.0
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
        
        # CPU and swap memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return MemoryProfile(
            gpu_total_gb=gpu_total,
            gpu_allocated_gb=gpu_allocated,
            gpu_reserved_gb=gpu_reserved,
            cpu_total_gb=memory.total / 1e9,
            cpu_used_gb=memory.used / 1e9,
            swap_total_gb=swap.total / 1e9,
            swap_used_gb=swap.used / 1e9,
        )
    
    def _update_history(self, profile: MemoryProfile):
        """Update memory usage history."""
        self.memory_history.append(profile)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
    
    def _check_optimization_needed(self, profile: MemoryProfile):
        """Check if memory optimization is needed."""
        gpu_usage_percent = (profile.gpu_reserved_gb / profile.gpu_total_gb) * 100 if profile.gpu_total_gb > 0 else 0
        cpu_usage_percent = (profile.cpu_used_gb / profile.cpu_total_gb) * 100
        
        optimization_needed = False
        optimization_level = "none"
        
        if gpu_usage_percent > self.gpu_critical_threshold:
            optimization_needed = True
            optimization_level = "critical"
        elif gpu_usage_percent > self.gpu_warning_threshold:
            optimization_needed = True
            optimization_level = "warning"
        elif cpu_usage_percent > self.cpu_warning_threshold:
            optimization_needed = True
            optimization_level = "cpu_warning"
        
        if optimization_needed:
            self._trigger_optimization(optimization_level, profile)
    
    def _trigger_optimization(self, level: str, profile: MemoryProfile):
        """Trigger memory optimization based on level."""
        current_time = time.time()
        # Prevent too frequent optimizations
        if current_time - self.last_optimization_time < 5.0:
            return
        
        logger.warning(f"Triggering memory optimization: {level}")
        
        if level == "critical":
            self._aggressive_cleanup()
        elif level == "warning":
            self._moderate_cleanup()
        elif level == "cpu_warning":
            self._cpu_cleanup()
        
        # Run registered callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(level, profile)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")
        
        self.optimization_count += 1
        self.last_optimization_time = current_time
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup for critical situations."""
        logger.info("Performing aggressive memory cleanup")
        
        # Clear all PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear tensor pools if available
        try:
            from app.utils.tensor_pools import tensor_pool_manager
            tensor_pool_manager.clear_pools()
        except ImportError:
            pass
        
        # Clear voice feature cache if available
        try:
            from app.utils.streaming_pipeline import voice_feature_cache
            voice_feature_cache.clear()
        except ImportError:
            pass
    
    def _moderate_cleanup(self):
        """Moderate memory cleanup for warning situations."""
        logger.info("Performing moderate memory cleanup")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def _cpu_cleanup(self):
        """CPU-focused memory cleanup."""
        logger.info("Performing CPU memory cleanup")
        gc.collect()
    
    def register_optimization_callback(self, callback: Callable[[str, MemoryProfile], None]):
        """Register callback for memory optimization events."""
        self.optimization_callbacks.append(callback)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        current_profile = self._get_memory_profile()
        
        return {
            "optimization_count": self.optimization_count,
            "last_optimization_time": self.last_optimization_time,
            "current_gpu_usage_percent": (
                (current_profile.gpu_reserved_gb / current_profile.gpu_total_gb) * 100
                if current_profile.gpu_total_gb > 0 else 0
            ),
            "current_cpu_usage_percent": (
                (current_profile.cpu_used_gb / current_profile.cpu_total_gb) * 100
            ),
            "history_length": len(self.memory_history),
            "strategy": self.strategy.value,
        }
    
    def set_strategy(self, strategy: MemoryStrategy):
        """Update memory management strategy."""
        self.strategy = strategy
        
        # Adjust thresholds based on strategy
        if strategy == MemoryStrategy.CONSERVATIVE:
            self.gpu_warning_threshold = 70.0
            self.gpu_critical_threshold = 85.0
        elif strategy == MemoryStrategy.BALANCED:
            self.gpu_warning_threshold = 85.0
            self.gpu_critical_threshold = 95.0
        elif strategy == MemoryStrategy.AGGRESSIVE:
            self.gpu_warning_threshold = 90.0
            self.gpu_critical_threshold = 98.0
        
        logger.info(f"Memory strategy updated to: {strategy.value}")


# Global memory manager
adaptive_memory_manager = AdaptiveMemoryManager()

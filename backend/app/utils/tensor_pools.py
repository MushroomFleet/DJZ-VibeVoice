"""Pre-allocated tensor pools for memory-efficient processing."""

import torch
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import gc

logger = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    """Specification for a tensor pool."""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str
    requires_grad: bool = False


class TensorPool:
    """Pool of pre-allocated tensors for specific shape/dtype/device."""
    
    def __init__(self, spec: TensorSpec, pool_size: int = 10):
        self.spec = spec
        self.pool_size = pool_size
        self.available = []
        self.in_use = set()
        self.lock = threading.Lock()
        
        # Pre-allocate tensors
        self._populate_pool()
    
    def _populate_pool(self):
        """Pre-allocate tensors for the pool."""
        for _ in range(self.pool_size):
            tensor = torch.zeros(
                self.spec.shape,
                dtype=self.spec.dtype,
                device=self.spec.device,
                requires_grad=self.spec.requires_grad
            )
            self.available.append(tensor)
    
    def get_tensor(self) -> Optional[torch.Tensor]:
        """Get a tensor from the pool."""
        with self.lock:
            if self.available:
                tensor = self.available.pop()
                self.in_use.add(id(tensor))
                # Zero out the tensor for reuse
                tensor.zero_()
                return tensor
            else:
                logger.warning(f"Tensor pool exhausted for shape {self.spec.shape}")
                # Create new tensor if pool exhausted
                tensor = torch.zeros(
                    self.spec.shape,
                    dtype=self.spec.dtype,
                    device=self.spec.device,
                    requires_grad=self.spec.requires_grad
                )
                self.in_use.add(id(tensor))
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> bool:
        """Return a tensor to the pool."""
        tensor_id = id(tensor)
        with self.lock:
            if tensor_id in self.in_use:
                self.in_use.remove(tensor_id)
                if len(self.available) < self.pool_size:
                    # Only return to pool if not full
                    self.available.append(tensor)
                    return True
                else:
                    # Pool full, let tensor be garbage collected
                    return False
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                "available": len(self.available),
                "in_use": len(self.in_use),
                "total_capacity": self.pool_size,
                "utilization": len(self.in_use) / (len(self.in_use) + len(self.available))
            }


class TensorPoolManager:
    """Manages multiple tensor pools for different tensor specifications."""
    
    def __init__(self):
        self.pools: Dict[str, TensorPool] = {}
        self.lock = threading.Lock()
        
        # Register common tensor shapes for voice processing
        self._register_common_pools()
    
    def _register_common_pools(self):
        """Register commonly used tensor pools."""
        common_specs = [
            # Audio chunks
            TensorSpec((4096,), torch.float32, "cuda"),
            TensorSpec((8192,), torch.float32, "cuda"),
            TensorSpec((16384,), torch.float32, "cuda"),
            
            # STFT outputs
            TensorSpec((1025, 128), torch.complex64, "cuda"),
            TensorSpec((1025, 256), torch.complex64, "cuda"),
            
            # Mel spectrograms
            TensorSpec((80, 128), torch.float32, "cuda"),
            TensorSpec((80, 256), torch.float32, "cuda"),
            
            # MFCC features
            TensorSpec((13, 128), torch.float32, "cuda"),
            TensorSpec((13, 256), torch.float32, "cuda"),
            
            # Batch dimensions
            TensorSpec((1, 4096), torch.float32, "cuda"),
            TensorSpec((4, 4096), torch.float32, "cuda"),
            TensorSpec((8, 4096), torch.float32, "cuda"),
        ]
        
        for spec in common_specs:
            self.register_pool(spec, pool_size=20)
    
    def _get_pool_key(self, spec: TensorSpec) -> str:
        """Generate unique key for tensor specification."""
        return f"{spec.shape}_{spec.dtype}_{spec.device}_{spec.requires_grad}"
    
    def register_pool(self, spec: TensorSpec, pool_size: int = 10) -> str:
        """Register a new tensor pool."""
        key = self._get_pool_key(spec)
        with self.lock:
            if key not in self.pools:
                self.pools[key] = TensorPool(spec, pool_size)
                logger.info(f"Registered tensor pool: {key}")
            return key
    
    def get_tensor(self, shape: Tuple[int, ...], 
                   dtype: torch.dtype = torch.float32,
                   device: str = "cuda",
                   requires_grad: bool = False) -> torch.Tensor:
        """Get a tensor from the appropriate pool."""
        spec = TensorSpec(shape, dtype, device, requires_grad)
        key = self._get_pool_key(spec)
        
        with self.lock:
            if key not in self.pools:
                # Auto-register pool if it doesn't exist
                self.pools[key] = TensorPool(spec, pool_size=5)
                logger.debug(f"Auto-registered tensor pool: {key}")
            
            return self.pools[key].get_tensor()
    
    def return_tensor(self, tensor: torch.Tensor) -> bool:
        """Return a tensor to its appropriate pool."""
        spec = TensorSpec(
            tensor.shape, 
            tensor.dtype, 
            tensor.device.type if hasattr(tensor.device, 'type') else str(tensor.device),
            tensor.requires_grad
        )
        key = self._get_pool_key(spec)
        
        with self.lock:
            if key in self.pools:
                return self.pools[key].return_tensor(tensor)
            return False
    
    def clear_pools(self):
        """Clear all tensor pools."""
        with self.lock:
            for pool in self.pools.values():
                pool.available.clear()
                pool.in_use.clear()
            logger.info("Cleared all tensor pools")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        with self.lock:
            return {key: pool.get_stats() for key, pool in self.pools.items()}


class ContextualTensorManager:
    """Context manager for automatic tensor pool management."""
    
    def __init__(self, pool_manager: TensorPoolManager):
        self.pool_manager = pool_manager
        self.acquired_tensors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Return all acquired tensors to their pools
        for tensor in self.acquired_tensors:
            self.pool_manager.return_tensor(tensor)
        self.acquired_tensors.clear()
    
    def get_tensor(self, shape: Tuple[int, ...], 
                   dtype: torch.dtype = torch.float32,
                   device: str = "cuda") -> torch.Tensor:
        """Get tensor and track it for automatic return."""
        tensor = self.pool_manager.get_tensor(shape, dtype, device)
        self.acquired_tensors.append(tensor)
        return tensor


# Global tensor pool manager
tensor_pool_manager = TensorPoolManager()

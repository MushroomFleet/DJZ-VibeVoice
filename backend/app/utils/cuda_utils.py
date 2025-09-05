"""CUDA utilities and device management."""

import torch
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class CUDAManager:
    """Manages CUDA device detection, memory, and optimization settings."""
    
    def __init__(self):
        self.device = self._detect_best_device()
        self.dtype = self._get_optimal_dtype()
        self.memory_fraction = 0.8  # Reserve 20% for system
        
    def _detect_best_device(self) -> str:
        """Detect the best available device with detailed logging."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            
            logger.info(f"CUDA detected: {device_count} devices available")
            logger.info(f"Using device {current_device}: {device_name}")
            logger.info(f"Device memory: {memory_gb:.1f} GB")
            
            # Check for sufficient memory (minimum 4GB recommended)
            if memory_gb < 4.0:
                logger.warning(f"GPU memory ({memory_gb:.1f}GB) may be insufficient for optimal performance")
            
            return f"cuda:{current_device}"
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple MPS acceleration")
            return "mps"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype based on device capabilities."""
        if self.device.startswith("cuda"):
            # Check if device supports bfloat16 (Ampere and newer)
            if torch.cuda.get_device_capability()[0] >= 8:
                logger.info("Using bfloat16 for optimal performance on Ampere+ GPU")
                return torch.bfloat16
            else:
                logger.info("Using float16 for CUDA device")
                return torch.float16
        else:
            logger.info("Using float32 for non-CUDA device")
            return torch.float32
    
    def setup_memory_optimization(self) -> None:
        """Configure CUDA memory settings for optimal performance."""
        if self.device.startswith("cuda"):
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory caching for faster allocation
            torch.cuda.empty_cache()
            
            # Enable cudnn benchmark for faster convolutions
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            logger.info(f"CUDA memory optimization enabled (using {self.memory_fraction*100}% of GPU memory)")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
                "utilization_percent": (reserved / total) * 100
            }
        return {}
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def get_model_load_kwargs(self) -> Dict[str, Any]:
        """Get optimized model loading arguments."""
        kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device.startswith("cuda") else None,
        }
        
        # Add flash attention for supported devices (if available)
        if self.device.startswith("cuda"):
            try:
                import flash_attn
                kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for optimal performance")
            except ImportError:
                logger.warning("Flash Attention not installed - using standard attention")
        
        return kwargs
    
    def validate_tensor_dtype(self, tensor: torch.Tensor, expected_dtype: Optional[torch.dtype] = None) -> bool:
        """Validate that tensor dtype matches expected dtype."""
        target_dtype = expected_dtype or self.dtype
        if tensor.dtype != target_dtype and tensor.dtype.is_floating_point:
            logger.warning(f"Tensor dtype mismatch: got {tensor.dtype}, expected {target_dtype}")
            return False
        return True
    
    def ensure_dtype_consistency(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensure all tensors in dict have consistent dtype."""
        result = {}
        for key, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.dtype.is_floating_point and tensor.dtype != self.dtype:
                    result[key] = tensor.to(self.dtype)
                    logger.debug(f"Converted tensor {key} from {tensor.dtype} to {self.dtype}")
                else:
                    result[key] = tensor
            else:
                result[key] = tensor
        return result
    
    def get_compatible_autocast_context(self, device: str):
        """Get compatible autocast context manager for the device."""
        if device.startswith("cuda"):
            return torch.amp.autocast('cuda', enabled=True, dtype=self.dtype)
        elif device == "mps":
            return torch.amp.autocast('cpu', enabled=False)  # MPS doesn't support autocast yet
        else:
            return torch.amp.autocast('cpu', enabled=False)


# Global instance
cuda_manager = CUDAManager()

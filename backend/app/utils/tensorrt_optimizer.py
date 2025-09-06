"""TensorRT optimization for production deployment."""

import torch
import logging
from typing import Optional, Dict, Any, List, Tuple
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT available for optimization")
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    torch_tensorrt = None
    logger.warning("TensorRT not available - using standard PyTorch")


class TensorRTOptimizer:
    """TensorRT optimization manager for VibeVoice models."""
    
    def __init__(self, cache_dir: str = "tensorrt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.optimized_models = {}
        self.available = TENSORRT_AVAILABLE
        
        # RTX 4090 specific settings
        self.rtx4090_optimizations = self._get_rtx4090_settings()
    
    def _get_rtx4090_settings(self) -> Dict[str, Any]:
        """Get RTX 4090 specific TensorRT settings."""
        return {
            "workspace_size": 4 << 30,  # 4GB workspace for RTX 4090
            "max_batch_size": 8,        # Optimal batch size
            "precision_modes": [torch.float16, torch.int8],  # FP16 + INT8
            "optimization_level": 5,     # Maximum optimization
            "dla_core": None,           # RTX 4090 doesn't have DLA
            "sparse_weights": True,     # Enable sparse weight optimization
            "use_calibration": True,    # Use calibration for INT8
        }
    
    def optimize_model(self, model: torch.nn.Module, 
                      example_inputs: List[torch.Tensor],
                      model_name: str = "vibevoice",
                      precision: str = "fp16") -> torch.nn.Module:
        """Optimize model using TensorRT."""
        if not self.available:
            logger.warning("TensorRT not available - returning original model")
            return model
        
        cache_key = f"{model_name}_{precision}_rtx4090"
        cache_path = self.cache_dir / f"{cache_key}.ts"
        
        # Check if optimized model exists in cache
        if cache_path.exists():
            logger.info(f"Loading cached TensorRT model: {cache_path}")
            try:
                return torch.jit.load(str(cache_path))
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")
        
        try:
            logger.info(f"Optimizing model with TensorRT ({precision}) for RTX 4090...")
            
            # Set precision mode
            precision_mode = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "int8": torch.int8,
            }.get(precision, torch.float16)
            
            # RTX 4090 optimized compilation settings
            compile_settings = {
                "inputs": example_inputs,
                "enabled_precisions": {precision_mode},
                "workspace_size": self.rtx4090_optimizations["workspace_size"],
                "max_batch_size": self.rtx4090_optimizations["max_batch_size"],
                "use_fp32_output": True,
                "truncate_long_and_double": True,
                "optimization_level": self.rtx4090_optimizations["optimization_level"],
            }
            
            # Add RTX 4090 specific optimizations
            if precision == "fp16":
                compile_settings.update({
                    "use_explicit_typing": True,
                    "allow_shape_tensors": True,
                    "sparse_weights": self.rtx4090_optimizations["sparse_weights"],
                })
            
            if precision == "int8":
                # INT8 quantization for maximum performance
                compile_settings.update({
                    "calibrator": self._create_calibrator(model, example_inputs),
                    "use_calibration_cache": True,
                })
            
            # Compile with TensorRT
            optimized_model = torch_tensorrt.compile(model, **compile_settings)
            
            # Save optimized model to cache
            torch.jit.save(optimized_model, str(cache_path))
            logger.info(f"TensorRT optimized model cached: {cache_path}")
            
            self.optimized_models[cache_key] = optimized_model
            return optimized_model
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            logger.info("Falling back to original model")
            return model
    
    def _create_calibrator(self, model: torch.nn.Module, 
                          example_inputs: List[torch.Tensor]) -> Optional[Any]:
        """Create INT8 calibrator for RTX 4090."""
        if not self.available or not trt:
            return None
        
        try:
            # Simple calibrator implementation
            class RTX4090Calibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, calibration_data):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.cache_file = str(self.cache_dir / "calibration_cache.bin")
                    self.data = calibration_data
                    self.batch_size = len(calibration_data)
                    self.current_index = 0
                
                def get_batch_size(self):
                    return self.batch_size
                
                def get_batch(self, names):
                    if self.current_index + self.batch_size > len(self.data):
                        return None
                    
                    batch = self.data[self.current_index:self.current_index + self.batch_size]
                    self.current_index += self.batch_size
                    return [batch]
                
                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()
                    return None
                
                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)
            
            # Generate calibration data
            calibration_data = [inp.detach().cpu().numpy() for inp in example_inputs]
            return RTX4090Calibrator(calibration_data)
            
        except Exception as e:
            logger.warning(f"Failed to create calibrator: {e}")
            return None
    
    def create_example_inputs(self, batch_size: int = 1,
                            sequence_length: int = 256,
                            feature_dim: int = 768,
                            device: str = "cuda") -> List[torch.Tensor]:
        """Create example inputs for TensorRT optimization."""
        return [
            torch.randn(batch_size, sequence_length, device=device),  # Text tokens
            torch.randn(batch_size, 80, 128, device=device),  # Voice features
            torch.randn(batch_size, sequence_length, feature_dim, device=device),  # Text features
        ]
    
    def benchmark_model(self, model: torch.nn.Module,
                       example_inputs: List[torch.Tensor],
                       num_warmup: int = 10,
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(*example_inputs)
        
        # Benchmark with CUDA events for precise timing
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_time.record()
            for _ in range(num_iterations):
                _ = model(*example_inputs)
            end_time.record()
        
        torch.cuda.synchronize()
        
        total_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time
        
        return {
            "avg_inference_time_sec": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_benchmark_time_sec": total_time,
            "num_iterations": num_iterations,
        }
    
    def compare_models(self, original_model: torch.nn.Module,
                      optimized_model: torch.nn.Module,
                      example_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Compare original and optimized model performance."""
        logger.info("Benchmarking original model...")
        original_stats = self.benchmark_model(original_model, example_inputs)
        
        logger.info("Benchmarking TensorRT optimized model...")
        optimized_stats = self.benchmark_model(optimized_model, example_inputs)
        
        speedup = original_stats["avg_inference_time_sec"] / optimized_stats["avg_inference_time_sec"]
        throughput_improvement = optimized_stats["throughput_samples_per_sec"] / original_stats["throughput_samples_per_sec"]
        
        comparison = {
            "original": original_stats,
            "optimized": optimized_stats,
            "speedup_factor": speedup,
            "throughput_improvement": throughput_improvement,
            "performance_gain_percent": (speedup - 1) * 100,
        }
        
        logger.info(f"TensorRT Optimization Results:")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Throughput improvement: {throughput_improvement:.2f}x")
        logger.info(f"  Performance gain: {comparison['performance_gain_percent']:.1f}%")
        
        return comparison
    
    def optimize_for_rtx4090(self, model: torch.nn.Module,
                           example_inputs: List[torch.Tensor]) -> torch.nn.Module:
        """Apply RTX 4090 specific optimizations."""
        if not self.available:
            return model
        
        logger.info("Applying RTX 4090 specific TensorRT optimizations...")
        
        # Try FP16 first (best balance of speed/quality for RTX 4090)
        fp16_model = self.optimize_model(model, example_inputs, "vibevoice_rtx4090", "fp16")
        
        # Benchmark FP16 model
        fp16_results = self.benchmark_model(fp16_model, example_inputs, num_iterations=50)
        
        # Try INT8 for maximum performance if FP16 is working well
        try:
            int8_model = self.optimize_model(model, example_inputs, "vibevoice_rtx4090", "int8")
            int8_results = self.benchmark_model(int8_model, example_inputs, num_iterations=50)
            
            # Choose best model based on performance
            if int8_results["throughput_samples_per_sec"] > fp16_results["throughput_samples_per_sec"] * 1.2:
                logger.info("Using INT8 optimized model for maximum RTX 4090 performance")
                return int8_model
            else:
                logger.info("Using FP16 optimized model for balanced RTX 4090 performance")
                return fp16_model
                
        except Exception as e:
            logger.warning(f"INT8 optimization failed: {e}, using FP16")
            return fp16_model
    
    def clear_cache(self):
        """Clear TensorRT model cache."""
        for cache_file in self.cache_dir.glob("*.ts"):
            cache_file.unlink()
        for cache_file in self.cache_dir.glob("*.bin"):
            cache_file.unlink()
        logger.info("TensorRT cache cleared")
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get TensorRT optimization information."""
        info = {
            "tensorrt_available": self.available,
            "cache_dir": str(self.cache_dir),
            "optimized_models": list(self.optimized_models.keys()),
            "rtx4090_optimizations": self.rtx4090_optimizations,
        }
        
        if self.available and trt:
            info.update({
                "tensorrt_version": trt.__version__,
                "builder_optimization_level": trt.BuilderOptimizationLevel.MAX,
                "supported_formats": ["FP32", "FP16", "INT8"],
            })
        
        return info


# Global TensorRT optimizer
tensorrt_optimizer = TensorRTOptimizer()

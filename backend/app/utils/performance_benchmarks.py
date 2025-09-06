"""Comprehensive performance benchmarking framework."""

import time
import torch
import numpy as np
import logging
import json
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import contextlib

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    throughput: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    iterations: int
    metadata: Dict[str, Any] = None


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpu_model: str
    gpu_memory_gb: float
    cuda_version: str
    pytorch_version: str
    python_version: str


class PerformanceBenchmark:
    """Advanced performance benchmarking framework."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.system_info = self._get_system_info()
        self.results: List[BenchmarkResult] = []
        
        # RTX 4090 specific benchmarking
        self.is_rtx4090 = self._detect_rtx4090()
        if self.is_rtx4090:
            logger.info("RTX 4090 detected - enabling specialized benchmarks")
    
    def _detect_rtx4090(self) -> bool:
        """Detect if running on RTX 4090."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            return 'rtx 4090' in gpu_name or 'geforce rtx 4090' in gpu_name
        return False
    
    def _get_system_info(self) -> SystemInfo:
        """Collect system information."""
        import platform
        
        # CPU info
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False)
        
        # Memory info
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU info
        gpu_model = "Unknown"
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return SystemInfo(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=torch.version.cuda or "Not available",
            pytorch_version=torch.__version__,
            python_version=platform.python_version(),
        )
    
    @contextlib.contextmanager
    def benchmark_context(self, test_name: str, warmup_iterations: int = 5):
        """Context manager for benchmarking operations."""
        # Warmup
        for _ in range(warmup_iterations):
            yield
        
        # Clear memory and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Actual benchmark
        yield
    
    def benchmark_function(self, func: Callable, 
                          test_name: str,
                          iterations: int = 100,
                          warmup_iterations: int = 10,
                          *args, **kwargs) -> BenchmarkResult:
        """Benchmark a function with comprehensive metrics."""
        logger.info(f"Benchmarking: {test_name}")
        
        # Warmup
        for _ in range(warmup_iterations):
            func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Collect measurements
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # Memory before
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / (1024**2)
            else:
                memory_before = psutil.Process().memory_info().rss / (1024**2)
            
            # Time measurement with CUDA events for precision
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                result = func(*args, **kwargs)
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event)  # Already in ms
            else:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / (1024**2)
            else:
                memory_after = psutil.Process().memory_info().rss / (1024**2)
            
            times.append(elapsed_time)
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        times_array = np.array(times)
        memory_array = np.array(memory_usage)
        
        benchmark_result = BenchmarkResult(
            test_name=test_name,
            avg_time_ms=float(np.mean(times_array)),
            min_time_ms=float(np.min(times_array)),
            max_time_ms=float(np.max(times_array)),
            std_time_ms=float(np.std(times_array)),
            throughput=1000.0 / float(np.mean(times_array)),  # samples per second
            memory_usage_mb=float(np.mean(memory_array)),
            gpu_utilization_percent=self._get_gpu_utilization(),
            iterations=iterations,
            metadata={"args_hash": hash(str(args) + str(kwargs))}
        )
        
        self.results.append(benchmark_result)
        
        logger.info(f"  Avg time: {benchmark_result.avg_time_ms:.2f}ms")
        logger.info(f"  Throughput: {benchmark_result.throughput:.2f} samples/sec")
        logger.info(f"  Memory usage: {benchmark_result.memory_usage_mb:.2f}MB")
        
        return benchmark_result
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return 0.0
    
    def benchmark_voice_processing_pipeline(self, voice_service, 
                                          test_configs: List[Dict[str, Any]]) -> Dict[str, BenchmarkResult]:
        """Benchmark the complete voice processing pipeline."""
        results = {}
        
        for config in test_configs:
            test_name = config.get("name", f"config_{len(results)}")
            text = config.get("text", "This is a test of voice synthesis.")
            voice_id = config.get("voice_id", "default")
            
            def voice_generation():
                return voice_service.generate_speech(text, voice_id)
            
            result = self.benchmark_function(
                voice_generation,
                test_name,
                iterations=config.get("iterations", 10),
                warmup_iterations=config.get("warmup", 3)
            )
            
            results[test_name] = result
        
        return results
    
    def benchmark_batch_processing(self, voice_service,
                                 batch_sizes: List[int],
                                 text_length: int = 50) -> Dict[int, BenchmarkResult]:
        """Benchmark batch processing performance."""
        results = {}
        test_text = "This is a benchmark test. " * (text_length // 25)
        
        for batch_size in batch_sizes:
            texts = [test_text] * batch_size
            voice_ids = ["default"] * batch_size
            
            def batch_generation():
                if hasattr(voice_service, 'generate_speech_batch_optimized'):
                    return voice_service.generate_speech_batch_optimized(texts, voice_ids)
                else:
                    return [voice_service.generate_speech(text, voice_id) 
                           for text, voice_id in zip(texts, voice_ids)]
            
            result = self.benchmark_function(
                batch_generation,
                f"batch_size_{batch_size}",
                iterations=5,
                warmup_iterations=2
            )
            
            results[batch_size] = result
        
        return results
    
    def benchmark_advanced_optimizations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark advanced optimization components."""
        results = {}
        
        # Test vectorized audio processing
        try:
            from app.utils.vectorized_audio import vectorized_processor
            
            def test_vectorized_audio():
                dummy_paths = ["test1.wav", "test2.wav"]
                audio_batch, lengths = vectorized_processor.load_audio_batch(dummy_paths)
                features = vectorized_processor.extract_voice_features_batch(audio_batch, lengths)
                return features
            
            results["vectorized_audio"] = self.benchmark_function(
                test_vectorized_audio, "vectorized_audio_processing", iterations=20
            )
        except Exception as e:
            logger.warning(f"Vectorized audio benchmark failed: {e}")
        
        # Test custom CUDA kernels
        try:
            from app.utils.custom_kernels import custom_kernel_manager
            
            if custom_kernel_manager.kernels_loaded:
                def test_custom_kernels():
                    voice_features = torch.randn(4, 512, device="cuda")
                    text_features = torch.randn(4, 512, device="cuda")
                    return custom_kernel_manager.voice_conditioning_forward(
                        voice_features, text_features, 1.0
                    )
                
                results["custom_kernels"] = self.benchmark_function(
                    test_custom_kernels, "custom_cuda_kernels", iterations=50
                )
        except Exception as e:
            logger.warning(f"Custom kernels benchmark failed: {e}")
        
        # Test tensor pools
        try:
            from app.utils.tensor_pools import tensor_pool_manager, ContextualTensorManager
            
            def test_tensor_pools():
                with ContextualTensorManager(tensor_pool_manager) as tm:
                    tensor1 = tm.get_tensor((1024, 256))
                    tensor2 = tm.get_tensor((512, 128))
                    return torch.matmul(tensor1[:512, :128], tensor2)
            
            results["tensor_pools"] = self.benchmark_function(
                test_tensor_pools, "tensor_pool_operations", iterations=100
            )
        except Exception as e:
            logger.warning(f"Tensor pools benchmark failed: {e}")
        
        # Test memory optimization
        try:
            from app.utils.memory_optimizer import adaptive_memory_manager
            
            def test_memory_optimization():
                stats = adaptive_memory_manager.get_optimization_stats()
                return stats
            
            results["memory_optimization"] = self.benchmark_function(
                test_memory_optimization, "memory_optimization", iterations=50
            )
        except Exception as e:
            logger.warning(f"Memory optimization benchmark failed: {e}")
        
        return results
    
    def benchmark_rtx4090_specific_features(self) -> Dict[str, BenchmarkResult]:
        """Benchmark RTX 4090 specific optimizations."""
        if not self.is_rtx4090:
            logger.warning("Not running on RTX 4090 - skipping specific benchmarks")
            return {}
        
        results = {}
        
        # Test Tensor Core utilization
        def test_tensor_cores():
            # Use shapes optimized for Tensor Cores
            a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
            b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
            return torch.matmul(a, b)
        
        results["tensor_cores"] = self.benchmark_function(
            test_tensor_cores, "rtx4090_tensor_cores", iterations=100
        )
        
        # Test high bandwidth memory utilization
        def test_memory_bandwidth():
            large_tensor = torch.randn(8192, 8192, device="cuda")
            return torch.fft.fft2(large_tensor)
        
        results["memory_bandwidth"] = self.benchmark_function(
            test_memory_bandwidth, "rtx4090_memory_bandwidth", iterations=20
        )
        
        # Test bfloat16 performance (Ada Lovelace specific)
        def test_bfloat16():
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            return torch.matmul(a, b)
        
        results["bfloat16"] = self.benchmark_function(
            test_bfloat16, "rtx4090_bfloat16", iterations=100
        )
        
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            gpu_suffix = "rtx4090" if self.is_rtx4090 else "gpu"
            filename = f"benchmark_results_{gpu_suffix}_{timestamp}.json"
        
        output_path = self.results_dir / filename
        
        data = {
            "system_info": asdict(self.system_info),
            "timestamp": time.time(),
            "is_rtx4090": self.is_rtx4090,
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {output_path}")
        return output_path
    
    def generate_report(self) -> str:
        """Generate a human-readable performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["=" * 60]
        report.append("DJZ-VIBEVOICE PERFORMANCE BENCHMARK REPORT")
        if self.is_rtx4090:
            report.append("🚀 RTX 4090 OPTIMIZED RESULTS")
        report.append("=" * 60)
        report.append("")
        
        # System info
        report.append("System Information:")
        report.append(f"  CPU: {self.system_info.cpu_model} ({self.system_info.cpu_cores} cores)")
        report.append(f"  RAM: {self.system_info.ram_gb:.1f} GB")
        report.append(f"  GPU: {self.system_info.gpu_model} ({self.system_info.gpu_memory_gb:.1f} GB)")
        report.append(f"  CUDA: {self.system_info.cuda_version}")
        report.append(f"  PyTorch: {self.system_info.pytorch_version}")
        report.append("")
        
        # Performance targets for RTX 4090
        if self.is_rtx4090:
            report.append("RTX 4090 Performance Targets:")
            report.append("  Target Total Speedup: 15-40x over CPU")
            report.append("  Target Memory Efficiency: 60-80% improvement")
            report.append("  Target Batch Processing: 8+ simultaneous generations")
            report.append("")
        
        # Results
        report.append("Benchmark Results:")
        report.append("-" * 40)
        
        # Group results by category
        optimization_results = []
        voice_results = []
        rtx4090_results = []
        
        for result in self.results:
            if any(keyword in result.test_name.lower() for keyword in 
                   ['vectorized', 'custom', 'tensor', 'memory']):
                optimization_results.append(result)
            elif 'rtx4090' in result.test_name.lower():
                rtx4090_results.append(result)
            else:
                voice_results.append(result)
        
        # Report voice processing results
        if voice_results:
            report.append("Voice Processing Performance:")
            for result in voice_results:
                report.append(f"  {result.test_name}:")
                report.append(f"    Average Time: {result.avg_time_ms:.2f} ms")
                report.append(f"    Throughput: {result.throughput:.2f} samples/sec")
                report.append(f"    Memory Usage: {result.memory_usage_mb:.2f} MB")
                report.append("")
        
        # Report optimization results
        if optimization_results:
            report.append("Advanced Optimizations Performance:")
            for result in optimization_results:
                report.append(f"  {result.test_name}:")
                report.append(f"    Average Time: {result.avg_time_ms:.2f} ms")
                report.append(f"    Throughput: {result.throughput:.2f} ops/sec")
                report.append(f"    GPU Utilization: {result.gpu_utilization_percent:.1f}%")
                report.append("")
        
        # Report RTX 4090 specific results
        if rtx4090_results:
            report.append("RTX 4090 Specific Optimizations:")
            for result in rtx4090_results:
                report.append(f"  {result.test_name}:")
                report.append(f"    Average Time: {result.avg_time_ms:.2f} ms")
                report.append(f"    Throughput: {result.throughput:.2f} ops/sec")
                report.append("")
        
        # Overall summary
        if voice_results:
            avg_voice_time = np.mean([r.avg_time_ms for r in voice_results])
            avg_voice_throughput = np.mean([r.throughput for r in voice_results])
            report.append("Overall Performance Summary:")
            report.append(f"  Average Voice Generation Time: {avg_voice_time:.2f} ms")
            report.append(f"  Average Voice Throughput: {avg_voice_throughput:.2f} samples/sec")
            
            if self.is_rtx4090:
                # Estimate total speedup (baseline CPU ~1-2 seconds per generation)
                estimated_cpu_time = 2000  # ms
                estimated_speedup = estimated_cpu_time / avg_voice_time
                report.append(f"  Estimated Total Speedup: {estimated_speedup:.1f}x over CPU")
                
                if estimated_speedup >= 15:
                    report.append("  🎉 RTX 4090 TARGET ACHIEVED!")
                elif estimated_speedup >= 10:
                    report.append("  ✅ Excellent performance achieved")
                else:
                    report.append("  ⚠️  Performance below RTX 4090 target")
        
        return "\n".join(report)
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, float]:
        """Compare current results with baseline results."""
        baseline_path = self.results_dir / baseline_file
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_path}")
            return {}
        
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        
        baseline_results = {r["test_name"]: r for r in baseline_data["results"]}
        comparisons = {}
        
        for result in self.results:
            if result.test_name in baseline_results:
                baseline = baseline_results[result.test_name]
                speedup = baseline["avg_time_ms"] / result.avg_time_ms
                comparisons[result.test_name] = speedup
                
                if speedup > 1.0:
                    logger.info(f"{result.test_name}: {speedup:.2f}x faster than baseline")
                else:
                    logger.warning(f"{result.test_name}: {1/speedup:.2f}x slower than baseline")
        
        return comparisons
    
    def run_comprehensive_benchmark(self, voice_service) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive benchmark suite...")
        
        all_results = {}
        
        # Basic voice processing benchmarks
        test_configs = [
            {"name": "short_text_synthesis", "text": "Hello world", "iterations": 20},
            {"name": "medium_text_synthesis", "text": "This is a medium length test. " * 10, "iterations": 10},
            {"name": "long_text_synthesis", "text": "Long text content. " * 50, "iterations": 5},
        ]
        
        voice_results = self.benchmark_voice_processing_pipeline(voice_service, test_configs)
        all_results.update(voice_results)
        
        # Batch processing benchmarks
        batch_sizes = [1, 2, 4] if not self.is_rtx4090 else [1, 2, 4, 8]
        batch_results = self.benchmark_batch_processing(voice_service, batch_sizes)
        all_results.update({f"batch_{k}": v for k, v in batch_results.items()})
        
        # Advanced optimizations benchmarks
        opt_results = self.benchmark_advanced_optimizations()
        all_results.update(opt_results)
        
        # RTX 4090 specific benchmarks
        if self.is_rtx4090:
            rtx_results = self.benchmark_rtx4090_specific_features()
            all_results.update(rtx_results)
        
        # Generate and save report
        report = self.generate_report()
        report_path = self.results_dir / f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save results
        results_path = self.save_results()
        
        logger.info("Comprehensive benchmark completed!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Results saved to: {results_path}")
        
        return {
            "report_path": str(report_path),
            "results_path": str(results_path),
            "summary": report,
            "all_results": all_results,
        }


# Global benchmark instance
performance_benchmark = PerformanceBenchmark()

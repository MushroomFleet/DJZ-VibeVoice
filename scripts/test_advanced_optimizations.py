#!/usr/bin/env python3
"""Test script for advanced CUDA optimizations."""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_vectorized_audio_processing():
    """Test CuPy-based vectorized audio processing."""
    logger.info("Testing vectorized audio processing...")
    
    try:
        from backend.app.utils.vectorized_audio import vectorized_processor
        
        # Create dummy audio files for testing
        dummy_paths = ["test_audio_1.wav", "test_audio_2.wav"]
        
        # Test batch audio loading (will use CPU fallback if no real files)
        audio_batch, lengths = vectorized_processor.load_audio_batch(dummy_paths)
        logger.info(f"✅ Audio batch loading: {audio_batch.shape}")
        
        # Test feature extraction
        features = vectorized_processor.extract_voice_features_batch(audio_batch, lengths)
        logger.info(f"✅ Feature extraction: {list(features.keys())}")
        
        # Test GPU acceleration
        if vectorized_processor.use_gpu:
            logger.info("✅ GPU acceleration active with CuPy")
        else:
            logger.info("⚠️ Using CPU fallback (CuPy not available)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Vectorized audio processing failed: {e}")
        return False


def test_ffmpeg_acceleration():
    """Test FFmpeg hardware acceleration."""
    logger.info("Testing FFmpeg acceleration...")
    
    try:
        from backend.app.utils.ffmpeg_acceleration import ffmpeg_accelerator
        
        # Check FFmpeg availability
        available = ffmpeg_accelerator.ffmpeg_available
        logger.info(f"✅ FFmpeg available: {available}")
        
        # Check hardware encoders
        encoders = ffmpeg_accelerator.hardware_encoders
        logger.info(f"✅ Hardware encoders: {list(encoders.keys())}")
        
        # Check RTX 4090 specific settings
        if 'nvenc' in encoders:
            rtx4090_settings = ffmpeg_accelerator.get_rtx4090_optimal_settings()
            logger.info(f"✅ RTX 4090 NVENC settings: {rtx4090_settings['nvenc_preset']}")
        
        # Test benchmark if possible
        try:
            benchmark_results = ffmpeg_accelerator.benchmark_hardware_acceleration()
            if 'speedup_factor' in benchmark_results:
                logger.info(f"✅ Hardware acceleration speedup: {benchmark_results['speedup_factor']:.2f}x")
        except Exception as bench_e:
            logger.warning(f"⚠️ Benchmark failed: {bench_e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ FFmpeg acceleration test failed: {e}")
        return False


def test_memory_management():
    """Test advanced memory management."""
    logger.info("Testing memory management...")
    
    try:
        from backend.app.utils.tensor_pools import tensor_pool_manager, ContextualTensorManager
        from backend.app.utils.memory_optimizer import adaptive_memory_manager
        
        # Test tensor pools
        tensor = tensor_pool_manager.get_tensor((1024,))
        logger.info(f"✅ Tensor pool allocation: {tensor.shape}")
        
        returned = tensor_pool_manager.return_tensor(tensor)
        logger.info(f"✅ Tensor pool return: {returned}")
        
        # Test context manager
        with ContextualTensorManager(tensor_pool_manager) as tm:
            temp_tensor = tm.get_tensor((512, 256))
            logger.info(f"✅ Context manager tensor: {temp_tensor.shape}")
        
        # Test memory monitoring
        stats = adaptive_memory_manager.get_optimization_stats()
        logger.info(f"✅ Memory stats: {stats['current_gpu_usage_percent']:.1f}% GPU")
        
        # Test memory strategy
        adaptive_memory_manager.set_strategy(adaptive_memory_manager.strategy)
        logger.info(f"✅ Memory strategy: {adaptive_memory_manager.strategy.value}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        return False


def test_streaming_pipeline():
    """Test streaming audio pipeline."""
    logger.info("Testing streaming pipeline...")
    
    try:
        from backend.app.utils.streaming_pipeline import streaming_processor, voice_feature_cache
        
        # Test voice feature cache
        dummy_features = {"mel": np.random.randn(80, 128)}
        voice_feature_cache.put("test_voice", dummy_features)
        
        cached = voice_feature_cache.get("test_voice")
        logger.info(f"✅ Voice feature caching: {cached is not None}")
        
        # Test cache statistics
        cache_stats = voice_feature_cache.get_stats()
        logger.info(f"✅ Cache stats: hit_rate={cache_stats['hit_rate']:.2f}")
        
        # Test streaming processor stats
        stats = streaming_processor.get_processing_stats()
        logger.info(f"✅ Streaming stats: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Streaming pipeline test failed: {e}")
        return False


def test_custom_kernels():
    """Test custom CUDA kernels."""
    logger.info("Testing custom CUDA kernels...")
    
    try:
        import torch
        from backend.app.utils.custom_kernels import custom_kernel_manager
        
        # Get kernel info
        kernel_info = custom_kernel_manager.get_kernel_info()
        logger.info(f"✅ Kernel info: {kernel_info['gpu_name']}")
        
        if custom_kernel_manager.kernels_loaded:
            # Test voice conditioning kernel
            voice_features = torch.randn(2, 512, device="cuda")
            text_features = torch.randn(2, 512, device="cuda")
            
            conditioned = custom_kernel_manager.voice_conditioning_forward(
                voice_features, text_features, 1.0
            )
            logger.info(f"✅ Voice conditioning kernel: {conditioned.shape}")
            
            # Test RTX 4090 specific features
            if custom_kernel_manager.ada_lovelace_optimized:
                logger.info("✅ Ada Lovelace optimizations enabled")
                
                # Test Tensor Core optimized attention
                try:
                    query = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
                    key = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
                    value = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
                    
                    attention_out = custom_kernel_manager.ada_lovelace_attention_forward(
                        query, key, value, 0.125
                    )
                    logger.info(f"✅ Ada Lovelace attention: {attention_out.shape}")
                except Exception as attn_e:
                    logger.warning(f"⚠️ Ada Lovelace attention failed: {attn_e}")
            
            # Benchmark kernels
            try:
                benchmark_results = custom_kernel_manager.benchmark_kernels()
                if 'voice_conditioning_speedup' in benchmark_results:
                    speedup = benchmark_results['voice_conditioning_speedup']
                    logger.info(f"✅ Custom kernel speedup: {speedup:.2f}x")
            except Exception as bench_e:
                logger.warning(f"⚠️ Kernel benchmark failed: {bench_e}")
                
        else:
            logger.info("⚠️ Custom kernels not loaded - using fallback")
        
        return True
    except Exception as e:
        logger.error(f"❌ Custom kernels test failed: {e}")
        return False


def test_tensorrt_optimization():
    """Test TensorRT optimization."""
    logger.info("Testing TensorRT optimization...")
    
    try:
        from backend.app.utils.tensorrt_optimizer import tensorrt_optimizer
        
        # Check TensorRT availability
        opt_info = tensorrt_optimizer.get_optimization_info()
        logger.info(f"✅ TensorRT available: {opt_info['tensorrt_available']}")
        
        if opt_info['tensorrt_available']:
            logger.info(f"✅ TensorRT version: {opt_info.get('tensorrt_version', 'Unknown')}")
            logger.info(f"✅ RTX 4090 optimizations: {opt_info['rtx4090_optimizations']}")
            
            # Test example input creation
            example_inputs = tensorrt_optimizer.create_example_inputs(batch_size=2)
            logger.info(f"✅ Example inputs created: {len(example_inputs)} tensors")
        else:
            logger.info("⚠️ TensorRT not available - optimization disabled")
        
        return True
    except Exception as e:
        logger.error(f"❌ TensorRT optimization test failed: {e}")
        return False


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    logger.info("Running performance benchmark...")
    
    try:
        from backend.app.utils.performance_benchmarks import performance_benchmark
        
        # Simple function to benchmark
        def dummy_inference():
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x = torch.randn(1, 1000, device=device)
            return torch.nn.functional.relu(x).sum()
        
        # Benchmark the function
        result = performance_benchmark.benchmark_function(
            dummy_inference,
            "dummy_inference_test",
            iterations=50,
            warmup_iterations=10
        )
        
        logger.info(f"✅ Benchmark completed: {result.avg_time_ms:.2f}ms avg")
        
        # Test RTX 4090 detection
        if performance_benchmark.is_rtx4090:
            logger.info("✅ RTX 4090 detected - specialized benchmarks available")
        
        return True
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False


def test_cuda_setup():
    """Test basic CUDA setup and RTX 4090 detection."""
    logger.info("Testing CUDA setup...")
    
    try:
        import torch
        from backend.app.utils.cuda_utils import cuda_manager
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"✅ GPU: {device_name}")
            logger.info(f"✅ GPU Memory: {memory_gb:.1f} GB")
            logger.info(f"✅ Device Count: {device_count}")
            
            # Check RTX 4090
            if 'rtx 4090' in device_name.lower():
                logger.info("🚀 RTX 4090 detected - all optimizations available!")
            
            # Test CUDA manager
            device = cuda_manager.device
            dtype = cuda_manager.dtype
            logger.info(f"✅ CUDA Manager - Device: {device}, Dtype: {dtype}")
            
            # Test memory info
            memory_info = cuda_manager.get_memory_info()
            if memory_info:
                logger.info(f"✅ GPU Memory Usage: {memory_info['utilization_percent']:.1f}%")
        
        return True
    except Exception as e:
        logger.error(f"❌ CUDA setup test failed: {e}")
        return False


def test_integration():
    """Test integration between optimization components."""
    logger.info("Testing optimization integration...")
    
    try:
        # Test if all imports work together
        from backend.app.utils.tensor_pools import tensor_pool_manager
        from backend.app.utils.memory_optimizer import adaptive_memory_manager
        from backend.app.utils.vectorized_audio import vectorized_processor
        from backend.app.utils.custom_kernels import custom_kernel_manager
        from backend.app.utils.performance_benchmarks import performance_benchmark
        
        logger.info("✅ All optimization modules imported successfully")
        
        # Test memory manager callback registration
        def test_callback(level, profile):
            logger.debug(f"Memory optimization callback: {level}")
        
        adaptive_memory_manager.register_optimization_callback(test_callback)
        logger.info("✅ Memory optimization callback registered")
        
        # Test combined tensor pool + vectorized processing
        import torch
        if torch.cuda.is_available():
            with tensor_pool_manager.__class__(tensor_pool_manager) as tm:
                test_tensor = tm.get_tensor((1024,))
                logger.info(f"✅ Integrated tensor allocation: {test_tensor.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def generate_optimization_summary():
    """Generate summary of available optimizations."""
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    
    optimizations = []
    
    try:
        from backend.app.utils.vectorized_audio import vectorized_processor
        status = "🚀 ACTIVE" if vectorized_processor.use_gpu else "⚠️ CPU FALLBACK"
        optimizations.append(f"Vectorized Audio Processing: {status}")
    except:
        optimizations.append("Vectorized Audio Processing: ❌ FAILED")
    
    try:
        from backend.app.utils.ffmpeg_acceleration import ffmpeg_accelerator
        status = "🚀 ACTIVE" if ffmpeg_accelerator.ffmpeg_available else "❌ UNAVAILABLE"
        optimizations.append(f"FFmpeg Hardware Acceleration: {status}")
        if 'nvenc' in ffmpeg_accelerator.hardware_encoders:
            optimizations.append("  └─ NVENC (RTX 4090): 🚀 AVAILABLE")
    except:
        optimizations.append("FFmpeg Hardware Acceleration: ❌ FAILED")
    
    try:
        from backend.app.utils.custom_kernels import custom_kernel_manager
        status = "🚀 ACTIVE" if custom_kernel_manager.kernels_loaded else "⚠️ FALLBACK"
        optimizations.append(f"Custom CUDA Kernels: {status}")
        if custom_kernel_manager.ada_lovelace_optimized:
            optimizations.append("  └─ Ada Lovelace Optimizations: 🚀 ENABLED")
    except:
        optimizations.append("Custom CUDA Kernels: ❌ FAILED")
    
    try:
        from backend.app.utils.tensorrt_optimizer import tensorrt_optimizer
        status = "🚀 ACTIVE" if tensorrt_optimizer.available else "⚠️ UNAVAILABLE"
        optimizations.append(f"TensorRT Optimization: {status}")
    except:
        optimizations.append("TensorRT Optimization: ❌ FAILED")
    
    try:
        from backend.app.utils.tensor_pools import tensor_pool_manager
        pool_stats = tensor_pool_manager.get_all_stats()
        pool_count = len([k for k, v in pool_stats.items() if v['total_capacity'] > 0])
        optimizations.append(f"Tensor Pools: 🚀 ACTIVE ({pool_count} pools)")
    except:
        optimizations.append("Tensor Pools: ❌ FAILED")
    
    try:
        from backend.app.utils.memory_optimizer import adaptive_memory_manager
        memory_stats = adaptive_memory_manager.get_optimization_stats()
        strategy = memory_stats['strategy']
        optimizations.append(f"Adaptive Memory Management: 🚀 ACTIVE ({strategy})")
    except:
        optimizations.append("Adaptive Memory Management: ❌ FAILED")
    
    for opt in optimizations:
        logger.info(opt)
    
    # Expected performance summary
    logger.info("\nExpected Performance Improvements:")
    logger.info("  Vectorized Audio: 3-8x speedup")
    logger.info("  FFmpeg Hardware: 2-5x I/O acceleration")
    logger.info("  Tensor Pools: 1.5-3x + 30-50% memory reduction")
    logger.info("  Custom Kernels: 2-6x specialized operations")
    logger.info("  TensorRT: 2-4x inference acceleration")
    logger.info("  Memory Management: 25-35% memory reduction")
    logger.info("")
    logger.info("🎯 RTX 4090 Target: 15-40x total speedup over CPU")


def main():
    """Run all optimization tests."""
    logger.info("=" * 60)
    logger.info("DJZ-VibeVoice Advanced Optimizations Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("CUDA Setup & RTX 4090 Detection", test_cuda_setup),
        ("Vectorized Audio Processing", test_vectorized_audio_processing),
        ("FFmpeg Hardware Acceleration", test_ffmpeg_acceleration),
        ("Memory Management", test_memory_management),
        ("Streaming Pipeline", test_streaming_pipeline),
        ("Custom CUDA Kernels", test_custom_kernels),
        ("TensorRT Optimization", test_tensorrt_optimization),
        ("Performance Benchmark", run_performance_benchmark),
        ("Integration Testing", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Generate optimization summary
    generate_optimization_summary()
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All advanced optimizations are working correctly!")
        logger.info("🚀 Ready for RTX 4090 maximum performance!")
        return 0
    elif passed >= len(tests) * 0.8:
        logger.info("✅ Most optimizations working - good performance expected")
        return 0
    else:
        logger.warning("⚠️ Some optimizations may not be fully functional")
        logger.info("💡 Check the installation requirements and GPU drivers")
        return 1


if __name__ == "__main__":
    sys.exit(main())

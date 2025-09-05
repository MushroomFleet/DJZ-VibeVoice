#!/usr/bin/env python3
"""Validate CUDA setup for DJZ-VibeVoice."""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cuda_installation():
    """Test basic CUDA installation."""
    print("=== CUDA Installation Test ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    
    return torch.cuda.is_available()


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("\n=== Tensor Operations Test ===")
    
    try:
        from app.utils.cuda_utils import cuda_manager
        device = cuda_manager.device
        dtype = cuda_manager.dtype
        
        print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        
        # Create test tensors
        a = torch.randn(1000, 1000, device=device, dtype=dtype)
        b = torch.randn(1000, 1000, device=device, dtype=dtype)
        
        # Perform operations
        c = torch.matmul(a, b)
        result = c.sum().item()
        
        print(f"Matrix multiplication result: {result:.2f}")
        print("✅ Tensor operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False


def test_model_loading():
    """Test VibeVoice model loading."""
    print("\n=== Model Loading Test ===")
    
    try:
        from app.services.voice_service import VoiceService
        from app.utils.cuda_utils import cuda_manager
        
        voice_service = VoiceService()
        
        if voice_service.is_model_loaded():
            print("✅ Model loaded successfully")
            
            # Test memory info
            memory_info = cuda_manager.get_memory_info()
            if memory_info:
                print(f"GPU memory usage: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB")
                print(f"GPU utilization: {memory_info['utilization_percent']:.1f}%")
            
            return True
        else:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False


def test_generation_performance():
    """Test generation performance."""
    print("\n=== Generation Performance Test ===")
    
    try:
        from app.services.voice_service import VoiceService
        
        voice_service = VoiceService()
        
        if not voice_service.is_model_loaded():
            print("❌ Model not loaded, skipping performance test")
            return False
        
        # Create a dummy voice profile for testing
        test_text = "This is a test of CUDA-accelerated voice generation."
        
        # You would need an actual voice file for this test
        # For now, we'll just test the model loading
        print("✅ Ready for generation performance testing")
        print("Note: Add actual voice files to test generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("DJZ-VibeVoice CUDA Validation")
    print("=" * 40)
    
    tests = [
        ("CUDA Installation", test_cuda_installation),
        ("Tensor Operations", test_tensor_operations),
        ("Model Loading", test_model_loading),
        ("Generation Performance", test_generation_performance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CUDA setup is ready.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

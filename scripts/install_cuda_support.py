#!/usr/bin/env python3
"""Installation script for CUDA support."""

import subprocess
import sys
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    return result


def detect_cuda_version():
    """Detect installed CUDA version."""
    try:
        result = run_command(["nvcc", "--version"], check=False)
        if result.returncode == 0:
            output = result.stdout
            # Check for specific versions first
            if "12.4" in output:
                print("⚠️  CUDA 12.4 detected. Using cu121 (most compatible with PyTorch)")
                return "121"
            elif "12.1" in output:
                return "121" 
            elif "12.0" in output:
                return "120"
            elif "11.8" in output:
                return "118"
            # Fallback for other 12.x versions - use latest stable
            elif "release 12." in output:
                print("⚠️  CUDA 12.x detected but not specifically supported. Using cu121 (most compatible)")
                return "121"
            # Fallback for other 11.x versions
            elif "release 11." in output:
                print("⚠️  CUDA 11.x detected but not specifically supported. Using cu118 (most compatible)")
                return "118"
        return None
    except FileNotFoundError:
        return None


def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    cuda_version = detect_cuda_version()
    
    if not cuda_version:
        print("❌ CUDA not detected. Please install CUDA Toolkit first.")
        print("Download from: https://developer.nvidia.com/cuda-downloads")
        return False
    
    print(f"✅ CUDA {cuda_version} detected")
    
    # Uninstall CPU version
    print("Uninstalling CPU-only PyTorch...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    
    # Install CUDA version
    print(f"Installing PyTorch with CUDA {cuda_version}...")
    cuda_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", cuda_url
    ]
    
    result = run_command(cmd)
    return result.returncode == 0


def install_additional_packages():
    """Install additional CUDA-related packages."""
    packages = [
        "accelerate",
        "optimum",
        "psutil",  # For performance monitoring
    ]
    
    # Try to install flash-attn (optional)
    try:
        print("Installing flash-attn (optional)...")
        run_command([sys.executable, "-m", "pip", "install", "flash-attn>=2.0.0"], check=False)
        print("✅ flash-attn installed")
    except Exception as e:
        print(f"⚠️  flash-attn installation failed: {e}")
        print("This is optional and won't affect basic CUDA functionality")
    
    print("Installing additional packages...")
    for package in packages:
        result = run_command([sys.executable, "-m", "pip", "install", package])
        if result.returncode == 0:
            print(f"✅ {package} installed")
        else:
            print(f"❌ {package} installation failed")
    
    return True


def main():
    """Main installation process."""
    print("DJZ-VibeVoice CUDA Installation")
    print("=" * 40)
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        print("❌ PyTorch CUDA installation failed")
        return False
    
    # Install additional packages
    install_additional_packages()
    
    # Validate installation
    print("\n=== Validation ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print("✅ Installation successful!")
        else:
            print("❌ CUDA not detected after installation")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 CUDA support installation completed!")
        print("Run 'python scripts/validate_cuda_setup.py' to test the setup.")
    else:
        print("\n❌ Installation failed. Check the errors above.")
    
    sys.exit(0 if success else 1)

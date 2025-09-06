"""Configuration module for VibeVoice application."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_NAME: str = "DJZ-VibeVoice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    PRODUCTION: bool = False  # Set to True for production deployment

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Model settings - Use Hugging Face model for weights, local package for code
    MODEL_PATH: str = "microsoft/VibeVoice-1.5B"
    DEVICE: str = "auto"  # auto-detect best device
    MAX_LENGTH: int = 1000
    CFG_SCALE: float = 1.3

    # CUDA-specific settings
    CUDA_MEMORY_FRACTION: float = 0.8  # Use 80% of GPU memory
    ENABLE_MIXED_PRECISION: bool = True
    ENABLE_FLASH_ATTENTION: bool = True
    ENABLE_MODEL_COMPILATION: bool = True  # PyTorch 2.0+ compile

    # Performance settings
    BATCH_SIZE: int = 1  # Adjust based on GPU memory
    MAX_CONCURRENT_GENERATIONS: int = 2  # Limit concurrent requests

    # Advanced optimization settings
    ENABLE_VECTORIZED_AUDIO: bool = True
    ENABLE_FFMPEG_ACCELERATION: bool = True
    ENABLE_CUSTOM_KERNELS: bool = True
    ENABLE_TENSORRT: bool = False  # Production only
    ENABLE_STREAMING_PIPELINE: bool = True
    
    # Memory management
    MEMORY_STRATEGY: str = "adaptive"  # conservative, balanced, aggressive, adaptive
    TENSOR_POOL_SIZE: int = 20
    VOICE_CACHE_SIZE: int = 100
    MEMORY_MONITORING_INTERVAL: float = 1.0  # seconds
    
    # GPU-specific optimizations
    ENABLE_TENSOR_CORES: bool = True  # Use Tensor Cores when available
    PREFER_BFLOAT16: bool = True  # RTX 4090 optimization
    CUDA_GRAPHS_ENABLED: bool = True  # CUDA graphs for static operations
    
    # Performance monitoring
    ENABLE_BENCHMARKING: bool = False
    BENCHMARK_RESULTS_DIR: str = "benchmark_results"
    PERFORMANCE_LOG_LEVEL: str = "INFO"
    
    # Generation behavior control (V1.10 RC1 Pipeline Fixes)
    PRESERVE_ORIGINAL_BEHAVIOR: bool = True  # DEFAULT: Keep original VibeVoice behavior
    ENABLE_CUSTOM_GENERATION_CONFIG: bool = False  # Disable forced generation parameters
    ENABLE_CFG_OVERRIDE: bool = False  # Disable forced CFG scale minimum

    # Optional generation parameters (only used if ENABLE_CUSTOM_GENERATION_CONFIG=True)
    GENERATION_DO_SAMPLE: bool = False  # Match VibeVoice default
    GENERATION_TEMPERATURE: float = 1.0
    GENERATION_TOP_P: float = 1.0
    GENERATION_REPETITION_PENALTY: float = 1.0

    # Performance optimization control
    OPTIMIZE_SINGLE_REQUESTS: bool = False  # DEFAULT: Don't optimize single requests
    OPTIMIZE_BATCH_REQUESTS: bool = True    # Batch optimization OK
    AUTO_USE_OPTIMIZATIONS: bool = False    # Let user choose
    
    # Vectorized audio settings
    VECTORIZED_CHUNK_SIZE: int = 4096
    VECTORIZED_BUFFER_SIZE: int = 64
    AUDIO_BATCH_SIZE: int = 8  # For batch processing
    
    # FFmpeg acceleration settings
    FFMPEG_HARDWARE_ACCELERATION: str = "auto"  # auto, nvenc, qsv, vaapi, software
    FFMPEG_PRESET: str = "p4"  # RTX 4090 optimized preset
    FFMPEG_BITRATE: str = "128k"
    
    # TensorRT settings (production)
    TENSORRT_PRECISION: str = "fp16"  # fp32, fp16, int8
    TENSORRT_WORKSPACE_SIZE_GB: int = 4  # RTX 4090 optimized
    TENSORRT_MAX_BATCH_SIZE: int = 8
    TENSORRT_CACHE_DIR: str = "tensorrt_cache"

    # Path settings - Updated for monorepo structure
    BACKEND_DIR: Path = Path(__file__).parent.parent
    ROOT_DIR: Path = BACKEND_DIR.parent
    BASE_DIR: Path = ROOT_DIR / "data"
    VOICES_DIR: Path = BASE_DIR / "voices"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"

    # Audio settings
    SAMPLE_RATE: int = 24000
    MAX_AUDIO_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    # keep non-blocking startup
    LOAD_MODEL_ON_STARTUP: bool = False

    # Silence HF tokenizers fork/parallelism warning
    TOKENIZERS_PARALLELISM: bool = False

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.VOICES_DIR.mkdir(exist_ok=True)
        self.OUTPUTS_DIR.mkdir(exist_ok=True)
        self.UPLOADS_DIR.mkdir(exist_ok=True)


settings = Settings()

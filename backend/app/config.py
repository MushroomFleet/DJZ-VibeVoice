"""Configuration module for VibeVoice application."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_NAME: str = "DJZ-VibeVoice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

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

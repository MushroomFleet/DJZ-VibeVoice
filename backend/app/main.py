"""Main FastAPI application."""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.api import router

# Advanced optimization imports
if settings.ENABLE_VECTORIZED_AUDIO or settings.ENABLE_FFMPEG_ACCELERATION:
    try:
        from app.utils.memory_optimizer import adaptive_memory_manager, MemoryStrategy
        from app.utils.tensor_pools import tensor_pool_manager
        from app.utils.vectorized_audio import vectorized_processor
        from app.utils.ffmpeg_acceleration import ffmpeg_accelerator
        from app.utils.streaming_pipeline import streaming_processor, voice_feature_cache
        from app.utils.custom_kernels import custom_kernel_manager
        from app.utils.performance_benchmarks import performance_benchmark
        OPTIMIZATIONS_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("Advanced CUDA optimizations loaded successfully")
    except ImportError as e:
        OPTIMIZATIONS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning(f"Advanced optimizations not available: {e}")
else:
    OPTIMIZATIONS_AVAILABLE = False
    logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for optimization components."""
    logger.info("Starting DJZ-VibeVoice with advanced optimizations...")
    
    # Initialize optimization components
    if OPTIMIZATIONS_AVAILABLE:
        try:
            # Set memory management strategy
            memory_strategy = MemoryStrategy(settings.MEMORY_STRATEGY)
            adaptive_memory_manager.set_strategy(memory_strategy)
            
            # Start memory monitoring
            adaptive_memory_manager.start_monitoring(
                interval=settings.MEMORY_MONITORING_INTERVAL
            )
            logger.info(f"Memory monitoring started (strategy: {settings.MEMORY_STRATEGY})")
            
            # Log GPU information
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # Log optimization status
                logger.info(f"Vectorized audio: {vectorized_processor.use_gpu}")
                logger.info(f"FFmpeg acceleration: {ffmpeg_accelerator.ffmpeg_available}")
                logger.info(f"Custom kernels: {custom_kernel_manager.kernels_loaded}")
                logger.info(f"Hardware encoders: {list(ffmpeg_accelerator.hardware_encoders.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced optimizations: {e}")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    if OPTIMIZATIONS_AVAILABLE:
        try:
            adaptive_memory_manager.stop_monitoring()
            streaming_processor.stop_processing()
            tensor_pool_manager.clear_pools()
            voice_feature_cache.clear()
            logger.info("Advanced optimizations cleaned up")
        except Exception as e:
            logger.error(f"Error during optimization cleanup: {e}")


# Create FastAPI app with lifespan management
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Beautiful AI Voice Synthesis Application",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Root endpoint - serve the main page
@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


# Run the application
if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Server running on http://{settings.HOST}:{settings.PORT}")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

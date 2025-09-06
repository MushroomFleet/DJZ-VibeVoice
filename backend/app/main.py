"""Main FastAPI application."""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
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

# Check for production build
backend_dir = Path(__file__).parent.parent
root_dir = backend_dir.parent
frontend_build_dir = root_dir / "frontend" / "dist"
is_production_build = frontend_build_dir.exists() and (frontend_build_dir / "index.html").exists()

if is_production_build:
    # Production mode: serve built frontend
    logger.info("Production mode detected: serving built frontend files")
    
    # Mount built frontend assets
    app.mount("/assets", StaticFiles(directory=str(frontend_build_dir / "assets")), name="assets")
    
    # Mount static files for API
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return FileResponse(str(frontend_build_dir / "index.html"))
        
    # Serve the built frontend index.html for all non-API routes (must be last)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # If it's an API route, let it pass through
        if full_path.startswith("api/"):
            return {"error": "API endpoint not found"}
        
        # For all other routes, serve the frontend index.html
        return FileResponse(str(frontend_build_dir / "index.html"))
        
else:
    # Development mode: serve backend static files only
    logger.info("Development mode: serving backend static files, frontend should run separately")
    
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

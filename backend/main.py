#!/usr/bin/env python3
"""
DJZ-VibeVoice Standalone Backend
Main entry point for the FastAPI backend server.
"""

import sys
import os
from pathlib import Path

# Add the models directory to Python path for VibeVoice imports
backend_dir = Path(__file__).parent
root_dir = backend_dir.parent
models_dir = root_dir / "models"
sys.path.insert(0, str(models_dir))

# Set working directory to backend for relative imports
os.chdir(backend_dir)

# Import the FastAPI application
from app.main import app, settings, logger

if __name__ == "__main__":
    import uvicorn
    
    # Update port to avoid conflicts (use 8001 instead of 8000)
    port = 8001
    
    logger.info(f"Starting DJZ-VibeVoice Backend v{settings.APP_VERSION}")
    logger.info(f"Server running on http://{settings.HOST}:{port}")
    logger.info(f"Frontend proxy should point to this address")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG,
        log_level="info",
    )

#!/usr/bin/env python3
"""
DJZ-VibeVoice Standalone Backend
Main entry point for the FastAPI backend server.
"""

import sys
import os
from pathlib import Path
import argparse

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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DJZ-VibeVoice Backend Server')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    args = parser.parse_args()
    
    # Configure for production or development
    is_production = args.production
    port = 8001
    
    if is_production:
        logger.info(f"Starting DJZ-VibeVoice Backend v{settings.APP_VERSION} in PRODUCTION mode")
        logger.info(f"Serving frontend from built files")
        logger.info(f"Server running on http://{settings.HOST}:{port}")
        
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=port,
            reload=False,  # Disable reload in production
            log_level="info",
        )
    else:
        logger.info(f"Starting DJZ-VibeVoice Backend v{settings.APP_VERSION} in DEVELOPMENT mode")
        logger.info(f"Server running on http://{settings.HOST}:{port}")
        logger.info(f"Frontend proxy should point to this address")
        
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=port,
            reload=settings.DEBUG,
            log_level="info",
        )

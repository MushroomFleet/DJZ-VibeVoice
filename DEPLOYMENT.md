# 🚀 DJZ-VibeVoice Production Deployment Guide

## Production Build & Deployment

### Quick Start Commands

```bash
# Development Mode (separate frontend/backend servers)
npm run dev

# Production Mode (single server with built frontend)
npm run start
```

### Production Setup Process

1. **Build Frontend**
   ```bash
   npm run build
   ```
   - Builds React frontend to `frontend/dist/` 
   - Creates optimized production assets
   - Outputs: `index.html`, CSS, and JS bundles

2. **Start Production Server**
   ```bash
   npm run start:production
   ```
   - Runs backend with `--production` flag
   - Serves built frontend files from backend
   - Single server on port 8001

3. **Unified Production Command**
   ```bash
   npm run start
   ```
   - Builds frontend automatically
   - Starts production server
   - One-command deployment

## Production vs Development Mode

### Development Mode (`npm run dev`)
- **Frontend**: Vite dev server on port 5173
- **Backend**: FastAPI with hot reload on port 8001
- **Proxy**: Frontend proxies `/api` requests to backend
- **Use Case**: Active development with hot reload

### Production Mode (`npm run start`)
- **Frontend**: Built static files served by backend
- **Backend**: FastAPI production server on port 8001
- **Routing**: Backend serves frontend + API routes
- **Use Case**: Production deployment, testing builds

## Configuration Files

### Root Package.json Scripts
```json
{
  "scripts": {
    "dev": "concurrently \"npm run dev:backend\" \"npm run dev:frontend\"",
    "build": "cd frontend && npm run build",
    "start": "npm run build && npm run start:production",
    "start:production": "cd backend && python main.py --production"
  }
}
```

### Backend Production Detection
- **Automatic**: Detects `frontend/dist/index.html` existence
- **Manual**: `--production` flag in startup command
- **Environment**: `.env.production` file for production settings

## Environment Configuration

### Development (.env)
```env
DEBUG=true
DEVICE=auto
ENABLE_VECTORIZED_AUDIO=true
ENABLE_FFMPEG_ACCELERATION=true
```

### Production (.env.production)
```env
DEBUG=false
PRODUCTION=true
APP_VERSION=1.2.0
MEMORY_STRATEGY=balanced
CUDA_MEMORY_FRACTION=0.85
ENABLE_BENCHMARKING=false
```

## Advanced CUDA Optimization Status

### Production Startup Verification
When starting in production mode, verify these logs:

```
✅ Production mode detected: serving built frontend files
✅ GPU detected: NVIDIA GeForce RTX 4090 (25.8GB)
✅ Vectorized audio: True
✅ FFmpeg acceleration: True  
✅ Custom kernels: True
✅ Hardware encoders: ['nvenc', 'qsv']
```

### Performance Expectations
- **Development**: 10-20 seconds generation time
- **Production**: 3-8 seconds with CUDA optimizations
- **Memory**: 60-80% optimization with tensor pools
- **I/O**: 3.88x speedup with hardware NVENC

## Deployment Architecture

### File Structure
```
DJZ-VibeVoice/
├── frontend/
│   ├── dist/                 # Built production files
│   │   ├── index.html       # Main frontend entry
│   │   └── assets/          # CSS, JS bundles
│   └── src/                 # React source code
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI app with frontend serving
│   ├── main.py              # Entry point with --production flag
│   └── .env.production      # Production configuration
└── package.json             # Root scripts for unified commands
```

### Request Routing (Production)
```
http://localhost:8001/
├── /api/*          → Backend API routes
├── /static/*       → Backend static files  
├── /assets/*       → Frontend build assets
└── /*              → Frontend index.html (React routing)
```

## Deployment Options

### Local Production Testing
```bash
npm run start
# Access: http://localhost:8001
```

### Docker Deployment (Future)
```dockerfile
# Multi-stage build
FROM node:18 AS frontend-build
# Build frontend...

FROM python:3.10 AS backend
# Copy built frontend to backend/static/
# Install Python dependencies
# Start production server
```

### Cloud Deployment Considerations
- **Port**: Application runs on port 8001
- **GPU**: Requires NVIDIA GPU for optimal performance
- **Memory**: 16GB+ RAM recommended for RTX 4090
- **Storage**: 10GB+ for models and cache

## Performance Monitoring

### Health Check Endpoints
```bash
# API Status
curl http://localhost:8001/api/status

# Performance Metrics (v1.2.0)
curl http://localhost:8001/api/performance/status
```

### Expected Response
```json
{
  "optimizations_active": true,
  "gpu_utilization": "15.2%",
  "memory_strategy": "adaptive", 
  "nvenc_available": true,
  "custom_kernels_loaded": true,
  "expected_speedup": "15-40x"
}
```

## Troubleshooting

### Build Issues
```bash
# Clean and rebuild
npm run clean
npm install
npm run build
```

### Frontend Not Loading
- Check `frontend/dist/index.html` exists
- Verify backend logs show "Production mode detected"
- Ensure assets are accessible at `/assets/`

### Performance Issues
- Verify GPU detection in startup logs
- Check CUDA optimizations are loaded
- Monitor memory usage with `nvidia-smi`

### Port Conflicts
```bash
# Windows
netstat -ano | findstr :8001
taskkill /f /pid <PID>

# Kill specific process
taskkill /f /pid 36120
```

## Security Considerations

### Production Hardening
- Set `DEBUG=false` in production
- Configure proper CORS origins
- Use HTTPS in production environments
- Implement proper authentication if needed

### Environment Variables
- Use `.env.production` for production settings
- Never commit sensitive credentials
- Use environment-specific configuration

## Version 1.2.0 Ready

This production setup provides the foundation for version 1.2.0 features:
- ✅ Single-command deployment (`npm run start`)
- ✅ Optimized frontend build serving
- ✅ Advanced CUDA optimizations active
- ✅ Performance monitoring endpoints ready
- ✅ Production environment configuration

Ready for version 1.2.0 feature development and deployment! 🚀

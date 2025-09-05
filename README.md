# 🎙️ DJZ-VibeVoice

A standalone AI-powered voice synthesis application built with React and FastAPI. Generate natural-sounding speech from text using Microsoft's VibeVoice model with custom voice profiles.

![DJZ-VibeVoice](https://img.shields.io/badge/DJZ-VibeVoice-purple?style=for-the-badge&logo=microphone)
![React](https://img.shields.io/badge/React-19+-blue?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi)

## ✨ Features

- 🎤 **Voice Training**: Upload audio files or record your voice directly  
- 📝 **Text-to-Speech**: Convert text or text files to natural speech  
- 🎭 **Multiple Speakers**: Support for up to 4 distinct speakers  
- 💾 **Voice Library**: Save and manage custom voice profiles  
- 🎨 **Beautiful UI**: Modern, responsive React interface with dark/light themes  
- ⚡ **Real-time Processing**: Fast speech generation with streaming support  
- 📊 **Audio Visualization**: Live waveform display during recording  
- 💾 **Download & Save**: Export generated audio files  

## 🏗️ Architecture

This is a **monorepo** containing:

```
djz-vibevoice/
├── frontend/              # React application (Vite + modern stack)
├── backend/               # FastAPI server with VibeVoice integration
├── models/                # VibeVoice model files
├── data/                  # Application data
│   ├── voices/           # Stored voice profiles
│   ├── outputs/          # Generated audio files
│   └── uploads/          # Temporary uploads
└── shared/               # Common utilities and types
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for backend) 
- **CUDA-capable GPU** (recommended)
- **8GB+ RAM**

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd djz-vibevoice
```

2. **Install all dependencies**
```bash
npm run install:all
```

3. **Set up Python environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

4. **Install VibeVoice model**
```bash
cd models/VibeVoice
pip install -e .
cd ../..
```

5. **Configure environment**
```bash
cd backend
cp .env.example .env
# Edit .env with your settings
cd ..
```

### Running the Application

**Option 1: Run both frontend and backend together**
```bash
npm run dev
```

**Option 2: Run separately**
```bash
# Terminal 1 - Backend (FastAPI server on port 8001)
npm run dev:backend

# Terminal 2 - Frontend (React dev server on port 5173)
npm run dev:frontend
```

**Access the application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8001

## 🔧 Configuration

### Backend Configuration

Edit `backend/.env`:

```env
# Application Settings
APP_NAME=DJZ-VibeVoice
APP_VERSION=1.0.0
DEBUG=False

# Server Settings
HOST=0.0.0.0
PORT=8001

# Model Settings
MODEL_PATH=microsoft/VibeVoice-1.5B
DEVICE=cuda
CFG_SCALE=1.3

# Audio Settings
SAMPLE_RATE=24000
MAX_AUDIO_SIZE_MB=50
```

### Frontend Configuration

The frontend automatically proxies API requests to the backend. No additional configuration needed.

## 🎯 Usage Examples

### Basic Text-to-Speech

1. Start the application (`npm run dev`)
2. Upload or record a voice sample
3. Enter your text
4. Click "Generate Speech"
5. Play and download the result

### Multi-Speaker Conversations

```text
Speaker 1: Hello, welcome to our podcast!
Speaker 2: Thanks, I'm excited to be here.
Speaker 1: Let's dive into today's topic.
```

### Voice Cloning

1. Record 10-30 seconds of clear speech
2. Save with a descriptive name
3. Use the voice for any text generation

## 📁 Project Structure

### Frontend (React + Vite)
- **Components**: Modular React components with CSS modules
- **Services**: API integration and data handling  
- **Context**: Global state management
- **Hooks**: Reusable React hooks

### Backend (FastAPI)
- **API Routes**: RESTful endpoints for voice and audio operations
- **Services**: Business logic and VibeVoice integration
- **Models**: Data models and validation
- **Configuration**: Environment-based settings

## 🛠️ Development

### Available Scripts

```bash
# Development
npm run dev                 # Run both frontend and backend
npm run dev:frontend        # Run React dev server only
npm run dev:backend         # Run FastAPI server only

# Building
npm run build              # Build frontend for production

# Maintenance
npm run install:all        # Install all dependencies
npm run clean             # Clean build artifacts
```

### API Endpoints

- `GET /api/voices` — List available voices
- `POST /api/voices/upload` — Upload voice file
- `POST /api/voices/record` — Save recorded voice
- `DELETE /api/voices/{id}` — Delete voice
- `POST /api/generate` — Generate speech from text
- `POST /api/generate/file` — Generate from text file
- `GET /api/audio/{filename}` — Download audio file
- `DELETE /api/audio/{filename}` — Delete audio file

## 🚦 System Requirements

**Minimum**
- Node.js 18+, Python 3.9+
- 8GB RAM, CPU with AVX support
- 5GB disk space

**Recommended**
- Node.js 20+, Python 3.10+
- 16GB RAM, NVIDIA GPU (8GB+ VRAM)
- 10GB disk space, SSD storage

## 🐛 Troubleshooting

### Common Issues

**Frontend won't start**
```bash
cd frontend && npm install
```

**Backend connection errors**
```bash
cd backend && python main.py
# Check if running on port 8001
```

**Out of memory errors**
- Use a smaller model variant
- Reduce batch size in generation
- Close other applications

**Slow generation**
- Ensure GPU is enabled (`DEVICE=cuda`)
- Use shorter text inputs
- Check GPU memory usage

### Getting Help

1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure ports 5173 and 8001 are available
4. Check GPU drivers are up to date

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) for the core AI model
- [FastAPI](https://fastapi.tiangolo.com) for the backend framework
- [React](https://react.dev) and [Vite](https://vitejs.dev) for the frontend stack
- All contributors and users of this project

## 🔗 Related Projects

- [VibeVoice Model](https://github.com/microsoft/VibeVoice)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [React Documentation](https://react.dev)

---

**DJZ-VibeVoice** - Standalone voice synthesis made simple 🎙️✨

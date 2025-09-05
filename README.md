# 🎙️ DJZ-VibeVoice

A standalone AI-powered voice synthesis application built with React and FastAPI. Generate natural-sounding speech from text using Microsoft's VibeVoice model with custom voice profiles.

![DJZ-VibeVoice](https://img.shields.io/badge/DJZ-VibeVoice-purple?style=for-the-badge&logo=microphone)
![React](https://img.shields.io/badge/React-19+-blue?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi)

## 📖 About This Project

**DJZ-VibeVoice** is a standalone monorepo evolution of voice synthesis technology, building upon the excellent work of the original projects:

- **🧠 Core AI Model**: Based on [VibeVoice](https://github.com/mypapit/VibeVoice) - Microsoft's state-of-the-art voice synthesis model
- **🔧 Original Implementation**: Forked from [vibevoice-studio](https://github.com/shamspias/vibevoice-studio) - Python/Gradio-based UI

We've completely rebuilt the application as a **modern web-based monorepo** with:
- **React frontend** with professional UI/UX design
- **FastAPI backend** with optimized performance
- **Audio Gallery** for managing generated speech files
- **Modular architecture** for future development

This project is currently **aimed at developers** but will include **end-user friendly options** in future releases.

## ✨ Features

- 🎤 **Voice Training**: Upload audio files or record your voice directly  
- 📝 **Text-to-Speech**: Convert text or text files to natural speech  
- 🎭 **Multiple Speakers**: Support for up to 4 distinct speakers  
- 💾 **Voice Library**: Save and manage custom voice profiles  
- 🎵 **Audio Gallery**: Browse, play, and manage generated audio files with search and filter
- 🎨 **Beautiful UI**: Modern, responsive React interface with dark/light themes  
- ⚡ **Real-time Processing**: Fast speech generation with streaming support  
- 📊 **Audio Visualization**: Live waveform display during recording  
- 💾 **Download & Save**: Export generated audio files  
- 🗂️ **File Management**: Bulk operations for audio library organization

## 🚀 Installation & Setup

### Prerequisites
- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for backend) 
- **Git** (for cloning repositories)
- **8GB+ RAM** (16GB recommended for faster generation)
- **CUDA-capable GPU** (optional but recommended for faster generation)

⚠️ **Performance Note**: Without GPU acceleration, speech generation will run on CPU and take 1-2 minutes per generation. GPU acceleration reduces this to 10-20 seconds.

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/MushroomFleet/DJZ-VibeVoice.git
cd DJZ-VibeVoice
```

#### 2. Install Node.js Dependencies
```bash
npm install
cd frontend
npm install
cd ..
```

#### 3. Set Up Python Backend Environment
```bash
cd backend
python -m venv venv

# Activate virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
cd ..
```

#### 4. Install VibeVoice Model
```bash
cd models/VibeVoice
pip install -e .
cd ../..
```

#### 5. Configure Backend Environment
```bash
cd backend
cp env.example .env
```

Edit `backend/.env` if needed (default settings work for most setups):
```env
# Application Settings
APP_NAME=DJZ-VibeVoice
APP_VERSION=1.0.0
DEBUG=False

# Server Settings
HOST=127.0.0.1
PORT=8001

# Model Settings
MODEL_PATH=microsoft/VibeVoice-1.5B
DEVICE=cpu  # Change to "cuda" if you have NVIDIA GPU
CFG_SCALE=1.0

# Audio Settings
SAMPLE_RATE=24000
MAX_AUDIO_SIZE_MB=50

# Performance Settings
LOAD_MODEL_ON_STARTUP=False
TOKENIZERS_PARALLELISM=false
```

#### 6. Start the Application
```bash
npm run dev
```

The application will start both frontend and backend. You should see:
- Frontend available at: http://localhost:5173
- Backend API at: http://localhost:8001

**Note**: The backend will take 30-60 seconds to fully load the AI model on first startup.

## 🎯 How to Use

### Quick Usage Guide

1. **Access the Application**
   - Open your browser to http://localhost:5173
   - Wait for the backend to finish loading (check console for "Model loaded successfully")

2. **Create Your First Voice Profile**
   - Click the microphone icon to record 10-30 seconds of clear speech
   - Or use the upload button to select an audio file (.wav, .mp3, .m4a, .flac, .ogg)
   - Give your voice a descriptive name and save

3. **Generate Speech**
   - Select your voice from the dropdown menu
   - Enter text in the text input area
   - Click "Generate Speech" 
   - Wait 1-2 minutes for CPU generation (10-20 seconds with GPU)

4. **Manage Your Audio**
   - Generated audio appears in the "Generated Audio" section
   - Click the folder icon to browse your Audio Gallery
   - Play, download, or delete audio files as needed

### Advanced Features

**Multi-Speaker Conversations**
```text
Speaker 1: Hello, welcome to our podcast!
Speaker 2: Thanks, I'm excited to be here.
Speaker 1: Let's dive into today's topic.
```

**Batch Text Processing**
- Upload `.txt` files for longer content
- Generate speech from entire documents
- Automatically save to Audio Gallery

**Voice Cloning Best Practices**
- Use 10-30 seconds of clear, high-quality audio
- Record in a quiet environment
- Speak naturally with varied intonation
- Save multiple voice variants for different use cases

## 🏗️ Architecture

This is a **monorepo** containing:

```
DJZ-VibeVoice/
├── frontend/              # React application (Vite + modern stack)
├── backend/               # FastAPI server with VibeVoice integration
├── models/                # VibeVoice model files
├── data/                  # Application data (auto-created)
│   ├── voices/           # Stored voice profiles
│   ├── outputs/          # Generated audio files (Audio Gallery)
│   └── uploads/          # Temporary uploads
└── package.json          # Root package management
```

### Frontend (React + Vite)
- **Components**: Modular React components with CSS modules
- **Audio Gallery**: Full-featured audio file management
- **Services**: API integration and data handling  
- **Context**: Global state management
- **Hooks**: Reusable React hooks for audio operations

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

**Voice Management**
- `GET /api/voices` — List available voices
- `POST /api/voices/upload` — Upload voice file
- `POST /api/voices/record` — Save recorded voice
- `DELETE /api/voices/{id}` — Delete voice

**Speech Generation**
- `POST /api/generate` — Generate speech from text
- `POST /api/generate/file` — Generate from text file

**Audio Gallery**
- `GET /api/audio/library` — Get audio library with metadata
- `GET /api/audio/{filename}` — Download audio file
- `DELETE /api/audio/{filename}` — Delete audio file

**System**
- `GET /api/health` — Check system status

## 🚦 System Requirements

**Minimum**
- Node.js 18+, Python 3.9+
- 8GB RAM, CPU with AVX support
- 5GB disk space
- **Expected Performance**: 1-2 minutes per speech generation

**Recommended**
- Node.js 20+, Python 3.10+
- 16GB RAM, NVIDIA GPU (8GB+ VRAM)
- 10GB disk space, SSD storage
- **Expected Performance**: 10-20 seconds per speech generation

## 🐛 Troubleshooting

### Common Issues

**"concurrently is not recognized"**
```bash
npm install
# Ensure you're in the root directory when running npm install
```

**Port 8001 already in use**
```bash
# Windows:
netstat -ano | findstr :8001
taskkill /f /pid <PID_NUMBER>

# macOS/Linux:
lsof -ti:8001 | xargs kill -9
```

**Backend fails to start**
```bash
cd backend
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
python main.py
```

**"Model not loaded" errors**
- Wait 30-60 seconds for initial model loading
- Check console for "Model loaded successfully" message
- Ensure sufficient RAM (8GB minimum)

**Slow generation (CPU)**
- Normal behavior on CPU - expect 1-2 minutes
- For faster generation, use NVIDIA GPU with CUDA
- Change `DEVICE=cuda` in `backend/.env` if GPU available

**Audio uploads fail**
- Ensure backend is running on port 8001
- Check file format (.wav, .mp3, .m4a, .flac, .ogg)
- File size limit is 50MB (configurable in .env)

**Frontend won't load**
```bash
cd frontend
npm install
npm run dev
```

**Tokenizer warnings**
```
The tokenizer class you load from this checkpoint is 'Qwen2Tokenizer'.
The class this function is called from is 'VibeVoiceTextTokenizerFast'.
```
- These warnings are normal and don't affect functionality
- The application will still generate speech correctly

### Performance Optimization

**For CPU Users:**
- Close unnecessary applications
- Use shorter text inputs (under 100 words)
- Consider using simpler configuration: `CFG_SCALE=1.0`

**For GPU Users:**
- Install CUDA toolkit if not already installed
- Change `DEVICE=cuda` in `backend/.env`
- Ensure GPU has 8GB+ VRAM for best performance

**Memory Management:**
- Restart application if generation becomes very slow
- Monitor system RAM usage
- Close browser tabs when not needed

### Getting Help

1. Check the console for error messages
2. Verify all dependencies are installed correctly
3. Ensure ports 5173 and 8001 are available
4. Check that Python virtual environment is activated
5. Confirm all installation steps were completed

## 🗺️ Roadmap

### Current State (v1.0)
- ✅ Web-based UI with React frontend
- ✅ FastAPI backend integration
- ✅ Voice recording and upload
- ✅ Text-to-speech generation
- ✅ Audio Gallery with file management
- ✅ CPU and GPU support
- ✅ Developer-focused setup

### Upcoming Features (v1.1+)
- 🔄 One-click installer for end users
- 🔄 Pre-built voice model downloads
- 🔄 Audio waveform visualization
- 🔄 Batch text processing improvements
- 🔄 Voice quality metrics
- 🔄 Export to multiple formats
- 🔄 Performance optimizations

### Future Vision
- 📋 Desktop application packaging
- 📋 Cloud deployment options
- 📋 API key integration for hosted models
- 📋 Community voice sharing
- 📋 Plugin system for extensions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[VibeVoice Model](https://github.com/mypapit/VibeVoice)** - The core AI model powering voice synthesis
- **[VibevoiceStudio](https://github.com/shamspias/vibevoice-studio)** - Original Python/Gradio implementation that inspired this project
- **[Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)** - Original research and model development
- **[FastAPI](https://fastapi.tiangolo.com)** - High-performance backend framework
- **[React](https://react.dev)** and **[Vite](https://vitejs.dev)** - Modern frontend development
- All contributors and users of this project

## 🔗 Related Projects

- [VibeVoice Model](https://github.com/mypapit/VibeVoice) - Core AI model
- [Original VibevoiceStudio](https://github.com/shamspias/vibevoice-studio) - Python/Gradio version
- [Microsoft VibeVoice Research](https://github.com/microsoft/VibeVoice) - Original research
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [React Documentation](https://react.dev)

---

**DJZ-VibeVoice** - Standalone voice synthesis made simple 🎙️✨

*From developers, for developers - evolving towards universal accessibility*

# 🚫 Do Not Upload Guide

This document shows the complete file structure of DJZ-VibeVoice with clear markings for what should **NOT** be uploaded to GitHub.

**Legend:**
- ❌ = **DO NOT UPLOAD** - These files should be excluded from version control
- ✅ = **SAFE TO UPLOAD** - These files should be included in the repository
- 📁 = Directory (follow individual file rules within)

---

## 📂 Root Directory

```
DJZ-VibeVoice/
├── ✅ .gitignore
├── ✅ ADVANCED-CUDA-IMPLEMENTATION-COMPLETE.md
├── ✅ CUDA-ADVANCED-DEVTEAM-HANDOFF.md
├── ✅ CUDA-DEVTEAM-HANDOFF.md
├── ✅ DEPLOYMENT.md                  🚀 NEW v1.2.0 - Production deployment guide
├── ✅ DJZVV-0.5.0-update.md
├── ✅ do-not-upload.md
├── ✅ docker-compose.cuda.yml
├── ✅ Dockerfile.cuda
├── ✅ eslint.config.js
├── ✅ LICENSE
├── ✅ package-lock.json
├── ✅ package.json
├── ✅ preprocessor_config.json
├── ✅ README.md
├── ✅ requirements-cuda-advanced.txt
├── ✅ requirements-cuda.txt
├── ✅ requirements.txt
├── ✅ V1_10_RC1_FINAL_COMPLETE_REPORT.md
├── ✅ V1_10_RC1_FINAL_VALIDATION_REPORT.md
├── ✅ V1_10_RC1_PIPELINE_EXECUTION_COMPLETE.md
├── ❌ v110rc1_diagnostic_report.json ❌ Generated test data
├── ✅ v110rc1-devteam-pipeline-check.md
├── ✅ v110rc1-report-explained.md
├── ✅ VERSION_1.2.0_PLAN.md          🚀 NEW v1.2.0 - Feature roadmap and planning
├── 📁 backend/
├── 📁 data/                          ❌ ENTIRE DIRECTORY - User data
├── 📁 frontend/
├── 📁 models/
└── 📁 scripts/
```

---

## 📂 Backend Directory (`backend/`)

```
backend/
├── ✅ env.example                    ✅ Template file - safe to share
├── ❌ .env                          ❌ Contains sensitive configuration
├── ❌ .env.production               ❌ Production environment variables (sensitive)
├── ✅ main.py
├── ✅ requirements.txt
├── ❌ v110rc1_diagnostic_report.json ❌ Generated test report data
├── ❌ benchmark_results/             ❌ ENTIRE DIRECTORY - Generated benchmark data
├── ❌ tensorrt_cache/               ❌ ENTIRE DIRECTORY - TensorRT cache files
└── 📁 app/
    ├── ✅ __init__.py
    ├── ✅ config.py
    ├── ✅ main.py
    ├── 📁 api/
    │   ├── ✅ __init__.py
    │   └── ✅ routes.py
    ├── 📁 models/
    │   ├── ✅ __init__.py
    │   └── ✅ voice_model.py
    ├── 📁 services/
    │   ├── ✅ __init__.py
    │   ├── ✅ audio_service.py
    │   ├── ✅ voice_service.py
    │   └── ❌ voice_service.py.backup_preprocessor_fix  ❌ Backup file
    ├── 📁 static/                    ✅ Static assets - safe to share
    │   ├── ✅ index.html
    │   ├── 📁 css/
    │   │   └── ✅ style.css
    │   ├── 📁 js/
    │   │   └── ✅ app.js
    │   └── 📁 media/
    │       ├── ✅ favicon.ico
    │       └── ✅ logo.png
    └── 📁 utils/                     🚀 NEW v1.2.0 CUDA Optimization Files
        ├── ✅ cuda_utils.py
        ├── ✅ custom_kernels.py
        ├── ✅ ffmpeg_acceleration.py
        ├── ✅ memory_optimizer.py
        ├── ✅ performance_benchmarks.py
        ├── ✅ performance_monitor.py
        ├── ✅ streaming_pipeline.py
        ├── ✅ tensor_pools.py
        ├── ✅ tensorrt_optimizer.py
        └── ✅ vectorized_audio.py
```

---

## 📂 Data Directory (`data/`) - ❌ **ENTIRE DIRECTORY EXCLUDED**

```
data/                                 ❌ ENTIRE DIRECTORY - Contains user data
├── 📁 outputs/                      ❌ Generated audio files (user content)
│   ├── ❌ Brag_994dffd5_20250905_191524.json
│   ├── ❌ Brag_994dffd5_20250905_191524.wav
│   ├── ❌ djz-sample_1_8f26fabb_20250905_050741.json
│   ├── ❌ djz-sample_1_8f26fabb_20250905_050741.wav
│   ├── ❌ S1-Brag_994dffd5_S2-djz-sample_1_8f26fabb_S3-Sandy-set_d8f97d27_20250905_203645.json
│   ├── ❌ S1-Brag_994dffd5_S2-djz-sample_1_8f26fabb_S3-Sandy-set_d8f97d27_20250905_203645.wav
│   └── ❌ [All other generated audio/json files...]
├── 📁 uploads/                      ❌ Temporary upload files
└── 📁 voices/                       ❌ User voice profiles (private data)
    ├── ❌ Brag_994dffd5.wav
    ├── ❌ djz-sample_1_8f26fabb.wav
    └── ❌ Sandy-set_d8f97d27.wav
```

**Reason:** The entire `data/` directory contains user-generated content, personal voice recordings, and generated audio files. This data is private and should never be uploaded to a public repository.

---

## 📂 Frontend Directory (`frontend/`)

```
frontend/
├── ❌ dist/                         ❌ Built frontend files (auto-generated by npm run build)
├── ✅ index.html
├── ❌ k.txt                         ❌ Unknown/temporary file
├── ❌ node_modules/                 ❌ Dependencies (auto-installed by npm install)
├── ✅ package-lock.json
├── ✅ package.json
├── ✅ vite.config.js
├── 📁 public/
│   └── ✅ vite.svg
└── 📁 src/
    ├── ✅ App.css
    ├── ✅ App.jsx
    ├── ✅ index.css
    ├── ✅ main.jsx
    ├── 📁 assets/
    │   └── ✅ react.svg
    ├── 📁 components/
    │   ├── 📁 audio/
    │   │   ├── 📁 AudioCard/
    │   │   │   ├── ✅ AudioCard.jsx
    │   │   │   └── ✅ AudioCard.module.css
    │   │   └── 📁 AudioLibrary/
    │   │       ├── ✅ AudioLibrary.jsx
    │   │       └── ✅ AudioLibrary.module.css
    │   ├── 📁 common/
    │   │   ├── 📁 Button/
    │   │   │   ├── ✅ Button.jsx
    │   │   │   └── ✅ Button.module.css
    │   │   └── 📁 LoadingOverlay/
    │   │       ├── ✅ LoadingOverlay.jsx
    │   │       └── ✅ LoadingOverlay.module.css
    │   ├── 📁 layout/
    │   │   ├── 📁 Header/
    │   │   │   ├── ✅ Header.jsx
    │   │   │   └── ✅ Header.module.css
    │   │   └── 📁 Layout/
    │   │       ├── ✅ Layout.jsx
    │   │       └── ✅ Layout.module.css
    │   ├── 📁 pages/
    │   │   └── 📁 MainPage/
    │   │       ├── ✅ MainPage.jsx
    │   │       └── ✅ MainPage.module.css
    │   ├── 📁 text/
    │   │   ├── 📁 GenerationSettings/
    │   │   │   ├── ✅ GenerationSettings.jsx
    │   │   │   └── ✅ GenerationSettings.module.css
    │   │   └── 📁 TextInput/
    │   │       ├── ✅ TextInput.jsx
    │   │       └── ✅ TextInput.module.css
    │   └── 📁 voice/
    │       ├── 📁 VoiceAssignment/
    │       │   ├── ✅ VoiceAssignment.jsx
    │       │   └── ✅ VoiceAssignment.module.css
    │       ├── 📁 VoiceCard/
    │       │   ├── ✅ VoiceCard.jsx
    │       │   └── ✅ VoiceCard.module.css
    │       ├── 📁 VoiceRecorder/
    │       │   └── ✅ VoiceRecorder.jsx
    │       ├── 📁 VoiceSelector/
    │       │   ├── ✅ VoiceSelector.jsx
    │       │   └── ✅ VoiceSelector.module.css
    │       └── 📁 VoiceUploader/
    │           ├── ✅ VoiceUploader.jsx
    │           └── ✅ VoiceUploader.module.css
    ├── 📁 contexts/
    │   └── ✅ AppContext.jsx
    ├── 📁 services/
    │   ├── ✅ api.js
    │   ├── ✅ audioService.js
    │   └── ✅ voiceService.js
    └── 📁 styles/
        ├── ✅ globals.css
        └── ✅ variables.css
```

---

## 📂 Models Directory (`models/`)

```
models/
└── 📁 VibeVoice/
    ├── ✅ k.txt
    ├── ✅ LICENSE
    ├── ✅ preprocessor_config.json
    ├── ✅ pyproject.toml
    ├── ✅ README.md
    ├── ✅ say.py
    ├── ✅ SECURITY.md
    ├── 📁 Figures/
    │   ├── ✅ Google_AI_Studio_2025-08-25T21_48_13.452Z.png
    │   ├── ✅ MOS-preference.png
    │   ├── ✅ VibeVoice_logo_white.png
    │   ├── ✅ VibeVoice_logo.png
    │   └── ✅ VibeVoice.jpg
    ├── 📁 vibevoice/
    │   ├── ✅ __init__.py
    │   ├── 📁 configs/
    │   │   ├── ✅ qwen2.5_1.5b_64k.json
    │   │   └── ✅ qwen2.5_7b_32k.json
    │   ├── 📁 modular/
    │   │   ├── ✅ __init__.py
    │   │   ├── ✅ configuration_vibevoice.py
    │   │   ├── ✅ modeling_vibevoice_inference.py
    │   │   ├── ✅ modeling_vibevoice.py
    │   │   ├── ✅ modular_vibevoice_diffusion_head.py
    │   │   ├── ✅ modular_vibevoice_text_tokenizer.py
    │   │   ├── ✅ modular_vibevoice_tokenizer.py
    │   │   └── ✅ streamer.py
    │   ├── 📁 processor/
    │   │   ├── ✅ __init__.py
    │   │   ├── ✅ vibevoice_processor.py
    │   │   └── ✅ vibevoice_tokenizer_processor.py
    │   ├── 📁 schedule/
    │   │   ├── ✅ __init__.py
    │   │   ├── ✅ dpm_solver.py
    │   │   └── ✅ timestep_sampler.py
    │   └── 📁 scripts/
    │       ├── ✅ __init__.py
    │       └── ✅ convert_nnscaler_checkpoint_to_transformers.py
    └── 📁 vibevoice.egg-info/         ❌ Build artifacts - auto-generated
        ├── ❌ dependency_links.txt
        ├── ❌ PKG-INFO
        ├── ❌ requires.txt
        ├── ❌ SOURCES.txt
        └── ❌ top_level.txt
```

---

## 📂 Scripts Directory (`scripts/`)

```
scripts/
├── ✅ debug_voice_conditioning.py
├── ✅ fix_preprocessor_detection.py          🚀 NEW v1.2.0 - Preprocessor fix utility
├── ✅ fix_voice_conditioning.py
├── ✅ install_cuda_support.py
├── ✅ test_advanced_optimizations.py         🚀 NEW v1.2.0 - CUDA optimization testing
├── ✅ test_cuda_fixes.py
├── ✅ test_tokenizer_fix.py
├── ✅ test_v110rc1_pipeline_fixes.py         🚀 NEW v1.2.0 - V1.10 RC1 diagnostic testing
├── ✅ test_voice_cloning.py
└── ✅ validate_cuda_setup.py
```

---

## 🚫 Additional Files to Exclude (Even if Not Present)

**These files should ALWAYS be excluded if they appear:**

### Build Artifacts & Dependencies
- ❌ `node_modules/` - Node.js dependencies (auto-installed)
- ❌ `__pycache__/` - Python cache directories
- ❌ `*.pyc` - Python compiled files
- ❌ `*.pyo` - Python optimized files
- ❌ `dist/` - Build output directories
- ❌ `build/` - Build directories
- ❌ `.pytest_cache/` - Test cache
- ❌ `venv/` - Python virtual environments
- ❌ `env/` - Python virtual environments
- ❌ `.venv/` - Python virtual environments

### Environment & IDE Files
- ❌ `.env` - Environment variables (sensitive)
- ❌ `.env.local` - Local environment overrides
- ❌ `.env.development` - Development environment files
- ❌ `.env.production` - Production environment files
- ❌ `.vscode/` - VS Code settings (unless shared team settings)
- ❌ `.idea/` - JetBrains IDE settings
- ❌ `*.swp` - Vim swap files
- ❌ `.DS_Store` - macOS system files
- ❌ `Thumbs.db` - Windows thumbnail cache

### Logs & Temporary Files
- ❌ `*.log` - Log files
- ❌ `logs/` - Log directories
- ❌ `tmp/` - Temporary directories
- ❌ `temp/` - Temporary directories
- ❌ `*.tmp` - Temporary files
- ❌ `k.txt` - Unknown/temporary files

### User Data & Generated Content
- ❌ `uploads/` - Any upload directories
- ❌ `downloads/` - Download directories
- ❌ `*.wav` - Audio files (unless sample/demo files in models)
- ❌ `*.mp3` - Audio files (unless sample/demo files)
- ❌ `*.m4a` - Audio files (unless sample/demo files)
- ❌ `*.flac` - Audio files (unless sample/demo files)
- ❌ `*.ogg` - Audio files (unless sample/demo files)

---

## ✅ Files That Should ALWAYS Be Included

### Core Application Files
- ✅ Source code (`.py`, `.js`, `.jsx`, `.css`, `.html`)
- ✅ Configuration templates (`.env.example`)
- ✅ Documentation (`.md` files)
- ✅ License files
- ✅ Package manifests (`package.json`, `requirements.txt`, `pyproject.toml`)
- ✅ Static assets (images, icons that are part of the app UI)
- ✅ Configuration files (non-sensitive)

### Required for 3rd Party Reproduction
Based on the README installation instructions, these files are **ESSENTIAL** for someone to download and reproduce the app:

#### Root Level
- ✅ `package.json` & `package-lock.json` - For `npm install`
- ✅ `requirements.txt` - For Python dependencies
- ✅ `README.md` - Installation and usage instructions
- ✅ `LICENSE` - Legal requirements
- ✅ All Docker files for containerized deployment
- ✅ All configuration templates

#### Backend Application
- ✅ All Python source code in `backend/`
- ✅ `backend/requirements.txt` - Backend dependencies
- ✅ `backend/env.example` - Configuration template
- ✅ All static assets for the web interface

#### Frontend Application
- ✅ All React source code in `frontend/src/`
- ✅ `frontend/package.json` - Frontend dependencies
- ✅ `frontend/vite.config.js` - Build configuration
- ✅ All component files and styles

#### VibeVoice Model
- ✅ All source code in `models/VibeVoice/vibevoice/`
- ✅ `models/VibeVoice/pyproject.toml` - For `pip install -e .`
- ✅ Configuration files and documentation
- ✅ All images and figures for documentation

#### Utility Scripts
- ✅ All scripts in `scripts/` directory for development and testing

---

## 🔧 How to Use This Guide

1. **Before committing:** Check this list to ensure you're not uploading sensitive or unnecessary files
2. **Update .gitignore:** Add patterns for any ❌ marked files to your `.gitignore`
3. **Review regularly:** Update this document when new file types are added to the project
4. **Team reference:** Share this with team members to maintain consistent practices

**Remember:** When in doubt, ask yourself:
- Does this contain sensitive information? → ❌ Don't upload
- Is this auto-generated and can be recreated? → ❌ Don't upload  
- Is this user-specific data? → ❌ Don't upload
- Is this required for someone else to reproduce the app? → ✅ Safe to upload
- Is this part of the source code or documentation? → ✅ Safe to upload

---

*This document helps maintain a clean, secure, and efficient GitHub repository for DJZ-VibeVoice v1.2.0 while ensuring all essential files for reproduction are included.*

## 🚀 v1.2.0 Production Ready

This guide reflects the current v1.2.0 state with:
- ✅ **Production deployment infrastructure** (DEPLOYMENT.md)
- ✅ **Version 1.2.0 roadmap** (VERSION_1.2.0_PLAN.md)  
- ✅ **Advanced CUDA optimizations** (backend/app/utils/)
- ❌ **Production environment files** (backend/.env.production)
- ❌ **Build artifacts** (frontend/dist/, node_modules/)
- ❌ **User data protection** (data/ directory)

**Ready for GitHub upload with proper security and clean repository structure.**

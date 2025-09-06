"""Vectorized audio processing using CuPy for GPU acceleration."""

import numpy as np
import torch
import librosa
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import soundfile as sf
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import CuPy with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy available for GPU-accelerated audio processing")
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to NumPy
    logger.warning("CuPy not available - using NumPy fallback")


class VectorizedAudioProcessor:
    """GPU-accelerated audio processing with vectorized operations."""
    
    def __init__(self, device: str = "cuda", sample_rate: int = 24000):
        self.device = device
        self.sample_rate = sample_rate
        self.use_gpu = device.startswith("cuda") and CUPY_AVAILABLE and cp.cuda.is_available()
        
        if self.use_gpu:
            logger.info("Initializing GPU-accelerated audio processing with CuPy")
            self._init_gpu_kernels()
        else:
            logger.info("Using CPU fallback for audio processing")
            
        # Pre-computed lookup tables
        self._fft_cache = {}
        self._window_cache = {}
        self._mel_filter_cache = {}
    
    def _init_gpu_kernels(self):
        """Initialize custom CUDA kernels for audio operations."""
        if not self.use_gpu:
            return
            
        try:
            # Custom kernel for audio normalization
            self.normalize_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void normalize_audio(float* audio, float* output, int n, float target_rms) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n) {
                    float rms = sqrtf(audio[tid] * audio[tid]);
                    output[tid] = audio[tid] * (target_rms / fmaxf(rms, 1e-8f));
                }
            }
            ''', 'normalize_audio')
            
            # Custom kernel for spectral envelope extraction
            self.spectral_envelope_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void extract_spectral_envelope(float* magnitude, float* envelope, 
                                         int n_frames, int n_bins, int window_size) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n_frames * n_bins) {
                    int frame = tid / n_bins;
                    int bin = tid % n_bins;
                    
                    float sum = 0.0f;
                    int start = max(0, bin - window_size/2);
                    int end = min(n_bins, bin + window_size/2 + 1);
                    
                    for (int i = start; i < end; i++) {
                        sum += magnitude[frame * n_bins + i];
                    }
                    envelope[tid] = sum / (end - start);
                }
            }
            ''', 'extract_spectral_envelope')
            
            logger.info("Custom CUDA kernels initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize custom kernels: {e}")
            self.normalize_kernel = None
            self.spectral_envelope_kernel = None
    
    @lru_cache(maxsize=128)
    def _get_mel_filters(self, n_mels: int, n_fft: int) -> cp.ndarray:
        """Get cached mel filter bank on GPU."""
        cache_key = (n_mels, n_fft, self.sample_rate)
        if cache_key not in self._mel_filter_cache:
            # Generate mel filters on CPU first
            mel_filters = librosa.filters.mel(
                sr=self.sample_rate, 
                n_fft=n_fft, 
                n_mels=n_mels
            ).astype(np.float32)
            
            if self.use_gpu:
                self._mel_filter_cache[cache_key] = cp.asarray(mel_filters)
            else:
                self._mel_filter_cache[cache_key] = mel_filters
                
        return self._mel_filter_cache[cache_key]
    
    def load_audio_batch(self, audio_paths: List[str]) -> Tuple[cp.ndarray, List[int]]:
        """Load multiple audio files in parallel on GPU."""
        if not self.use_gpu:
            return self._load_audio_batch_cpu(audio_paths)
        
        audio_data = []
        lengths = []
        
        # Load all files to CPU first (I/O bound)
        cpu_audio = []
        for path in audio_paths:
            try:
                audio, sr = sf.read(str(path))
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                cpu_audio.append(audio.astype(np.float32))
                lengths.append(len(audio))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                cpu_audio.append(np.zeros(self.sample_rate, dtype=np.float32))
                lengths.append(self.sample_rate)
        
        # Transfer to GPU and pad to same length
        max_length = max(lengths) if lengths else self.sample_rate
        batch_audio = np.zeros((len(cpu_audio), max_length), dtype=np.float32)
        
        for i, audio in enumerate(cpu_audio):
            batch_audio[i, :len(audio)] = audio
        
        return cp.asarray(batch_audio), lengths
    
    def _load_audio_batch_cpu(self, audio_paths: List[str]) -> Tuple[np.ndarray, List[int]]:
        """CPU fallback for audio loading."""
        audio_data = []
        lengths = []
        
        for path in audio_paths:
            try:
                audio, sr = sf.read(str(path))
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                audio_data.append(audio.astype(np.float32))
                lengths.append(len(audio))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                audio_data.append(np.zeros(self.sample_rate, dtype=np.float32))
                lengths.append(self.sample_rate)
        
        # Pad to same length
        max_length = max(lengths) if lengths else self.sample_rate
        batch_audio = np.zeros((len(audio_data), max_length), dtype=np.float32)
        
        for i, audio in enumerate(audio_data):
            batch_audio[i, :len(audio)] = audio
        
        return batch_audio, lengths
    
    def extract_voice_features_batch(self, audio_batch: cp.ndarray, 
                                   lengths: List[int]) -> Dict[str, cp.ndarray]:
        """Extract voice features for multiple audio samples simultaneously."""
        if not self.use_gpu:
            return self._extract_voice_features_batch_cpu(audio_batch, lengths)
        
        batch_size, max_length = audio_batch.shape
        
        # Vectorized STFT computation
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        
        # Apply windowing function (CuPy doesn't have hann, use numpy and convert)
        import numpy as np
        window = cp.asarray(np.hann(win_length))
        
        # Compute STFT for entire batch using broadcasting
        stft_batch = self._batch_stft(audio_batch, n_fft, hop_length, win_length, window)
        magnitude = cp.abs(stft_batch)
        phase = cp.angle(stft_batch)
        
        # Extract mel spectrograms
        mel_filters = self._get_mel_filters(n_mels=80, n_fft=n_fft)
        mel_specs = cp.dot(mel_filters, magnitude.reshape(-1, magnitude.shape[-1]))
        mel_specs = mel_specs.reshape(batch_size, 80, -1)
        
        # Extract spectral features using vectorized operations
        features = {
            'magnitude': magnitude,
            'phase': phase,
            'mel_spectrogram': mel_specs,
            'spectral_centroid': self._batch_spectral_centroid(magnitude),
            'spectral_bandwidth': self._batch_spectral_bandwidth(magnitude),
            'zero_crossing_rate': self._batch_zero_crossing_rate(audio_batch),
            'mfcc': self._batch_mfcc(mel_specs),
        }
        
        return features
    
    def _extract_voice_features_batch_cpu(self, audio_batch: np.ndarray, 
                                        lengths: List[int]) -> Dict[str, np.ndarray]:
        """CPU fallback for feature extraction."""
        batch_size, max_length = audio_batch.shape
        
        # Basic feature extraction using librosa
        features = {
            'magnitude': [],
            'phase': [],
            'mel_spectrogram': [],
            'spectral_centroid': [],
            'spectral_bandwidth': [],
            'zero_crossing_rate': [],
            'mfcc': [],
        }
        
        for i in range(batch_size):
            audio = audio_batch[i, :lengths[i]]
            
            # STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=80, n_fft=2048, hop_length=512
            )
            
            # Other features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            zcr = librosa.feature.zero_crossing_rate(audio)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            features['magnitude'].append(magnitude)
            features['phase'].append(phase)
            features['mel_spectrogram'].append(mel_spec)
            features['spectral_centroid'].append(spectral_centroid)
            features['spectral_bandwidth'].append(spectral_bandwidth)
            features['zero_crossing_rate'].append(zcr)
            features['mfcc'].append(mfcc)
        
        # Convert lists to arrays
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def _batch_stft(self, audio_batch: cp.ndarray, n_fft: int, hop_length: int, 
                   win_length: int, window: cp.ndarray) -> cp.ndarray:
        """Compute STFT for entire batch using vectorized operations."""
        batch_size, signal_length = audio_batch.shape
        n_frames = 1 + (signal_length - win_length) // hop_length
        
        # Create output array
        stft_output = cp.zeros((batch_size, n_fft // 2 + 1, n_frames), dtype=cp.complex64)
        
        # Vectorized frame extraction and windowing
        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            end = start + win_length
            
            # Extract frame for all samples in batch
            frame_batch = audio_batch[:, start:end]  # Shape: (batch_size, win_length)
            
            # Apply window to all frames
            windowed_frames = frame_batch * window[None, :]  # Broadcasting
            
            # Pad to n_fft
            if win_length < n_fft:
                padding = n_fft - win_length
                windowed_frames = cp.pad(windowed_frames, ((0, 0), (0, padding)))
            
            # Compute FFT for all frames in batch
            fft_result = cp.fft.fft(windowed_frames, n=n_fft, axis=1)
            stft_output[:, :, frame_idx] = fft_result[:, :n_fft // 2 + 1]
        
        return stft_output
    
    def _batch_spectral_centroid(self, magnitude: cp.ndarray) -> cp.ndarray:
        """Compute spectral centroid for batch using vectorized operations."""
        freqs = cp.fft.fftfreq(magnitude.shape[1] * 2 - 2, 1/self.sample_rate)[:magnitude.shape[1]]
        
        # Vectorized computation across batch
        numerator = cp.sum(magnitude * freqs[None, :, None], axis=1)
        denominator = cp.sum(magnitude, axis=1)
        
        return numerator / (denominator + 1e-8)
    
    def _batch_spectral_bandwidth(self, magnitude: cp.ndarray) -> cp.ndarray:
        """Compute spectral bandwidth for batch using vectorized operations."""
        freqs = cp.fft.fftfreq(magnitude.shape[1] * 2 - 2, 1/self.sample_rate)[:magnitude.shape[1]]
        centroid = self._batch_spectral_centroid(magnitude)
        
        # Vectorized bandwidth computation
        diff_sq = (freqs[None, :, None] - centroid[:, None, :]) ** 2
        numerator = cp.sum(magnitude * diff_sq, axis=1)
        denominator = cp.sum(magnitude, axis=1)
        
        return cp.sqrt(numerator / (denominator + 1e-8))
    
    def _batch_zero_crossing_rate(self, audio_batch: cp.ndarray) -> cp.ndarray:
        """Compute zero crossing rate for batch using vectorized operations."""
        # Sign changes detection
        signs = cp.sign(audio_batch)
        sign_changes = cp.abs(cp.diff(signs, axis=1))
        
        # Count zero crossings per frame
        frame_length = 1024
        hop_length = 512
        n_frames = 1 + (audio_batch.shape[1] - frame_length) // hop_length
        
        zcr = cp.zeros((audio_batch.shape[0], n_frames))
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, sign_changes.shape[1])
            zcr[:, i] = cp.sum(sign_changes[:, start:end], axis=1) / (2 * (end - start))
        
        return zcr
    
    def _batch_mfcc(self, mel_specs: cp.ndarray, n_mfcc: int = 13) -> cp.ndarray:
        """Compute MFCC for batch using vectorized DCT."""
        # Log mel spectrogram
        log_mel = cp.log(mel_specs + 1e-8)
        
        # DCT-II for MFCC computation
        batch_size, n_mels, n_frames = log_mel.shape
        
        # Create DCT matrix
        dct_matrix = cp.zeros((n_mfcc, n_mels))
        for i in range(n_mfcc):
            for j in range(n_mels):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / cp.sqrt(n_mels)
                else:
                    dct_matrix[i, j] = cp.sqrt(2.0 / n_mels) * cp.cos(
                        cp.pi * i * (2 * j + 1) / (2 * n_mels)
                    )
        
        # Apply DCT to all frames in batch
        mfcc = cp.zeros((batch_size, n_mfcc, n_frames))
        for frame_idx in range(n_frames):
            mfcc[:, :, frame_idx] = cp.dot(dct_matrix, log_mel[:, :, frame_idx].T).T
        
        return mfcc
    
    def normalize_audio_batch(self, audio_batch: cp.ndarray, target_rms: float = 0.1) -> cp.ndarray:
        """Normalize audio batch using custom CUDA kernel."""
        if not self.use_gpu or self.normalize_kernel is None:
            # CPU fallback
            if self.use_gpu:
                audio_batch = cp.asnumpy(audio_batch)
            normalized = np.zeros_like(audio_batch)
            for i in range(len(audio_batch)):
                rms = np.sqrt(np.mean(audio_batch[i] ** 2))
                normalized[i] = audio_batch[i] * (target_rms / max(rms, 1e-8))
            return cp.asarray(normalized) if self.use_gpu else normalized
        
        output = cp.zeros_like(audio_batch)
        
        # Launch custom kernel for each sample in batch
        for i in range(audio_batch.shape[0]):
            block_size = 256
            grid_size = (audio_batch.shape[1] + block_size - 1) // block_size
            
            self.normalize_kernel(
                (grid_size,), (block_size,),
                (audio_batch[i], output[i], audio_batch.shape[1], target_rms)
            )
        
        return output
    
    def to_torch_tensor(self, cupy_array: cp.ndarray, device: str) -> torch.Tensor:
        """Convert CuPy array to PyTorch tensor with zero-copy when possible."""
        if self.use_gpu and device.startswith("cuda"):
            # Zero-copy conversion from CuPy to PyTorch
            return torch.as_tensor(cupy_array, device=device)
        else:
            # Transfer to CPU then to PyTorch
            cpu_array = cp.asnumpy(cupy_array) if self.use_gpu else cupy_array
            return torch.from_numpy(cpu_array).to(device)


# Global processor instance
vectorized_processor = VectorizedAudioProcessor()

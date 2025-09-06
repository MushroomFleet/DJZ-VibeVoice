"""Zero-copy streaming audio pipeline for real-time processing."""

import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import logging
from typing import Optional, Generator, List, Dict, Any, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data in the streaming pipeline."""
    data: torch.Tensor
    sample_rate: int
    chunk_id: int
    timestamp: float
    metadata: Dict[str, Any] = None


class RingBuffer:
    """GPU-memory ring buffer for streaming audio processing."""
    
    def __init__(self, max_size: int, chunk_size: int, device: str = "cuda"):
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.device = device
        
        # Pre-allocate GPU memory
        self.buffer = torch.zeros(
            (max_size, chunk_size), 
            dtype=torch.float32, 
            device=device
        )
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = threading.Lock()
    
    def push(self, chunk: torch.Tensor) -> bool:
        """Add audio chunk to buffer."""
        with self.lock:
            if self.count >= self.max_size:
                # Buffer full, advance read pointer (overwrite old data)
                self.read_idx = (self.read_idx + 1) % self.max_size
                self.count -= 1
            
            # Copy chunk to pre-allocated memory
            if chunk.device != self.device:
                chunk = chunk.to(self.device, non_blocking=True)
            
            # Ensure chunk fits in buffer
            if chunk.numel() <= self.chunk_size:
                self.buffer[self.write_idx, :chunk.numel()] = chunk.flatten()
            else:
                # Truncate if too large
                self.buffer[self.write_idx] = chunk.flatten()[:self.chunk_size]
                
            self.write_idx = (self.write_idx + 1) % self.max_size
            self.count += 1
            return True
    
    def pop(self) -> Optional[torch.Tensor]:
        """Get next audio chunk from buffer."""
        with self.lock:
            if self.count == 0:
                return None
            
            chunk = self.buffer[self.read_idx].clone()
            self.read_idx = (self.read_idx + 1) % self.max_size
            self.count -= 1
            return chunk
    
    def get_available_space(self) -> int:
        """Get number of available slots in buffer."""
        return self.max_size - self.count


class StreamingAudioProcessor:
    """Streaming audio processor with zero-copy operations."""
    
    def __init__(self, chunk_size: int = 4096, buffer_size: int = 64, device: str = "cuda"):
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.device = device
        
        # Pre-allocated buffers
        self.input_buffer = RingBuffer(buffer_size, chunk_size, device)
        self.output_buffer = RingBuffer(buffer_size, chunk_size, device)
        
        # Processing thread
        self.processing_thread = None
        self.should_stop = threading.Event()
        self.is_running = False
        
        # Pre-allocated tensors for processing
        self._init_processing_tensors()
        
        # Performance metrics
        self.chunks_processed = 0
        self.total_processing_time = 0.0
        self.last_benchmark_time = time.time()
    
    def _init_processing_tensors(self):
        """Initialize pre-allocated tensors for processing."""
        self.temp_chunk = torch.zeros(
            self.chunk_size, dtype=torch.float32, device=self.device
        )
        self.window = torch.hann_window(
            self.chunk_size, device=self.device
        )
        self.fft_buffer = torch.zeros(
            self.chunk_size // 2 + 1, dtype=torch.complex64, device=self.device
        )
    
    def start_processing(self, processor_func: Callable[[torch.Tensor], torch.Tensor]):
        """Start streaming processing thread."""
        if self.is_running:
            return
        
        self.processor_func = processor_func
        self.should_stop.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.is_running = True
        logger.info("Streaming audio processor started")
    
    def stop_processing(self):
        """Stop streaming processing thread."""
        if not self.is_running:
            return
        
        self.should_stop.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.is_running = False
        logger.info("Streaming audio processor stopped")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        while not self.should_stop.is_set():
            # Get input chunk
            input_chunk = self.input_buffer.pop()
            if input_chunk is None:
                time.sleep(0.001)  # 1ms sleep if no data
                continue
            
            try:
                start_time = time.perf_counter()
                
                # Process chunk using provided function
                processed_chunk = self.processor_func(input_chunk)
                
                # Add to output buffer
                self.output_buffer.push(processed_chunk)
                
                # Update metrics
                self.chunks_processed += 1
                self.total_processing_time += time.perf_counter() - start_time
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
    
    def push_audio(self, audio: torch.Tensor) -> bool:
        """Add audio data to input buffer."""
        return self.input_buffer.push(audio)
    
    def get_processed_audio(self) -> Optional[torch.Tensor]:
        """Get processed audio from output buffer."""
        return self.output_buffer.pop()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current buffer and performance statistics."""
        current_time = time.time()
        time_delta = current_time - self.last_benchmark_time
        
        stats = {
            "input_buffer_count": self.input_buffer.count,
            "output_buffer_count": self.output_buffer.count,
            "input_buffer_available": self.input_buffer.get_available_space(),
            "output_buffer_available": self.output_buffer.get_available_space(),
            "chunks_processed": self.chunks_processed,
            "avg_processing_time_ms": (
                (self.total_processing_time / self.chunks_processed * 1000) 
                if self.chunks_processed > 0 else 0
            ),
            "processing_rate_chunks_per_sec": (
                self.chunks_processed / time_delta if time_delta > 0 else 0
            ),
            "is_running": self.is_running,
        }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.chunks_processed = 0
        self.total_processing_time = 0.0
        self.last_benchmark_time = time.time()


class VoiceFeatureCache:
    """LRU cache for pre-computed voice features."""
    
    def __init__(self, max_size: int = 100, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, voice_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached voice features."""
        with self.lock:
            if voice_id in self.cache:
                # Move to end of access order
                self.access_order.remove(voice_id)
                self.access_order.append(voice_id)
                self.hit_count += 1
                return self.cache[voice_id]
            
            self.miss_count += 1
            return None
    
    def put(self, voice_id: str, features: Dict[str, torch.Tensor]):
        """Cache voice features with LRU eviction."""
        with self.lock:
            # Ensure features are on correct device
            device_features = {}
            for key, tensor in features.items():
                if isinstance(tensor, torch.Tensor):
                    device_features[key] = tensor.to(self.device, non_blocking=True)
                else:
                    device_features[key] = tensor
            
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and voice_id not in self.cache:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
                logger.debug(f"Evicted voice features for: {oldest}")
            
            # Add/update cache
            self.cache[voice_id] = device_features
            if voice_id in self.access_order:
                self.access_order.remove(voice_id)
            self.access_order.append(voice_id)
            
            logger.debug(f"Cached voice features for: {voice_id}")
    
    def clear(self):
        """Clear all cached features."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Voice feature cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "cached_voices": list(self.cache.keys())
            }


class StreamingVoiceProcessor:
    """High-level streaming voice processing interface."""
    
    def __init__(self, voice_service, chunk_duration_ms: int = 100):
        self.voice_service = voice_service
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = 24000  # VibeVoice sample rate
        
        # Calculate chunk size
        self.chunk_size = int(self.sample_rate * chunk_duration_ms / 1000)
        
        # Initialize streaming processor
        self.processor = StreamingAudioProcessor(
            chunk_size=self.chunk_size,
            buffer_size=32,  # Smaller buffer for real-time
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Voice feature cache
        self.feature_cache = VoiceFeatureCache(max_size=50)
        
        # Current processing parameters
        self.current_voice_id = None
        self.current_text = None
        
    def start_streaming(self, voice_id: str, text: str):
        """Start streaming voice synthesis."""
        self.current_voice_id = voice_id
        self.current_text = text
        
        # Pre-compute voice features if not cached
        if self.feature_cache.get(voice_id) is None:
            self._precompute_voice_features(voice_id)
        
        # Start processing with voice synthesis function
        self.processor.start_processing(self._process_audio_chunk)
        
        logger.info(f"Started streaming synthesis for voice: {voice_id}")
    
    def stop_streaming(self):
        """Stop streaming voice synthesis."""
        self.processor.stop_processing()
        logger.info("Stopped streaming synthesis")
    
    def _precompute_voice_features(self, voice_id: str):
        """Pre-compute and cache voice features."""
        voice_profile = self.voice_service.get_voice_profile(voice_id)
        if not voice_profile:
            logger.error(f"Voice profile not found: {voice_id}")
            return
        
        try:
            # Use vectorized processor to extract features
            from app.utils.vectorized_audio import vectorized_processor
            
            voice_batch, voice_lengths = vectorized_processor.load_audio_batch([voice_profile.file_path])
            features = vectorized_processor.extract_voice_features_batch(voice_batch, voice_lengths)
            
            # Convert to torch tensors and cache
            torch_features = {}
            for key, value in features.items():
                if hasattr(value, 'shape'):  # CuPy or NumPy array
                    torch_features[key] = vectorized_processor.to_torch_tensor(value[0], self.processor.device)
            
            self.feature_cache.put(voice_id, torch_features)
            logger.info(f"Pre-computed features for voice: {voice_id}")
            
        except Exception as e:
            logger.error(f"Failed to pre-compute voice features: {e}")
    
    def _process_audio_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """Process a single audio chunk (placeholder for actual synthesis)."""
        # This is a simplified version - real implementation would use
        # the voice service with cached features for streaming synthesis
        
        # For now, just apply a simple transformation
        processed = audio_chunk * 0.8  # Volume adjustment
        return processed
    
    def feed_text_chunk(self, text_chunk: str) -> bool:
        """Feed a chunk of text for processing."""
        # Convert text to audio features and feed to processor
        # This would require chunked text-to-speech processing
        
        # For now, generate a dummy audio chunk
        dummy_audio = torch.randn(self.chunk_size, device=self.processor.device) * 0.1
        return self.processor.push_audio(dummy_audio)
    
    def get_output_chunk(self) -> Optional[torch.Tensor]:
        """Get next processed audio chunk."""
        return self.processor.get_processed_audio()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        processor_stats = self.processor.get_processing_stats()
        cache_stats = self.feature_cache.get_stats()
        
        return {
            "processor": processor_stats,
            "feature_cache": cache_stats,
            "chunk_duration_ms": self.chunk_duration_ms,
            "chunk_size": self.chunk_size,
            "sample_rate": self.sample_rate,
            "current_voice": self.current_voice_id,
        }


# Global instances
streaming_processor = StreamingAudioProcessor()
voice_feature_cache = VoiceFeatureCache()

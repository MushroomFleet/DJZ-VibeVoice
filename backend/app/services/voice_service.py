import os
import re
import logging
from typing import Optional, List, Dict, Tuple
import uuid
import numpy as np

import torch

from app.config import settings
from app.models import VoiceProfile, VoiceType
from app.utils.cuda_utils import cuda_manager
from app.utils.tensor_pools import tensor_pool_manager, ContextualTensorManager
from app.utils.vectorized_audio import vectorized_processor
from app.utils.memory_optimizer import adaptive_memory_manager

logger = logging.getLogger(__name__)

# Silence the fork/parallelism warning from HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class SpeakerAssignment:
    """Represents a voice assignment to a speaker slot."""
    def __init__(self, speaker_id: int, voice_id: str, voice_profile: VoiceProfile):
        self.speaker_id = speaker_id
        self.voice_id = voice_id
        self.voice_profile = voice_profile


class MultiSpeakerSession:
    """Manages multi-speaker voice assignments for a session."""
    def __init__(self, voice_service: 'VoiceService'):
        self.voice_service = voice_service
        self.assignments: Dict[int, SpeakerAssignment] = {}
        self.max_speakers = 4

    def assign_voice(self, speaker_id: int, voice_id: str) -> bool:
        """Assign a voice to a speaker slot (1-4)."""
        if not (1 <= speaker_id <= self.max_speakers):
            raise ValueError(f"Speaker ID must be between 1 and {self.max_speakers}")
        
        voice_profile = self.voice_service.get_voice_profile(voice_id)
        if not voice_profile:
            raise ValueError(f"Voice profile {voice_id} not found")
        
        self.assignments[speaker_id] = SpeakerAssignment(speaker_id, voice_id, voice_profile)
        return True

    def clear_assignment(self, speaker_id: int) -> bool:
        """Clear voice assignment for a speaker slot."""
        if speaker_id in self.assignments:
            del self.assignments[speaker_id]
            return True
        return False

    def get_voice_array(self, referenced_speakers: List[int]) -> List[str]:
        """Build ordered voice sample array for referenced speakers."""
        voice_paths = []
        for speaker_id in referenced_speakers:
            if speaker_id not in self.assignments:
                raise ValueError(f"No voice assigned to Speaker {speaker_id}")
            voice_paths.append(self.assignments[speaker_id].voice_profile.file_path)
        return voice_paths

    def validate_assignments(self, referenced_speakers: List[int]) -> bool:
        """Check if all referenced speakers have voice assignments."""
        for speaker_id in referenced_speakers:
            if speaker_id not in self.assignments:
                return False
        return True

    def get_assignments_dict(self) -> Dict[int, Dict[str, str]]:
        """Get current assignments as a dictionary."""
        return {
            speaker_id: {
                'voice_id': assignment.voice_id,
                'voice_name': assignment.voice_profile.name
            }
            for speaker_id, assignment in self.assignments.items()
        }


class VoiceService:
    """Service for voice synthesis operations."""

    def __init__(self):
        """Initialize the voice service."""
        self.model = None
        self.processor = None
        self.voices_cache: Dict[str, VoiceProfile] = {}
        self.model_loaded = False
        self.multi_speaker_session = MultiSpeakerSession(self)
        self._initialize_model()
        self._load_voices()
        
        # Register memory optimization callback
        adaptive_memory_manager.register_optimization_callback(
            self._handle_memory_optimization
        )

    def _initialize_model(self):
        """Initialize the VibeVoice model with optimized CUDA settings."""
        try:
            try:
                logger.info("Attempting to import VibeVoice modules...")
                from vibevoice.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference,
                )
                from vibevoice.processor.vibevoice_processor import (
                    VibeVoiceProcessor,
                )
                logger.info("VibeVoice modules imported successfully!")
            except ImportError as import_error:
                logger.error(f"VibeVoice import failed: {import_error}")
                logger.error(
                    "VibeVoice not installed. Install with:\n"
                    "  git clone https://github.com/microsoft/VibeVoice.git\n"
                    "  cd VibeVoice && pip install -e ."
                )
                return

            # Setup CUDA optimizations
            cuda_manager.setup_memory_optimization()
            
            device = cuda_manager.device
            dtype = cuda_manager.dtype
            
            logger.info(f"Loading model from {settings.MODEL_PATH} on device={device} dtype={dtype}")
            
            # Get optimized loading arguments
            load_kwargs = cuda_manager.get_model_load_kwargs()
            
            # Load processor with fallback handling
            try:
                # Use proper HuggingFace resolution to avoid preprocessor_config.json warning
                from huggingface_hub import snapshot_download
                resolved_model_path = snapshot_download(settings.MODEL_PATH)
                logger.info(f"Resolved model path: {resolved_model_path}")
                self.processor = VibeVoiceProcessor.from_pretrained(resolved_model_path)
            except Exception as proc_error:
                logger.warning(f"Failed to load processor from {settings.MODEL_PATH}: {proc_error}")
                logger.info("Attempting fallback with original model path...")
                try:
                    self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise proc_error
                logger.info("Creating processor with default configuration...")
                # Create processor with default settings if config is missing
                from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
                from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
                
                tokenizer = VibeVoiceTextTokenizerFast.from_pretrained("Qwen/Qwen2.5-1.5B")
                audio_processor = VibeVoiceTokenizerProcessor()
                
                self.processor = VibeVoiceProcessor(
                    tokenizer=tokenizer,
                    audio_processor=audio_processor,
                    speech_tok_compress_ratio=3200,
                    db_normalize=True,
                )
            
            # Load model with optimizations
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    settings.MODEL_PATH,
                    **load_kwargs,
                )
            except Exception as e1:
                logger.warning(f"Optimized load failed ({e1}). Retrying with basic settings.")
                # Fallback without optimizations
                basic_kwargs = {"torch_dtype": dtype}
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    settings.MODEL_PATH,
                    **basic_kwargs,
                )
            
            # Move model to device and set eval mode
            self.model.to(device)
            self.model.eval()
            
            # Configure inference settings
            try:
                self.model.set_ddpm_inference_steps(num_steps=10)
            except Exception:
                pass
            
            # Enable compilation for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and device.startswith("cuda"):
                try:
                    logger.info("Compiling model for optimized inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compilation successful")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            self.model_loaded = True
            
            # Log memory usage
            memory_info = cuda_manager.get_memory_info()
            if memory_info:
                logger.info(f"GPU memory after model load: {memory_info['allocated_gb']:.1f}GB allocated, "
                           f"{memory_info['utilization_percent']:.1f}% utilization")
            
            logger.info("Model loaded successfully with CUDA optimizations.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model_loaded = False

    def _load_voices(self):
        """Load available voice profiles from the voices directory."""
        settings.VOICES_DIR.mkdir(exist_ok=True)

        voice_files = []
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
            voice_files.extend(settings.VOICES_DIR.glob(ext))

        if not voice_files:
            logger.info(
                "No voice files found in voices/ yet. Upload or record to add voices."
            )

        for voice_file in voice_files:
            voice_id = str(uuid.uuid4())
            profile = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file),
            )
            self.voices_cache[voice_id] = profile
            logger.info(f"Loaded voice: {voice_file.stem}")

    def _manage_memory_for_generation(self, clear_cache: bool = True):
        """Manage GPU memory before generation."""
        if clear_cache:
            cuda_manager.clear_memory()
        
        memory_info = cuda_manager.get_memory_info()
        if memory_info and memory_info['utilization_percent'] > 90:
            logger.warning("High GPU memory usage detected, clearing cache")
            cuda_manager.clear_memory()

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        num_speakers: int = 1,
        cfg_scale: float = 1.3,
        use_optimizations: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate speech with configurable optimizations.
        
        Args:
            text: Text to synthesize
            voice_id: Voice profile ID to use
            num_speakers: Number of speakers (for multi-speaker mode)
            cfg_scale: Classifier-free guidance scale
            use_optimizations: If True, use optimized generation; If False, use original behavior;
                             If None, use AUTO_USE_OPTIMIZATIONS setting
        
        Returns:
            Generated audio array or None if failed
        """
        # Determine optimization usage
        if use_optimizations is None:
            use_optimizations = settings.AUTO_USE_OPTIMIZATIONS
        
        # Route to appropriate generation method
        if settings.PRESERVE_ORIGINAL_BEHAVIOR and not use_optimizations:
            return self._generate_speech_original(text, voice_id, num_speakers, cfg_scale)
        else:
            return self._generate_speech_optimized(text, voice_id, num_speakers, cfg_scale)

    def _generate_speech_original(
        self,
        text: str,
        voice_id: str,
        num_speakers: int = 1,
        cfg_scale: float = 1.3,
    ) -> Optional[np.ndarray]:
        """Generate speech using original VibeVoice behavior (no forced parameters)."""
        try:
            # Validate voice
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            # If model unavailable, return placeholder audio
            if not (self.model_loaded and self.model and self.processor):
                logger.warning("Model not loaded — returning sample placeholder audio.")
                return self._generate_sample_audio(text)

            logger.info(f"Generating speech (ORIGINAL mode) with voice: {voice_profile.name}")
            logger.info(f"Voice file path: {voice_profile.file_path}")
            
            # Verify voice file exists and is accessible
            if not os.path.exists(voice_profile.file_path):
                logger.error(f"Voice file not found: {voice_profile.file_path}")
                return self._generate_sample_audio(text)
            
            # Format text for multi-speaker scenarios
            formatted_text = self._format_text_for_speakers(text, num_speakers)
            logger.info(f"Formatted text: {formatted_text}")

            # Prepare inputs with minimal processing (original behavior)
            logger.info("Processing voice samples through VibeVoiceProcessor...")
            try:
                inputs = self.processor(
                    text=[formatted_text],
                    voice_samples=[[voice_profile.file_path]],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            except Exception as proc_error:
                logger.error(f"Voice processor failed: {proc_error}")
                return self._generate_sample_audio(text)

            # Move inputs to device efficiently
            device = cuda_manager.device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device, non_blocking=True)

            logger.info(f"Starting ORIGINAL generation with cfg_scale={cfg_scale}")

            # Original VibeVoice generation (no forced parameters)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,  # Use user's exact cfg_scale
                tokenizer=self.processor.tokenizer,
                # NO generation_config - use model defaults
                verbose=False,
            )

            # Extract and process audio
            if (
                getattr(outputs, "speech_outputs", None)
                and outputs.speech_outputs[0] is not None
            ):
                audio_tensor = outputs.speech_outputs[0]

                # Cast to float32 for NumPy compatibility
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)

                # Move to CPU and convert to NumPy
                audio_array = audio_tensor.detach().cpu().numpy()
                audio_array = np.clip(audio_array, -1.0, 1.0)

                logger.info("Original generation completed successfully")
                return audio_array

            logger.error("No speech output generated by the model.")
            return None

        except Exception as e:
            logger.error(f"Original speech generation error: {e}", exc_info=True)
            return self._generate_sample_audio(text)

    def _generate_speech_optimized(
        self,
        text: str,
        voice_id: str,
        num_speakers: int = 1,
        cfg_scale: float = 1.3,
    ) -> Optional[np.ndarray]:
        """Generate speech using optimized CUDA behavior with all enhancements."""
        try:
            # Validate voice
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            # If model unavailable, return placeholder audio
            if not (self.model_loaded and self.model and self.processor):
                logger.warning("Model not loaded — returning sample placeholder audio.")
                return self._generate_sample_audio(text)

            # Manage memory before generation
            self._manage_memory_for_generation()

            logger.info(f"Generating speech (OPTIMIZED mode) with voice: {voice_profile.name}")
            logger.info(f"Voice file path: {voice_profile.file_path}")
            
            # Verify voice file exists and is accessible
            if not os.path.exists(voice_profile.file_path):
                logger.error(f"Voice file not found: {voice_profile.file_path}")
                return self._generate_sample_audio(text)
            
            # Check voice file size
            voice_file_size = os.path.getsize(voice_profile.file_path)
            logger.info(f"Voice file size: {voice_file_size} bytes")

            # Format text for multi-speaker scenarios
            formatted_text = self._format_text_for_speakers(text, num_speakers)
            logger.info(f"Formatted text: {formatted_text}")

            # Prepare inputs with optimizations and detailed logging
            logger.info("Processing voice samples through VibeVoiceProcessor...")
            try:
                inputs = self.processor(
                    text=[formatted_text],
                    voice_samples=[[voice_profile.file_path]],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                logger.info(f"Processor input keys: {list(inputs.keys())}")
                
                # Log details about voice processing
                if "speech_tensors" in inputs and inputs["speech_tensors"] is not None:
                    logger.info(f"Speech tensors shape: {inputs['speech_tensors'].shape}")
                    logger.info(f"Speech tensors dtype: {inputs['speech_tensors'].dtype}")
                else:
                    logger.warning("No speech_tensors found in processor inputs!")
                    
                if "speech_masks" in inputs and inputs["speech_masks"] is not None:
                    logger.info(f"Speech masks shape: {inputs['speech_masks'].shape}")
                else:
                    logger.warning("No speech_masks found in processor inputs!")
                    
            except Exception as proc_error:
                logger.error(f"Voice processor failed: {proc_error}")
                logger.error("This indicates a problem with voice sample processing")
                return self._generate_sample_audio(text)

            # Move inputs to device efficiently
            device = cuda_manager.device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device, non_blocking=True)

            logger.info("Starting generation...")

            # Ensure all input tensors match model dtype to prevent mismatch errors
            model_dtype = cuda_manager.dtype
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.dtype != model_dtype:
                    # Convert to model dtype if it's a floating point tensor
                    if v.dtype.is_floating_point:
                        inputs[k] = v.to(model_dtype)
                        logger.debug(f"Converted input {k} from {v.dtype} to {model_dtype}")

            # Configure generation parameters based on settings
            generation_config = None  # Use model defaults by default
            if settings.ENABLE_CUSTOM_GENERATION_CONFIG:
                generation_config = {
                    "do_sample": settings.GENERATION_DO_SAMPLE,
                    "temperature": settings.GENERATION_TEMPERATURE,
                    "top_p": settings.GENERATION_TOP_P,
                    "repetition_penalty": settings.GENERATION_REPETITION_PENALTY,
                }
                logger.info(f"Using custom generation config: {generation_config}")
            
            # Respect user's cfg_scale choice unless override is enabled
            if settings.ENABLE_CFG_OVERRIDE:
                effective_cfg_scale = max(cfg_scale, 1.5)  # Minimum 1.5 for voice cloning
                logger.info(f"CFG override enabled: using cfg_scale={effective_cfg_scale}")
            else:
                effective_cfg_scale = cfg_scale
                logger.info(f"Using user-specified cfg_scale: {effective_cfg_scale}")
            
            # Generate with memory-efficient settings using updated autocast API
            with torch.amp.autocast('cuda', enabled=device.startswith("cuda"), dtype=model_dtype):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=effective_cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config=generation_config,
                    verbose=False,
                    use_cache=True,  # Enable KV caching
                )

            # Extract and process audio
            if (
                getattr(outputs, "speech_outputs", None)
                and outputs.speech_outputs[0] is not None
            ):
                audio_tensor = outputs.speech_outputs[0]

                # Cast to float32 for NumPy compatibility
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)

                # Move to CPU and convert to NumPy
                audio_array = audio_tensor.detach().cpu().numpy()
                audio_array = np.clip(audio_array, -1.0, 1.0)

                logger.info("Optimized generation completed successfully")
                return audio_array

            logger.error("No speech output generated by the model.")
            return None

        except Exception as e:
            logger.error(f"Optimized speech generation error: {e}", exc_info=True)
            # Clear memory on error
            cuda_manager.clear_memory()
            return self._generate_sample_audio(text)

    def _format_text_for_speakers(self, text: str, num_speakers: int) -> str:
        """
        Ensure text has 'Speaker i:'
        prefixes when multiple speakers are requested.
        """
        if num_speakers <= 1:
            if not text.strip().startswith("Speaker"):
                return f"Speaker 0: {text}"
            return text

        lines = [ln.strip() for ln in text.splitlines()]
        formatted = []
        current = 0
        for ln in lines:
            if not ln:
                continue
            if ln.startswith("Speaker"):
                formatted.append(ln)
            else:
                formatted.append(f"Speaker {current}: {ln}")
                current = (current + 1) % num_speakers
        return "\n".join(formatted)

    def _generate_sample_audio(self, text: str) -> np.ndarray:
        """Generate a more pleasant placeholder tone (float32)."""
        duration = float(min(10.0, max(1.0, len(text) * 0.05)))  # seconds
        sr = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        freqs = [220.0, 440.0, 660.0]
        audio = sum(
            0.25 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs)
        )

        # Simple envelope
        env = np.minimum(1.0, np.linspace(0, 1.0, len(t)) * 3.0) * np.exp(-t * 0.6)
        audio = (audio * env).astype(np.float32)

        # Normalize
        peak = float(np.max(np.abs(audio))) if audio.size else 1.0
        if peak > 0:
            audio = 0.8 * (audio / peak)
        return audio

    def add_voice_profile(
        self,
        name: str,
        audio_path: str,
        voice_type: VoiceType = VoiceType.UPLOADED,
    ) -> VoiceProfile:
        """Add a new voice profile to the in-memory cache."""
        voice_id = str(uuid.uuid4())
        profile = VoiceProfile(
            id=voice_id,
            name=name,
            type=voice_type,
            file_path=audio_path,
        )
        self.voices_cache[voice_id] = profile
        logger.info(f"Added voice profile: {name} (type: {voice_type})")
        return profile

    def delete_voice_profile(self, voice_id: str) -> bool:
        """Delete a voice profile and its associated file."""
        try:
            profile = self.voices_cache.get(voice_id)
            if not profile:
                return False

            # Delete the audio file
            if os.path.exists(profile.file_path):
                os.remove(profile.file_path)
                logger.info(f"Deleted voice file: {profile.file_path}")

            # Remove from cache
            del self.voices_cache[voice_id]
            logger.info(f"Deleted voice profile: {profile.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete voice profile: {e}")
            raise

    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Return all available voice profiles."""
        return list(self.voices_cache.values())

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Return a specific voice profile by id."""
        return self.voices_cache.get(voice_id)

    def is_model_loaded(self) -> bool:
        """Return True if model is loaded."""
        return self.model_loaded

    def load_model_if_needed(self):
        """Load the model if it hasn't been loaded yet."""
        if not self.model_loaded:
            logger.info("Loading model on demand...")
            self._initialize_model()
        return self.model_loaded

    # Multi-Speaker Methods

    def _extract_referenced_speakers(self, text: str) -> List[int]:
        """Extract speaker IDs referenced in text via [1], [2], [3], [4] markers at line beginnings."""
        # Look for [X] at the beginning of lines or after whitespace
        pattern = r'(?:^|\s)\[([1-4])\]'
        matches = re.findall(pattern, text, re.MULTILINE)
        return [int(match) for match in matches]

    def _convert_marker_syntax_to_speaker_format(self, text: str) -> Tuple[str, List[int]]:
        """
        Convert [X] text format to Speaker X: text format for VibeVoice.
        Expects format: [1] first line text, [2] second line text, etc.
        Returns: (converted_text, referenced_speaker_ids)
        
        Example:
        Input: "[1] whats going on everyone, i want to introduce someone\n[2] Hey, my name's Sandy - hello all"
        Output: ("Speaker 1: whats going on everyone, i want to introduce someone\nSpeaker 2: Hey, my name's Sandy - hello all", [1, 2])
        """
        lines = text.strip().split('\n')
        converted_lines = []
        referenced_speakers = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for [X] at the beginning of the line
            match = re.match(r'^\[([1-4])\]\s*(.*)', line)
            if match:
                speaker_id = int(match.group(1))
                content = match.group(2).strip()
                referenced_speakers.append(speaker_id)
                converted_lines.append(f"Speaker {speaker_id}: {content}")
            else:
                # If no marker, treat as continuation of previous speaker or default to Speaker 1
                if not converted_lines:
                    # No previous context, default to Speaker 1
                    referenced_speakers.append(1)
                    converted_lines.append(f"Speaker 1: {line}")
                else:
                    # Continue previous speaker (extract speaker number from last line)
                    last_line = converted_lines[-1]
                    speaker_match = re.match(r'^Speaker (\d+):', last_line)
                    if speaker_match:
                        speaker_id = int(speaker_match.group(1))
                        # Append to the last speaker's content
                        converted_lines[-1] += f" {line}"
                    else:
                        # Fallback to Speaker 1
                        referenced_speakers.append(1)
                        converted_lines.append(f"Speaker 1: {line}")
        
        converted_text = '\n'.join(converted_lines)
        return converted_text, referenced_speakers

    def _build_voice_samples_array(
        self, 
        speaker_assignments: Dict[int, str], 
        referenced_speakers: List[int]
    ) -> List[str]:
        """Build ordered voice sample array for VibeVoice model matching exact speaker sequence."""
        voice_paths = []
        
        # Build voice array matching the exact speaker sequence
        # This ensures [1,2,1,2] becomes [voice1, voice2, voice1, voice2] not [voice1, voice2]
        for speaker_id in referenced_speakers:
            voice_id = speaker_assignments.get(speaker_id)
            if not voice_id:
                raise ValueError(f"No voice assigned to Speaker {speaker_id}")
            
            voice_profile = self.get_voice_profile(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")
            
            voice_paths.append(voice_profile.file_path)
        
        return voice_paths

    def generate_speech_multi_speaker(
        self,
        text: str,
        speaker_assignments: Dict[int, str],  # {speaker_id: voice_id}
        cfg_scale: float = 1.3,
    ) -> Optional[np.ndarray]:
        """
        Generate speech with multiple speakers using assignment mapping.
        
        Args:
            text: Text with [1], [2], [3], [4] markers
            speaker_assignments: Dict mapping speaker IDs to voice IDs
            cfg_scale: Generation scale parameter
            
        Returns:
            Generated audio array or None if failed
        """
        try:
            # Extract referenced speakers from text
            referenced_speakers = self._extract_referenced_speakers(text)
            if not referenced_speakers:
                raise ValueError("No speaker markers [1], [2], [3], [4] found in text")

            # Validate that all referenced speakers have assignments
            missing_assignments = [
                speaker_id for speaker_id in set(referenced_speakers)
                if speaker_id not in speaker_assignments or not speaker_assignments[speaker_id]
            ]
            if missing_assignments:
                raise ValueError(f"Missing voice assignments for speakers: {missing_assignments}")

            # Convert marker syntax to speaker format
            converted_text, _ = self._convert_marker_syntax_to_speaker_format(text)
            logger.info(f"Converted text: {converted_text}")

            # Build voice samples array
            voice_samples = self._build_voice_samples_array(speaker_assignments, referenced_speakers)
            logger.info(f"Using voice samples: {[os.path.basename(path) for path in voice_samples]}")

            # If model unavailable, return placeholder audio
            if not (self.model_loaded and self.model and self.processor):
                logger.warning("Model not loaded — returning sample placeholder audio.")
                return self._generate_sample_audio(text)

            # Prepare inputs for multi-speaker generation
            inputs = self.processor(
                text=[converted_text],  # Wrap in list for batch processing
                voice_samples=[voice_samples],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move inputs to same device as model and ensure dtype consistency
            model_device = next(self.model.parameters()).device
            model_dtype = cuda_manager.dtype
            
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model_device, non_blocking=True)
                    # Ensure dtype consistency for floating point tensors
                    if v.dtype != model_dtype and v.dtype.is_floating_point:
                        inputs[k] = inputs[k].to(model_dtype)
                        logger.debug(f"Converted multi-speaker input {k} from {v.dtype} to {model_dtype}")

            logger.info(f"Starting multi-speaker generation with {len(voice_samples)} voices...")

            # Generate audio with updated autocast API and dtype consistency
            with torch.amp.autocast('cuda', enabled=model_device.type == "cuda", dtype=model_dtype):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                )

            # Extract audio
            if (
                getattr(outputs, "speech_outputs", None)
                and outputs.speech_outputs[0] is not None
            ):
                audio_tensor = outputs.speech_outputs[0]

                # Cast to float32 so NumPy can convert (bf16/fp16 -> float32)
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)

                # Move to CPU and convert to NumPy
                audio_array = audio_tensor.detach().cpu().numpy()

                # Safety clamp
                audio_array = np.clip(audio_array, -1.0, 1.0)

                logger.info("Multi-speaker generation completed successfully")
                return audio_array

            logger.error("No speech output generated by the model.")
            return None

        except Exception as e:
            logger.error(f"Multi-speaker generation error: {e}", exc_info=True)
            # Return a nicer sample audio rather than crashing
            return self._generate_sample_audio(text)

    # Speaker Assignment Management

    def assign_voice_to_speaker(self, speaker_id: int, voice_id: str) -> bool:
        """Assign a voice to a speaker slot."""
        return self.multi_speaker_session.assign_voice(speaker_id, voice_id)

    def clear_speaker_assignment(self, speaker_id: int) -> bool:
        """Clear voice assignment for a speaker slot."""
        return self.multi_speaker_session.clear_assignment(speaker_id)

    def get_speaker_assignments(self) -> Dict[int, Dict[str, str]]:
        """Get current speaker assignments."""
        return self.multi_speaker_session.get_assignments_dict()

    def clear_all_speaker_assignments(self) -> bool:
        """Clear all speaker assignments."""
        self.multi_speaker_session.assignments.clear()
        return True

    # Advanced Batch Processing Methods (Phase 2 Optimizations)

    def _handle_memory_optimization(self, level: str, profile):
        """Handle memory optimization events."""
        if level == "critical":
            # Clear voice cache
            self.voices_cache.clear()
            logger.warning("Voice cache cleared due to memory pressure")
        elif level == "warning":
            # Clear model cache
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()

    def generate_speech_batch_optimized(
        self,
        texts: List[str],
        voice_ids: List[str],
        cfg_scale: float = 1.3,
    ) -> List[Optional[np.ndarray]]:
        """Generate speech for multiple requests using vectorized processing."""
        try:
            if not (self.model_loaded and self.model and self.processor):
                logger.warning("Model not loaded — returning sample placeholder audio.")
                return [self._generate_sample_audio(text) for text in texts]

            # Validate voices and collect paths
            voice_paths = []
            valid_requests = []
            
            for i, (text, voice_id) in enumerate(zip(texts, voice_ids)):
                voice_profile = self.voices_cache.get(voice_id)
                if voice_profile and os.path.exists(voice_profile.file_path):
                    voice_paths.append(voice_profile.file_path)
                    valid_requests.append(i)
                else:
                    logger.warning(f"Invalid voice for request {i}: {voice_id}")

            if not voice_paths:
                return [self._generate_sample_audio(text) for text in texts]

            # Load voice samples in batch using vectorized processor
            logger.info(f"Loading {len(voice_paths)} voice samples in batch...")
            voice_batch, voice_lengths = vectorized_processor.load_audio_batch(voice_paths)
            
            # Extract voice features in batch
            voice_features = vectorized_processor.extract_voice_features_batch(
                voice_batch, voice_lengths
            )
            
            # Use tensor pools for memory efficiency
            with ContextualTensorManager(tensor_pool_manager) as tm:
                # Convert to PyTorch tensors using tensor pools
                device = cuda_manager.device
                voice_tensors = []
                for i in range(len(voice_paths)):
                    # Use pre-computed features instead of raw audio
                    feature_tensor = vectorized_processor.to_torch_tensor(
                        voice_features['mel_spectrogram'][i], device
                    )
                    voice_tensors.append(feature_tensor)

                # Batch process texts
                valid_texts = [texts[i] for i in valid_requests]
                
                # Use processor with enhanced batch processing
                inputs = self.processor(
                    text=valid_texts,
                    voice_samples=[[path] for path in voice_paths],  # Fallback for now
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move to device and ensure dtype consistency
                model_dtype = cuda_manager.dtype
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device, non_blocking=True)
                        if v.dtype.is_floating_point and v.dtype != model_dtype:
                            inputs[k] = inputs[k].to(model_dtype)

                # Generate with enhanced voice conditioning
                with cuda_manager.get_compatible_autocast_context(device):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={
                            "do_sample": True,
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "use_cache": True,
                        },
                        verbose=False,
                    )

                # Extract audio outputs
                results = [None] * len(texts)
                
                for i, request_idx in enumerate(valid_requests):
                    if (hasattr(outputs, 'speech_outputs') and 
                        i < len(outputs.speech_outputs) and 
                        outputs.speech_outputs[i] is not None):
                        
                        audio_tensor = outputs.speech_outputs[i]
                        if audio_tensor.dtype != torch.float32:
                            audio_tensor = audio_tensor.to(torch.float32)
                        
                        audio_array = audio_tensor.detach().cpu().numpy()
                        results[request_idx] = np.clip(audio_array, -1.0, 1.0)
                    else:
                        results[request_idx] = self._generate_sample_audio(texts[request_idx])

                # Fill invalid requests with placeholder
                for i in range(len(texts)):
                    if results[i] is None:
                        results[i] = self._generate_sample_audio(texts[i])

                return results

        except Exception as e:
            logger.error(f"Batch generation error: {e}", exc_info=True)
            return [self._generate_sample_audio(text) for text in texts]

    def generate_speech_optimized(self, text: str, voice_id: str, cfg_scale: float = 1.3) -> Optional[np.ndarray]:
        """Optimized speech generation with memory management."""
        with ContextualTensorManager(tensor_pool_manager) as tm:
            # Use tensor pools for temporary allocations
            return self.generate_speech(text, voice_id, cfg_scale=cfg_scale)

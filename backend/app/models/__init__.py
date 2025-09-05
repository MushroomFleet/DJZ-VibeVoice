"""Models module."""

from .voice_model import (
    VoiceProfile,
    GenerationRequest,
    GenerationResponse,
    AudioRecording,
    VoiceType,
    AudioFile,
    AudioLibraryResponse,
    # Multi-Speaker Models
    MultiSpeakerGenerationRequest,
    MultiSpeakerGenerationResponse,
    SpeakerAssignmentRequest,
    SpeakerAssignmentResponse,
)

__all__ = [
    "VoiceProfile",
    "GenerationRequest",
    "GenerationResponse",
    "AudioRecording",
    "VoiceType",
    "AudioFile",
    "AudioLibraryResponse",
    # Multi-Speaker Models
    "MultiSpeakerGenerationRequest",
    "MultiSpeakerGenerationResponse",
    "SpeakerAssignmentRequest",
    "SpeakerAssignmentResponse",
]

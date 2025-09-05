"""Data models for the application."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class VoiceType(str, Enum):
    """Voice type enumeration."""

    RECORDED = "recorded"
    UPLOADED = "uploaded"
    PRESET = "preset"


class VoiceProfile(BaseModel):
    """Voice profile model."""

    id: str
    name: str
    type: VoiceType
    file_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    description: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class GenerationRequest(BaseModel):
    """TTS generation request model."""

    text: str
    voice_id: str
    num_speakers: int = Field(default=1, ge=1, le=4)
    cfg_scale: float = Field(default=1.3, ge=1.0, le=2.0)
    output_format: str = Field(default="wav")


class GenerationResponse(BaseModel):
    """TTS generation response model."""

    success: bool
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    message: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)


class AudioRecording(BaseModel):
    """Audio recording model."""

    name: str
    audio_data: str  # Base64 encoded audio
    format: str = "wav"


class AudioFile(BaseModel):
    """Generated audio file model."""

    filename: str
    voice_name: str
    duration: float
    size: int  # File size in bytes
    text_preview: str  # First 100 chars of the text
    created_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AudioLibraryResponse(BaseModel):
    """Audio library response model."""

    success: bool
    audio_files: List[AudioFile]
    total: int
    message: Optional[str] = None


# Multi-Speaker Models

class MultiSpeakerGenerationRequest(BaseModel):
    """Multi-speaker TTS generation request model."""

    text: str = Field(..., description="Text with [1], [2], [3], [4] speaker markers")
    speaker_assignments: Dict[int, str] = Field(..., description="Mapping of speaker IDs to voice IDs")
    cfg_scale: float = Field(default=1.3, ge=1.0, le=2.0, description="CFG scale for generation")
    output_format: str = Field(default="wav", description="Output audio format")


class SpeakerAssignmentResponse(BaseModel):
    """Speaker assignment response model."""

    speaker_id: int = Field(..., ge=1, le=4, description="Speaker slot ID (1-4)")
    voice_id: Optional[str] = Field(None, description="Assigned voice ID")
    voice_name: Optional[str] = Field(None, description="Assigned voice name")
    assigned: bool = Field(default=False, description="Whether this speaker has a voice assigned")


class MultiSpeakerGenerationResponse(BaseModel):
    """Multi-speaker TTS generation response model."""

    success: bool
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    referenced_speakers: List[int] = Field(default=[], description="Speaker IDs found in the text")
    message: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SpeakerAssignmentRequest(BaseModel):
    """Request to assign a voice to a speaker slot."""

    voice_id: str = Field(..., description="Voice ID to assign to the speaker")

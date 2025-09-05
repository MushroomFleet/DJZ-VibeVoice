"""API routes for the application."""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

from app.models import (
    VoiceProfile,
    GenerationRequest,
    GenerationResponse,
    AudioRecording,
    VoiceType,
    AudioLibraryResponse,
    MultiSpeakerGenerationRequest,
    MultiSpeakerGenerationResponse,
    SpeakerAssignmentRequest,
    SpeakerAssignmentResponse,
)
from app.services import VoiceService, AudioService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

voice_service = VoiceService()
audio_service = AudioService()


@router.get("/voices", response_model=List[VoiceProfile])
async def get_voices(search: Optional[str] = Query(None)):
    """Get all voice profiles with optional search."""
    voices = voice_service.get_voice_profiles()

    if search:
        search_lower = search.lower()
        voices = [v for v in voices if search_lower in v.name.lower()]

    return voices


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile."""
    try:
        success = voice_service.delete_voice_profile(voice_id)
        if success:
            return {"success": True, "message": "Voice deleted successfully"}
        else:
            raise HTTPException(404, "Voice not found")
    except Exception as e:
        logger.error(f"Failed to delete voice: {e}")
        raise HTTPException(500, f"Failed to delete voice: {str(e)}")


@router.post("/voices/upload")
async def upload_voice(file: UploadFile = File(...), name: str = Form(...)):
    try:
        logger.info(f"Uploading voice: {name}, file: {file.filename}")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                400, f"Unsupported format. Use: {settings.SUPPORTED_FORMATS}"
            )

        content = await file.read()
        size = len(content)
        max_size = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if size > max_size:
            raise HTTPException(
                400, f"File too large. Max {settings.MAX_AUDIO_SIZE_MB}MB"
            )

        # save raw
        raw_path = settings.VOICES_DIR / f"{name}_{uuid.uuid4().hex[:8]}{file_ext}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved voice file to: {raw_path}")

        # convert to wav if needed
        final_path = raw_path
        if file_ext != ".wav":
            wav_path = raw_path.with_suffix(".wav")
            audio_service.convert_to_wav(str(raw_path), str(wav_path))
            try:
                os.remove(raw_path)
            except Exception:
                pass
            final_path = wav_path
            logger.info(f"Converted to WAV: {final_path}")

        profile = voice_service.add_voice_profile(
            name=name,
            audio_path=str(final_path),
            voice_type=VoiceType.UPLOADED,
        )
        return {
            "success": True,
            "voice": profile,
            "message": "Voice uploaded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")


@router.post("/voices/record")
async def record_voice(recording: AudioRecording):
    """Save a recorded voice sample reliably (handles webm/mp4/ogg)."""
    try:
        logger.info(f"Saving recorded voice: {recording.name}")

        # Use container extension from client format; default webm
        ext = (recording.format or "webm").lower().lstrip(".")
        raw_path = (
            settings.VOICES_DIR / f"{recording.name}_{uuid.uuid4().hex[:8]}.{ext}"
        )

        # Write raw and convert to wav
        wav_path = audio_service.base64_to_audio(
            base64_data=recording.audio_data,
            output_path=raw_path,
            format=ext,
        )
        logger.info(f"Saved recording to: {wav_path}")

        profile = voice_service.add_voice_profile(
            name=recording.name,
            audio_path=wav_path,
            voice_type=VoiceType.RECORDED,
        )
        return {
            "success": True,
            "voice": profile,
            "message": "Recording saved successfully",
        }

    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise HTTPException(500, f"Recording failed: {str(e)}")


@router.post("/generate", response_model=GenerationResponse)
async def generate_speech(request: GenerationRequest):
    try:
        logger.info(f"Generating speech for text length: {len(request.text)}")

        # Get voice profile to use its name
        voice_profile = voice_service.get_voice_profile(request.voice_id)
        voice_name = voice_profile.name if voice_profile else "unknown"

        audio_array = voice_service.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            num_speakers=request.num_speakers,
            cfg_scale=request.cfg_scale,
        )

        if audio_array is None:
            return GenerationResponse(
                success=False,
                message="Failed to generate audio. Please check if model is loaded.",
            )

        # Create filename with voice name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{voice_name}_{timestamp}.wav"
        filepath = audio_service.save_audio(audio_array, filename=filename)
        logger.info(f"Saved generated audio to: {filepath}")

        duration = len(audio_array) / settings.SAMPLE_RATE

        # Save metadata for audio library
        audio_service.save_audio_metadata(
            filename, voice_name, duration, request.text[:100]
        )

        return GenerationResponse(
            success=True,
            audio_url=f"/api/audio/{filename}",
            duration=duration,
            message="Audio generated successfully",
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerationResponse(success=False, message=f"Generation failed: {str(e)}")


@router.post("/generate/file")
async def generate_from_file(
    file: UploadFile = File(...),
    voice_id: str = Form(...),
    cfg_scale: float = Form(1.3),
):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        logger.info(f"Generating from file: {file.filename}, text length: {len(text)}")
        req = GenerationRequest(text=text, voice_id=voice_id, cfg_scale=cfg_scale)
        return await generate_speech(req)
    except Exception as e:
        logger.error(f"File generation error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@router.get("/audio/library", response_model=AudioLibraryResponse)
async def get_audio_library(search: Optional[str] = Query(None)):
    """Get all generated audio files with metadata."""
    try:
        audio_files = audio_service.get_audio_library(search)
        return AudioLibraryResponse(
            success=True, audio_files=audio_files, total=len(audio_files)
        )
    except Exception as e:
        logger.error(f"Failed to get audio library: {e}")
        return AudioLibraryResponse(
            success=False, audio_files=[], total=0, message=str(e)
        )


@router.delete("/audio/{filename}")
async def delete_audio(filename: str):
    """Delete a generated audio file."""
    try:
        filepath = settings.OUTPUTS_DIR / filename
        if not filepath.exists():
            raise HTTPException(404, "Audio file not found")

        os.remove(filepath)

        # Also remove metadata file if exists
        metadata_file = filepath.with_suffix(".json")
        if metadata_file.exists():
            os.remove(metadata_file)

        return {"success": True, "message": "Audio deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete audio: {e}")
        raise HTTPException(500, f"Failed to delete audio: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = settings.OUTPUTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(
        filepath,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": voice_service.is_model_loaded(),
    }


# Multi-Speaker Endpoints

@router.post("/voices/assign-speaker/{speaker_id}")
async def assign_voice_to_speaker(
    speaker_id: int, 
    request: SpeakerAssignmentRequest
):
    """Assign a voice to a specific speaker slot (1-4)."""
    try:
        if not (1 <= speaker_id <= 4):
            raise HTTPException(400, "Speaker ID must be between 1 and 4")
        
        success = voice_service.assign_voice_to_speaker(speaker_id, request.voice_id)
        
        if success:
            return {
                "success": True, 
                "message": f"Voice assigned to Speaker {speaker_id}",
                "speaker_id": speaker_id,
                "voice_id": request.voice_id
            }
        else:
            raise HTTPException(400, "Failed to assign voice to speaker")
            
    except ValueError as e:
        logger.error(f"Speaker assignment error: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Unexpected error in speaker assignment: {e}")
        raise HTTPException(500, f"Failed to assign voice: {str(e)}")


@router.get("/voices/speaker-assignments", response_model=List[SpeakerAssignmentResponse])
async def get_speaker_assignments():
    """Get current voice-to-speaker assignments."""
    try:
        assignments = voice_service.get_speaker_assignments()
        
        # Create response for all 4 speaker slots
        response = []
        for speaker_id in range(1, 5):
            if speaker_id in assignments:
                assignment = assignments[speaker_id]
                response.append(SpeakerAssignmentResponse(
                    speaker_id=speaker_id,
                    voice_id=assignment['voice_id'],
                    voice_name=assignment['voice_name'],
                    assigned=True
                ))
            else:
                response.append(SpeakerAssignmentResponse(
                    speaker_id=speaker_id,
                    voice_id=None,
                    voice_name=None,
                    assigned=False
                ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get speaker assignments: {e}")
        raise HTTPException(500, f"Failed to get assignments: {str(e)}")


@router.delete("/voices/speaker-assignments/{speaker_id}")
async def clear_speaker_assignment(speaker_id: int):
    """Clear voice assignment for specific speaker slot."""
    try:
        if not (1 <= speaker_id <= 4):
            raise HTTPException(400, "Speaker ID must be between 1 and 4")
        
        success = voice_service.clear_speaker_assignment(speaker_id)
        
        if success:
            return {
                "success": True, 
                "message": f"Speaker {speaker_id} assignment cleared",
                "speaker_id": speaker_id
            }
        else:
            return {
                "success": False,
                "message": f"No assignment found for Speaker {speaker_id}",
                "speaker_id": speaker_id
            }
            
    except Exception as e:
        logger.error(f"Failed to clear speaker assignment: {e}")
        raise HTTPException(500, f"Failed to clear assignment: {str(e)}")


@router.delete("/voices/speaker-assignments")
async def clear_all_speaker_assignments():
    """Clear all voice assignments."""
    try:
        success = voice_service.clear_all_speaker_assignments()
        
        if success:
            return {
                "success": True,
                "message": "All speaker assignments cleared"
            }
        else:
            raise HTTPException(500, "Failed to clear assignments")
            
    except Exception as e:
        logger.error(f"Failed to clear all speaker assignments: {e}")
        raise HTTPException(500, f"Failed to clear assignments: {str(e)}")


@router.post("/voices/generate-multi-speaker", response_model=MultiSpeakerGenerationResponse)
async def generate_multi_speaker_speech(request: MultiSpeakerGenerationRequest):
    """Generate speech with multi-speaker support."""
    try:
        logger.info(f"Generating multi-speaker speech for text length: {len(request.text)}")
        logger.info(f"Speaker assignments: {request.speaker_assignments}")

        # Extract referenced speakers from text for validation
        referenced_speakers = voice_service._extract_referenced_speakers(request.text)
        logger.info(f"Referenced speakers in text: {referenced_speakers}")

        if not referenced_speakers:
            return MultiSpeakerGenerationResponse(
                success=False,
                message="No speaker markers [1], [2], [3], [4] found in text",
                referenced_speakers=[]
            )

        # Generate the multi-speaker audio
        audio_array = voice_service.generate_speech_multi_speaker(
            text=request.text,
            speaker_assignments=request.speaker_assignments,
            cfg_scale=request.cfg_scale,
        )

        if audio_array is None:
            return MultiSpeakerGenerationResponse(
                success=False,
                message="Failed to generate multi-speaker audio. Please check if model is loaded.",
                referenced_speakers=referenced_speakers
            )

        # Create filename with multi-speaker indicator and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_speakers = list(set(referenced_speakers))
        speaker_names = []
        
        for speaker_id in sorted(unique_speakers):
            voice_id = request.speaker_assignments.get(speaker_id)
            if voice_id:
                voice_profile = voice_service.get_voice_profile(voice_id)
                if voice_profile:
                    speaker_names.append(f"S{speaker_id}-{voice_profile.name}")
        
        filename_prefix = "_".join(speaker_names) if speaker_names else "multispeaker"
        filename = f"{filename_prefix}_{timestamp}.wav"
        
        filepath = audio_service.save_audio(audio_array, filename=filename)
        logger.info(f"Saved multi-speaker audio to: {filepath}")

        duration = len(audio_array) / settings.SAMPLE_RATE

        # Save metadata for audio library
        text_preview = request.text[:100]
        voice_name = f"Multi-Speaker ({len(unique_speakers)} voices)"
        audio_service.save_audio_metadata(
            filename, voice_name, duration, text_preview
        )

        return MultiSpeakerGenerationResponse(
            success=True,
            audio_url=f"/api/audio/{filename}",
            duration=duration,
            referenced_speakers=referenced_speakers,
            message="Multi-speaker audio generated successfully",
        )
        
    except ValueError as e:
        logger.error(f"Multi-speaker generation validation error: {e}")
        return MultiSpeakerGenerationResponse(
            success=False,
            message=str(e),
            referenced_speakers=voice_service._extract_referenced_speakers(request.text)
        )
    except Exception as e:
        logger.error(f"Multi-speaker generation error: {e}")
        return MultiSpeakerGenerationResponse(
            success=False, 
            message=f"Multi-speaker generation failed: {str(e)}",
            referenced_speakers=voice_service._extract_referenced_speakers(request.text)
        )

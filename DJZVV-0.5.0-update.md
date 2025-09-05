# DJZ-VibeVoice v0.5.0 - Multi-Speaker Implementation Update

## Overview

Version 0.5.0 introduces comprehensive multi-speaker support to DJZ-VibeVoice, enabling users to assign up to 4 different voice samples and control them using text prompt markers `[1]`, `[2]`, `[3]`, `[4]` within their input text.

### Key Features
- **Multi-Speaker Voice Assignment**: UI system for binding voices to speaker IDs 1-4
- **Text Prompt Markers**: Support for `[1]`, `[2]`, `[3]`, `[4]` syntax in text input
- **Backward Compatibility**: Single-speaker mode remains unchanged
- **Enhanced UI**: Intuitive voice assignment interface with visual feedback
- **Flexible Generation**: Support for mixed-speaker content within single generation

## Technical Architecture

### Core Implementation Strategy

The multi-speaker functionality leverages the native VibeVoice model capability to process multiple voice samples simultaneously. The implementation focuses on:

1. **Voice Sample Array Management**: Converting single voice selection to voice sample arrays
2. **Text Format Conversion**: Transforming `[1], [2], [3], [4]` markers to "Speaker 1:", "Speaker 2:" format
3. **State Management**: Tracking voice-to-speaker assignments in application state
4. **UI Enhancement**: Adding speaker assignment controls and validation

### Data Flow Architecture

```
User Input Text with [1], [2] markers
    ↓
Text Processing Engine (Frontend)
    ↓
Speaker Assignment Validation
    ↓
API Call with Voice Mapping
    ↓
Backend Text Conversion ([1] → "Speaker 1:")
    ↓
Voice Sample Array Construction
    ↓
VibeVoice Model Generation
    ↓
Audio Output
```

## Detailed Implementation Plan

### 1. Backend Changes

#### 1.1 Voice Service Enhancement (`backend/app/services/voice_service.py`)

**New Classes:**
```python
class SpeakerAssignment:
    speaker_id: int  # 1-4
    voice_id: str
    voice_profile: VoiceProfile

class MultiSpeakerSession:
    assignments: Dict[int, SpeakerAssignment]  # {1: assignment, 2: assignment, ...}
    max_speakers: int = 4
    
    def assign_voice(self, speaker_id: int, voice_id: str) -> bool
    def clear_assignment(self, speaker_id: int) -> bool
    def get_voice_array(self, referenced_speakers: List[int]) -> List[str]
    def validate_assignments(self, referenced_speakers: List[int]) -> bool
```

**Enhanced VoiceService Methods:**
```python
def generate_speech_multi_speaker(
    self,
    text: str,
    speaker_assignments: Dict[int, str],  # {speaker_id: voice_id}
    cfg_scale: float = 1.3,
) -> Optional[np.ndarray]:
    """Generate speech with multiple speakers using assignment mapping."""
    
def _convert_marker_syntax_to_speaker_format(self, text: str) -> Tuple[str, List[int]]:
    """Convert [1], [2] markers to Speaker 1:, Speaker 2: format.
    Returns: (converted_text, referenced_speaker_ids)
    """
    
def _extract_referenced_speakers(self, text: str) -> List[int]:
    """Extract speaker IDs referenced in text via [1], [2], [3], [4] markers."""
    
def _build_voice_samples_array(
    self, 
    speaker_assignments: Dict[int, str], 
    referenced_speakers: List[int]
) -> List[str]:
    """Build ordered voice sample array for VibeVoice model."""
```

**Text Processing Logic:**
```python
def _convert_marker_syntax_to_speaker_format(self, text: str) -> Tuple[str, List[int]]:
    """
    Convert: "Hello [1], how are you [2]? I'm fine [1]."
    To: "Speaker 1: Hello Speaker 1:, how are you Speaker 2:? I'm fine Speaker 1:."
    
    Returns: (converted_text, [1, 2, 1])  # referenced speakers in order
    """
    import re
    
    # Find all [1], [2], [3], [4] markers
    pattern = r'\[([1-4])\]'
    referenced_speakers = []
    
    def replace_marker(match):
        speaker_id = int(match.group(1))
        referenced_speakers.append(speaker_id)
        return f"Speaker {speaker_id}:"
    
    converted_text = re.sub(pattern, replace_marker, text)
    return converted_text, referenced_speakers
```

#### 1.2 API Routes Enhancement (`backend/app/api/routes.py`)

**New Endpoints:**
```python
@router.post("/voices/assign-speaker/{speaker_id}")
async def assign_voice_to_speaker(
    speaker_id: int,
    voice_id: str,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """Assign a voice to a specific speaker slot (1-4)."""

@router.get("/voices/speaker-assignments")
async def get_speaker_assignments(
    voice_service: VoiceService = Depends(get_voice_service)
):
    """Get current voice-to-speaker assignments."""

@router.delete("/voices/speaker-assignments/{speaker_id}")
async def clear_speaker_assignment(
    speaker_id: int,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """Clear voice assignment for specific speaker slot."""

@router.post("/voices/generate-multi-speaker")
async def generate_multi_speaker_speech(
    request: MultiSpeakerGenerationRequest,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """Generate speech with multi-speaker support."""
```

**Request/Response Models:**
```python
class MultiSpeakerGenerationRequest(BaseModel):
    text: str
    speaker_assignments: Dict[int, str]  # {speaker_id: voice_id}
    cfg_scale: float = 1.3

class SpeakerAssignmentResponse(BaseModel):
    speaker_id: int
    voice_id: Optional[str]
    voice_name: Optional[str]
    assigned: bool

class MultiSpeakerGenerationResponse(BaseModel):
    success: bool
    audio_url: Optional[str]
    duration: Optional[float]
    referenced_speakers: List[int]
    message: Optional[str]
```

### 2. Frontend Changes

#### 2.1 State Management (`frontend/src/contexts/AppContext.jsx`)

**Enhanced State Structure:**
```javascript
const initialState = {
  // ... existing state
  speakerAssignments: {
    1: null,  // voice_id or null
    2: null,
    3: null,
    4: null
  },
  multiSpeakerMode: false,
  generationSettings: {
    cfgScale: 1.3,
    numSpeakers: 1,
    outputFormat: 'wav',
    enableMultiSpeaker: false  // NEW
  }
};
```

**New Actions:**
```javascript
case 'SET_SPEAKER_ASSIGNMENT':
  return {
    ...state,
    speakerAssignments: {
      ...state.speakerAssignments,
      [action.payload.speakerId]: action.payload.voiceId
    }
  };

case 'CLEAR_SPEAKER_ASSIGNMENT':
  return {
    ...state,
    speakerAssignments: {
      ...state.speakerAssignments,
      [action.payload.speakerId]: null
    }
  };

case 'TOGGLE_MULTI_SPEAKER_MODE':
  return {
    ...state,
    multiSpeakerMode: action.payload
  };

case 'CLEAR_ALL_SPEAKER_ASSIGNMENTS':
  return {
    ...state,
    speakerAssignments: { 1: null, 2: null, 3: null, 4: null }
  };
```

#### 2.2 New VoiceAssignment Component

**File: `frontend/src/components/voice/VoiceAssignment/VoiceAssignment.jsx`**
```javascript
import React from 'react';
import { useApp } from '../../../contexts/AppContext';
import Button from '../../common/Button/Button';
import styles from './VoiceAssignment.module.css';

const VoiceAssignment = () => {
  const { state, dispatch } = useApp();
  const { voices, speakerAssignments, selectedVoiceId } = state;

  const assignVoiceToSpeaker = (speakerId) => {
    if (!selectedVoiceId) {
      toast.error('Please select a voice first');
      return;
    }

    dispatch({
      type: 'SET_SPEAKER_ASSIGNMENT',
      payload: { speakerId, voiceId: selectedVoiceId }
    });
    
    toast.success(`Voice assigned to Speaker ${speakerId}`);
  };

  const clearSpeakerAssignment = (speakerId) => {
    dispatch({
      type: 'CLEAR_SPEAKER_ASSIGNMENT',
      payload: { speakerId }
    });
  };

  const getVoiceName = (voiceId) => {
    const voice = voices.find(v => v.id === voiceId);
    return voice ? voice.name : 'Unassigned';
  };

  return (
    <div className={styles.voiceAssignment}>
      <div className={styles.header}>
        <h3>Speaker Voice Assignments</h3>
        <p>Assign voices to speaker slots for multi-speaker generation</p>
      </div>
      
      <div className={styles.speakerGrid}>
        {[1, 2, 3, 4].map(speakerId => (
          <div 
            key={speakerId} 
            className={`${styles.speakerSlot} ${
              speakerAssignments[speakerId] ? styles.assigned : styles.unassigned
            }`}
          >
            <div className={styles.speakerHeader}>
              <span className={styles.speakerId}>Speaker {speakerId}</span>
              <span className={styles.markerSyntax}>[{speakerId}]</span>
            </div>
            
            <div className={styles.voiceInfo}>
              <span className={styles.voiceName}>
                {getVoiceName(speakerAssignments[speakerId])}
              </span>
            </div>
            
            <div className={styles.actions}>
              <Button
                onClick={() => assignVoiceToSpeaker(speakerId)}
                variant="primary"
                size="small"
                disabled={!selectedVoiceId}
                className={styles.assignButton}
              >
                Use {speakerId}
              </Button>
              
              {speakerAssignments[speakerId] && (
                <Button
                  onClick={() => clearSpeakerAssignment(speakerId)}
                  variant="secondary"
                  size="small"
                  className={styles.clearButton}
                >
                  Clear
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <div className={styles.footer}>
        <Button
          onClick={() => dispatch({ type: 'CLEAR_ALL_SPEAKER_ASSIGNMENTS' })}
          variant="outline"
          size="small"
          className={styles.clearAllButton}
        >
          Clear All Assignments
        </Button>
      </div>
    </div>
  );
};

export default VoiceAssignment;
```

**CSS File: `frontend/src/components/voice/VoiceAssignment/VoiceAssignment.module.css`**
```css
.voiceAssignment {
  padding: 1rem;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--background-secondary);
}

.header {
  margin-bottom: 1.5rem;
  text-align: center;
}

.header h3 {
  margin: 0 0 0.5rem 0;
  color: var(--text-primary);
}

.header p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.speakerGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.speakerSlot {
  padding: 1rem;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.speakerSlot.assigned {
  background: var(--success-light);
  border: 2px solid var(--success-color);
}

.speakerSlot.unassigned {
  background: var(--background-primary);
  border: 2px dashed var(--border-color);
}

.speakerHeader {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.speakerId {
  font-weight: 600;
  color: var(--text-primary);
}

.markerSyntax {
  background: var(--primary-color);
  color: white;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.8rem;
}

.voiceInfo {
  margin-bottom: 1rem;
  min-height: 1.5rem;
}

.voiceName {
  color: var(--text-secondary);
  font-style: italic;
}

.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.assignButton {
  flex: 1;
  min-width: 80px;
}

.clearButton {
  flex: 0 0 auto;
}

.footer {
  text-align: center;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
}

.clearAllButton {
  min-width: 120px;
}
```

#### 2.3 Enhanced TextInput Component

**Updates to `frontend/src/components/text/TextInput/TextInput.jsx`:**
```javascript
// Add multi-speaker syntax helper
const MultiSpeakerHelper = ({ show }) => {
  if (!show) return null;
  
  return (
    <div className={styles.syntaxHelper}>
      <h4>Multi-Speaker Syntax</h4>
      <p>Use <code>[1]</code>, <code>[2]</code>, <code>[3]</code>, <code>[4]</code> to control which speaker says what:</p>
      <div className={styles.example}>
        <strong>Example:</strong><br/>
        <code>Hello [1], how are you today [2]? I'm doing great [1]!</code>
      </div>
      <p className={styles.note}>
        Make sure to assign voices to speaker slots before generating!
      </p>
    </div>
  );
};

// Add syntax highlighting for markers
const highlightSpeakerMarkers = (text) => {
  return text.replace(/\[([1-4])\]/g, '<span class="speaker-marker">[$1]</span>');
};
```

#### 2.4 Enhanced MainPage Component

**Updates to `frontend/src/components/pages/MainPage/MainPage.jsx`:**
```javascript
import VoiceAssignment from '../../voice/VoiceAssignment/VoiceAssignment';

// Add multi-speaker mode toggle
const MultiSpeakerToggle = () => {
  const { state, dispatch } = useApp();
  
  return (
    <div className={styles.modeToggle}>
      <label className={styles.toggleLabel}>
        <input
          type="checkbox"
          checked={state.multiSpeakerMode}
          onChange={(e) => dispatch({ 
            type: 'TOGGLE_MULTI_SPEAKER_MODE', 
            payload: e.target.checked 
          })}
        />
        <span>Enable Multi-Speaker Mode</span>
      </label>
    </div>
  );
};

// Enhanced generation logic
const handleGenerate = async () => {
  // ... existing validation
  
  try {
    setIsGenerating(true);
    dispatch({ type: 'SET_LOADING', payload: { loading: true, text: 'Generating speech...' } });

    let result;
    if (state.multiSpeakerMode) {
      // Validate speaker assignments
      const referencedSpeakers = extractReferencedSpeakers(textInput);
      const missingAssignments = referencedSpeakers.filter(
        speakerId => !state.speakerAssignments[speakerId]
      );
      
      if (missingAssignments.length > 0) {
        throw new Error(`Missing voice assignments for speakers: ${missingAssignments.join(', ')}`);
      }
      
      result = await voiceService.generateMultiSpeakerSpeech({
        text: textInput,
        speaker_assignments: state.speakerAssignments,
        cfg_scale: generationSettings.cfgScale
      });
    } else {
      // Single speaker mode (existing logic)
      result = await voiceService.generateSpeech({
        text: textInput,
        voice_id: selectedVoiceId,
        cfg_scale: generationSettings.cfgScale
      });
    }

    // ... handle result
  } catch (error) {
    // ... error handling
  }
};

// Helper function
const extractReferencedSpeakers = (text) => {
  const matches = text.match(/\[([1-4])\]/g);
  if (!matches) return [];
  return [...new Set(matches.map(match => parseInt(match.slice(1, -1))))];
};
```

### 3. API Service Updates

#### 3.1 Voice Service (`frontend/src/services/voiceService.js`)

**New Methods:**
```javascript
export const voiceService = {
  // ... existing methods
  
  async assignVoiceToSpeaker(speakerId, voiceId) {
    const response = await api.post(`/voices/assign-speaker/${speakerId}`, { voice_id: voiceId });
    return response.data;
  },
  
  async getSpeakerAssignments() {
    const response = await api.get('/voices/speaker-assignments');
    return response.data;
  },
  
  async clearSpeakerAssignment(speakerId) {
    const response = await api.delete(`/voices/speaker-assignments/${speakerId}`);
    return response.data;
  },
  
  async generateMultiSpeakerSpeech({ text, speaker_assignments, cfg_scale = 1.3 }) {
    try {
      const response = await api.post('/voices/generate-multi-speaker', {
        text,
        speaker_assignments,
        cfg_scale
      });
      
      if (response.data.success) {
        return {
          success: true,
          audio_url: response.data.audio_url,
          duration: response.data.duration,
          referenced_speakers: response.data.referenced_speakers
        };
      } else {
        throw new Error(response.data.message || 'Generation failed');
      }
    } catch (error) {
      console.error('Multi-speaker generation error:', error);
      throw error;
    }
  }
};
```

## Implementation Timeline

### Phase 1: Backend Foundation (Days 1-2)
1. **Voice Service Enhancement**
   - Implement `MultiSpeakerSession` class
   - Add text conversion methods (`_convert_marker_syntax_to_speaker_format`)
   - Update `generate_speech` to support multi-speaker mode
   
2. **API Routes**
   - Create speaker assignment endpoints
   - Add multi-speaker generation endpoint
   - Test basic multi-speaker generation

### Phase 2: Frontend State Management (Days 3-4)
1. **Context Enhancement**
   - Add speaker assignment state
   - Implement assignment actions
   - Add multi-speaker mode toggle

2. **Service Layer**
   - Implement API integration methods
   - Add error handling for speaker validation

### Phase 3: UI Components (Days 5-6)
1. **VoiceAssignment Component**
   - Build speaker slot interface
   - Implement voice assignment logic
   - Add visual feedback and validation

2. **TextInput Enhancement**
   - Add syntax highlighting for `[1], [2], [3], [4]` markers
   - Include helper text and examples
   - Implement speaker validation

### Phase 4: Integration & Testing (Days 7-8)
1. **MainPage Integration**
   - Add multi-speaker mode toggle
   - Integrate VoiceAssignment component
   - Update generation flow

2. **Testing & Refinement**
   - Test multi-speaker generation end-to-end
   - Validate error handling
   - Ensure backward compatibility

## Testing Strategy

### Unit Tests
- Text conversion functions (`[1] → Speaker 1:`)
- Speaker assignment validation
- Voice array construction
- API endpoint responses

### Integration Tests
- Multi-speaker generation workflow
- Voice assignment persistence
- Error handling for missing assignments
- Single-speaker backward compatibility

### User Acceptance Tests
- Voice assignment UI usability
- Text input with speaker markers
- Generation with mixed speakers
- Error messages and validation feedback

## Migration & Backward Compatibility

### Existing User Data
- **No breaking changes** to existing voice profiles
- **Single-speaker mode** remains default
- **Existing generations** continue to work unchanged

### Configuration Updates
- New environment variables for multi-speaker limits
- Optional feature flags for gradual rollout
- Database migration scripts (if needed for persistence)

### User Migration Path
1. **Seamless transition**: Users can continue using single-speaker mode
2. **Opt-in multi-speaker**: Toggle to enable new functionality
3. **Guided introduction**: Help text and examples for new syntax

## Future Enhancements (Post v0.5.0)

### Potential Improvements
1. **Voice Assignment Presets**: Save common speaker configurations
2. **Advanced Text Processing**: Support for speaker name syntax ("Alice: Hello there")
3. **Real-time Preview**: Audio preview for individual speakers
4. **Batch Generation**: Multiple text inputs with same speaker assignments
5. **Speaker Mixing Controls**: Individual volume/effects per speaker

### Performance Optimizations
1. **Voice Caching**: Cache loaded voice samples for faster generation
2. **Parallel Processing**: Concurrent voice processing where possible
3. **Smart Model Loading**: Load only required voice models

## Conclusion

Version 0.5.0 represents a significant enhancement to DJZ-VibeVoice, adding professional multi-speaker capabilities while maintaining the simplicity and reliability of the existing system. The implementation provides a solid foundation for future audio generation features while ensuring a smooth transition for existing users.

The modular design allows for incremental rollout and easy maintenance, with clear separation between single-speaker and multi-speaker functionality. The UI design prioritizes user experience with intuitive voice assignment and clear visual feedback.

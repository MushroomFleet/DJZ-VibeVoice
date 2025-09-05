import React, { useState } from 'react';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import VoiceSelector from '../../voice/VoiceSelector/VoiceSelector';
import VoiceAssignment from '../../voice/VoiceAssignment/VoiceAssignment';
import TextInput from '../../text/TextInput/TextInput';
import GenerationSettings from '../../text/GenerationSettings/GenerationSettings';
import Button from '../../common/Button/Button';
import toast from 'react-hot-toast';
import styles from './MainPage.module.css';

const MultiSpeakerToggle = () => {
  const { state, dispatch } = useApp();
  
  const handleToggle = (enabled) => {
    dispatch({ 
      type: 'TOGGLE_MULTI_SPEAKER_MODE', 
      payload: enabled 
    });
    
    if (enabled) {
      toast.success('Multi-speaker mode enabled');
    } else {
      toast.success('Single-speaker mode enabled');
    }
  };
  
  return (
    <div className={styles.modeToggle}>
      <div className={styles.toggleHeader}>
        <h3>
          <i className={state.multiSpeakerMode ? "fas fa-users" : "fas fa-user"} />
          Generation Mode
        </h3>
        <p>Choose between single-speaker or multi-speaker generation</p>
      </div>
      
      <div className={styles.toggleButtons}>
        <Button
          variant={!state.multiSpeakerMode ? 'primary' : 'secondary'}
          onClick={() => handleToggle(false)}
          className={styles.modeButton}
        >
          <i className="fas fa-user" />
          Single Speaker
        </Button>
        <Button
          variant={state.multiSpeakerMode ? 'primary' : 'secondary'}
          onClick={() => handleToggle(true)}
          className={styles.modeButton}
        >
          <i className="fas fa-users" />
          Multi-Speaker
        </Button>
      </div>
      
      {state.multiSpeakerMode && (
        <div className={styles.modeDescription}>
          <i className="fas fa-info-circle" />
          Use [1], [2], [3], [4] in your text to control different speakers
        </div>
      )}
    </div>
  );
};

const MainPage = () => {
  const { state, dispatch } = useApp();
  const { 
    selectedVoiceId, 
    textInput, 
    generationSettings, 
    generatedAudio, 
    loading,
    multiSpeakerMode,
    speakerAssignments
  } = state;

  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    if (!textInput.trim()) {
      toast.error('Please enter some text');
      return;
    }

    if (isGenerating) {
      toast.error('Generation already in progress. Please wait...');
      return;
    }

    try {
      setIsGenerating(true);
      dispatch({ 
        type: 'SET_LOADING', 
        payload: { 
          loading: true, 
          text: multiSpeakerMode 
            ? 'Generating multi-speaker speech... This may take 2-3 minutes on CPU.' 
            : 'Generating speech... This may take 1-2 minutes on CPU.' 
        } 
      });

      let result;
      
      if (multiSpeakerMode) {
        // Multi-speaker validation and generation
        const validation = voiceService.validateSpeakerAssignments(textInput, speakerAssignments);
        
        if (!validation.isValid) {
          throw new Error(`Missing voice assignments for speakers: ${validation.missingAssignments.join(', ')}`);
        }
        
        if (validation.referencedSpeakers.length === 0) {
          throw new Error('No speaker markers [1], [2], [3], [4] found in text. Please add speaker markers for multi-speaker mode.');
        }
        
        // Convert speakerAssignments object to the format expected by the API
        const assignmentsForAPI = {};
        validation.referencedSpeakers.forEach(speakerId => {
          if (speakerAssignments[speakerId]) {
            assignmentsForAPI[speakerId] = speakerAssignments[speakerId];
          }
        });
        
        result = await voiceService.generateMultiSpeakerSpeech({
          text: textInput,
          speaker_assignments: assignmentsForAPI,
          cfg_scale: generationSettings.cfgScale
        });
      } else {
        // Single speaker mode validation and generation
        if (!selectedVoiceId) {
          throw new Error('Please select a voice');
        }
        
        result = await voiceService.generateSpeech({
          text: textInput,
          voice_id: selectedVoiceId,
          num_speakers: generationSettings.numSpeakers,
          cfg_scale: generationSettings.cfgScale
        });
      }

      if (result.success) {
        dispatch({ type: 'SET_GENERATED_AUDIO', payload: result });
        
        if (multiSpeakerMode) {
          const speakerCount = result.referenced_speakers ? result.referenced_speakers.length : 'multiple';
          toast.success(`Multi-speaker audio generated successfully with ${speakerCount} speakers`);
        } else {
          toast.success('Speech generated successfully');
        }
      } else {
        toast.error(result.message || 'Generation failed');
      }
    } catch (error) {
      console.error('Generation error:', error);
      toast.error(error.message || 'Generation error');
    } finally {
      setIsGenerating(false);
      dispatch({ type: 'SET_LOADING', payload: { loading: false } });
    }
  };

  // Helper function to check if generation is ready
  const isGenerationReady = () => {
    if (!textInput.trim()) return false;
    
    if (multiSpeakerMode) {
      const validation = voiceService.validateSpeakerAssignments(textInput, speakerAssignments);
      return validation.isValid && validation.referencedSpeakers.length > 0;
    } else {
      return selectedVoiceId !== null;
    }
  };

  const getGenerationButtonText = () => {
    if (isGenerating) {
      return multiSpeakerMode ? 'Generating Multi-Speaker...' : 'Generating...';
    }
    return multiSpeakerMode ? 'Generate Multi-Speaker Audio' : 'Generate Speech';
  };

  return (
    <div className={styles.mainPage}>
      {/* Generation Mode Toggle */}
      <section className="card section">
        <MultiSpeakerToggle />
      </section>

      {/* Voice Selection Section */}
      <section className="card section">
        <h2 className="section-title">
          <i className={multiSpeakerMode ? "fas fa-users" : "fas fa-user-circle"} />
          {multiSpeakerMode ? 'Voice Management' : 'Voice Selection'}
        </h2>
        
        {!multiSpeakerMode && <VoiceSelector />}
        
        {multiSpeakerMode && (
          <>
            <VoiceSelector />
            <VoiceAssignment />
          </>
        )}
      </section>

      {/* Text Input Section */}
      <section className="card section">
        <h2 className="section-title">
          <i className="fas fa-keyboard" />
          Text Input
        </h2>
        <TextInput />
        
        <div className={styles.settingsSection}>
          <GenerationSettings />
        </div>
        
        <Button
          onClick={handleGenerate}
          variant="primary"
          size="large"
          disabled={loading || isGenerating || !isGenerationReady()}
          className={styles.generateButton}
          title={
            !isGenerationReady() 
              ? (multiSpeakerMode 
                  ? 'Please assign voices to all referenced speakers' 
                  : 'Please select a voice and enter text'
                )
              : ''
          }
        >
          <i className={isGenerating ? "fas fa-spinner fa-spin" : (multiSpeakerMode ? "fas fa-users" : "fas fa-magic")} />
          {getGenerationButtonText()}
        </Button>
      </section>

      {/* Output Section */}
      {generatedAudio && (
        <section className="card section">
          <h2 className="section-title">
            <i className="fas fa-volume-up" />
            Generated Audio
            {generatedAudio.referenced_speakers && generatedAudio.referenced_speakers.length > 1 && (
              <span className={styles.speakerBadge}>
                <i className="fas fa-users" />
                {generatedAudio.referenced_speakers.length} speakers
              </span>
            )}
          </h2>
          <div className={styles.audioOutput}>
            <audio 
              controls 
              src={generatedAudio.audio_url}
              className={styles.audioPlayer}
            >
              Your browser does not support the audio element.
            </audio>
            <div className={styles.audioInfo}>
              <span>Duration: {generatedAudio.duration?.toFixed(1)}s</span>
              {generatedAudio.referenced_speakers && (
                <span>Speakers: {generatedAudio.referenced_speakers.join(', ')}</span>
              )}
              <div className={styles.audioActions}>
                <Button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = generatedAudio.audio_url;
                    link.download = `generated_${multiSpeakerMode ? 'multispeaker_' : ''}${Date.now()}.wav`;
                    link.click();
                  }}
                  variant="secondary"
                  size="small"
                >
                  <i className="fas fa-download" />
                  Download
                </Button>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default MainPage;

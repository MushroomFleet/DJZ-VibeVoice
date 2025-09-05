import React, { useState } from 'react';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import VoiceSelector from '../../voice/VoiceSelector/VoiceSelector';
import TextInput from '../../text/TextInput/TextInput';
import GenerationSettings from '../../text/GenerationSettings/GenerationSettings';
import Button from '../../common/Button/Button';
import toast from 'react-hot-toast';
import styles from './MainPage.module.css';

const MainPage = () => {
  const { state, dispatch } = useApp();
  const { 
    selectedVoiceId, 
    textInput, 
    generationSettings, 
    generatedAudio, 
    loading 
  } = state;

  const handleGenerate = async () => {
    if (!selectedVoiceId) {
      toast.error('Please select a voice');
      return;
    }

    if (!textInput.trim()) {
      toast.error('Please enter some text');
      return;
    }

    try {
      dispatch({ 
        type: 'SET_LOADING', 
        payload: { loading: true, text: 'Generating speech...' } 
      });

      const result = await voiceService.generateSpeech({
        text: textInput,
        voice_id: selectedVoiceId,
        num_speakers: generationSettings.numSpeakers,
        cfg_scale: generationSettings.cfgScale
      });

      if (result.success) {
        dispatch({ type: 'SET_GENERATED_AUDIO', payload: result });
        toast.success('Speech generated successfully');
      } else {
        toast.error(result.message || 'Generation failed');
      }
    } catch (error) {
      console.error('Generation error:', error);
      toast.error('Generation error');
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { loading: false } });
    }
  };

  return (
    <div className={styles.mainPage}>
      {/* Voice Selection Section */}
      <section className="card section">
        <h2 className="section-title">
          <i className="fas fa-user-circle" />
          Voice Selection
        </h2>
        <VoiceSelector />
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
          disabled={loading || !selectedVoiceId || !textInput.trim()}
          className={styles.generateButton}
        >
          <i className="fas fa-magic" />
          Generate Speech
        </Button>
      </section>

      {/* Output Section */}
      {generatedAudio && (
        <section className="card section">
          <h2 className="section-title">
            <i className="fas fa-volume-up" />
            Generated Audio
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
              <div className={styles.audioActions}>
                <Button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = generatedAudio.audio_url;
                    link.download = `generated_${Date.now()}.wav`;
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

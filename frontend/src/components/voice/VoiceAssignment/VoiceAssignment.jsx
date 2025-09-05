import React from 'react';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import Button from '../../common/Button/Button';
import toast from 'react-hot-toast';
import styles from './VoiceAssignment.module.css';

const VoiceAssignment = () => {
  const { state, dispatch } = useApp();
  const { voices, speakerAssignments, selectedVoiceId } = state;

  const assignVoiceToSpeaker = async (speakerId) => {
    if (!selectedVoiceId) {
      toast.error('Please select a voice first');
      return;
    }

    try {
      await voiceService.assignVoiceToSpeaker(speakerId, selectedVoiceId);
      
      dispatch({
        type: 'SET_SPEAKER_ASSIGNMENT',
        payload: { speakerId, voiceId: selectedVoiceId }
      });
      
      toast.success(`Voice assigned to Speaker ${speakerId}`);
    } catch (error) {
      console.error('Failed to assign voice:', error);
      toast.error(`Failed to assign voice: ${error.message}`);
    }
  };

  const clearSpeakerAssignment = async (speakerId) => {
    try {
      await voiceService.clearSpeakerAssignment(speakerId);
      
      dispatch({
        type: 'CLEAR_SPEAKER_ASSIGNMENT',
        payload: { speakerId }
      });
      
      toast.success(`Speaker ${speakerId} assignment cleared`);
    } catch (error) {
      console.error('Failed to clear assignment:', error);
      toast.error(`Failed to clear assignment: ${error.message}`);
    }
  };

  const clearAllAssignments = async () => {
    try {
      await voiceService.clearAllSpeakerAssignments();
      
      dispatch({ type: 'CLEAR_ALL_SPEAKER_ASSIGNMENTS' });
      
      toast.success('All speaker assignments cleared');
    } catch (error) {
      console.error('Failed to clear all assignments:', error);
      toast.error(`Failed to clear assignments: ${error.message}`);
    }
  };

  const getVoiceName = (voiceId) => {
    if (!voiceId) return 'Unassigned';
    const voice = voices.find(v => v.id === voiceId);
    return voice ? voice.name : 'Unknown Voice';
  };

  const hasAssignments = Object.values(speakerAssignments).some(voiceId => voiceId !== null);

  return (
    <div className={styles.voiceAssignment}>
      <div className={styles.header}>
        <h3>Speaker Voice Assignments</h3>
        <p>Assign voices to speaker slots for multi-speaker generation</p>
      </div>
      
      <div className={styles.speakerGrid}>
        {[1, 2, 3, 4].map(speakerId => {
          const isAssigned = speakerAssignments[speakerId] !== null;
          const voiceName = getVoiceName(speakerAssignments[speakerId]);
          
          return (
            <div 
              key={speakerId} 
              className={`${styles.speakerSlot} ${
                isAssigned ? styles.assigned : styles.unassigned
              }`}
            >
              <div className={styles.speakerHeader}>
                <span className={styles.speakerId}>Speaker {speakerId}</span>
                <span className={styles.markerSyntax}>[{speakerId}]</span>
              </div>
              
              <div className={styles.voiceInfo}>
                <span className={styles.voiceName} title={voiceName}>
                  {voiceName}
                </span>
              </div>
              
              <div className={styles.actions}>
                <Button
                  onClick={() => assignVoiceToSpeaker(speakerId)}
                  variant="primary"
                  size="small"
                  disabled={!selectedVoiceId}
                  className={styles.assignButton}
                  title={selectedVoiceId ? `Assign ${getVoiceName(selectedVoiceId)} to Speaker ${speakerId}` : 'Select a voice first'}
                >
                  Use {speakerId}
                </Button>
                
                {isAssigned && (
                  <Button
                    onClick={() => clearSpeakerAssignment(speakerId)}
                    variant="secondary"
                    size="small"
                    className={styles.clearButton}
                    title={`Clear assignment for Speaker ${speakerId}`}
                  >
                    Clear
                  </Button>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {hasAssignments && (
        <div className={styles.footer}>
          <Button
            onClick={clearAllAssignments}
            variant="outline"
            size="small"
            className={styles.clearAllButton}
            title="Clear all speaker assignments"
          >
            <i className="fas fa-trash" />
            Clear All Assignments
          </Button>
        </div>
      )}
      
      <div className={styles.helpText}>
        <p>
          <strong>How to use:</strong> Select a voice above, then click "Use 1", "Use 2", etc. 
          In your text, use <code>[1]</code>, <code>[2]</code>, <code>[3]</code>, <code>[4]</code> 
          to control which speaker says what.
        </p>
        <p>
          <strong>Example:</strong> "Hello [1], how are you today [2]? I'm doing great [1]!"
        </p>
      </div>
    </div>
  );
};

export default VoiceAssignment;

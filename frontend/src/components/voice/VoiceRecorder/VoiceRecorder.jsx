import React from 'react';
import Button from '../../common/Button/Button';

// Temporary placeholder component - will be fully implemented later
const VoiceRecorder = ({ onVoiceAdded }) => {
  return (
    <div style={{ 
      padding: '2rem', 
      textAlign: 'center',
      background: 'var(--bg-tertiary)',
      borderRadius: 'var(--radius-lg)',
      border: '2px dashed var(--border-color)'
    }}>
      <div style={{ marginBottom: '1rem' }}>
        <i className="fas fa-microphone" style={{ fontSize: '3rem', color: 'var(--accent-primary)' }} />
      </div>
      <h3>Voice Recorder</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Voice recording functionality will be implemented in the next phase.
      </p>
      <Button variant="secondary" disabled>
        <i className="fas fa-microphone" />
        Record Voice (Coming Soon)
      </Button>
    </div>
  );
};

export default VoiceRecorder;

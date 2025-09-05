import React from 'react';
import Button from '../../common/Button/Button';
import styles from './VoiceCard.module.css';

const VoiceCard = ({ voice, isSelected, onSelect, onDelete }) => {
  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Unknown';
    }
  };

  const getVoiceTypeIcon = (voiceType) => {
    switch (voiceType?.toLowerCase()) {
      case 'recorded':
        return 'fa-microphone';
      case 'uploaded':
        return 'fa-upload';
      default:
        return 'fa-user-circle';
    }
  };

  const getVoiceTypeColor = (voiceType) => {
    switch (voiceType?.toLowerCase()) {
      case 'recorded':
        return 'var(--error)';
      case 'uploaded':
        return 'var(--info)';
      default:
        return 'var(--text-secondary)';
    }
  };

  return (
    <div 
      className={`${styles.voiceCard} ${isSelected ? styles.selected : ''}`}
      onClick={onSelect}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className={styles.selectedIndicator}>
          <i className="fas fa-check-circle" />
        </div>
      )}

      {/* Voice type badge */}
      <div className={styles.typeBadge}>
        <i 
          className={`fas ${getVoiceTypeIcon(voice.voice_type)}`}
          style={{ color: getVoiceTypeColor(voice.voice_type) }}
        />
        <span>{voice.voice_type || 'Voice'}</span>
      </div>

      {/* Voice info */}
      <div className={styles.voiceInfo}>
        <div className={styles.voiceIcon}>
          <i className="fas fa-user-circle" />
        </div>
        
        <div className={styles.voiceDetails}>
          <h3 className={styles.voiceName}>{voice.name}</h3>
          
          <div className={styles.metadata}>
            {voice.created_at && (
              <span className={styles.metaItem}>
                <i className="fas fa-calendar" />
                {formatDate(voice.created_at)}
              </span>
            )}
            
            {voice.audio_path && (
              <span className={styles.metaItem}>
                <i className="fas fa-file-audio" />
                Audio file
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className={styles.actions}>
        <Button
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          variant={isSelected ? 'success' : 'primary'}
          size="small"
          className={styles.selectBtn}
        >
          {isSelected ? (
            <>
              <i className="fas fa-check" />
              Selected
            </>
          ) : (
            <>
              <i className="fas fa-hand-pointer" />
              Select
            </>
          )}
        </Button>
        
        <Button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          variant="danger"
          size="small"
          title="Delete Voice"
        >
          <i className="fas fa-trash" />
        </Button>
      </div>
    </div>
  );
};

export default VoiceCard;

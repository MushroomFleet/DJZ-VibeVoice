import React, { useState, useRef, useEffect } from 'react';
import Button from '../../common/Button/Button';
import { audioService } from '../../../services/audioService';
import styles from './AudioCard.module.css';

const AudioCard = ({ audioFile, onDelete, onDownload, onSelect, isSelected }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const audioRef = useRef(null);

  const {
    filename,
    voice_name,
    duration: fileDuration,
    created_at,
    text_preview,
    file_size
  } = audioFile;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  const handlePlayPause = () => {
    const audio = audioRef.current;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e) => {
    const audio = audioRef.current;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    audioRef.current.volume = newVolume;
  };

  const formatTime = (time) => {
    if (isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className={`${styles.audioCard} ${isSelected ? styles.selected : ''}`}>
      <audio
        ref={audioRef}
        src={audioService.getAudioUrl(filename)}
        preload="metadata"
      />

      <div className={styles.cardHeader}>
        <div className={styles.selectCheckbox}>
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => onSelect(filename)}
            className={styles.checkbox}
          />
        </div>
        <div className={styles.fileInfo}>
          <h3 className={styles.voiceName}>{voice_name || 'Unknown Voice'}</h3>
          <p className={styles.filename}>{filename}</p>
        </div>
        <div className={styles.cardActions}>
          <Button
            onClick={() => onDownload(filename)}
            variant="secondary"
            size="small"
            title="Download"
            className={styles.actionButton}
          >
            <i className="fas fa-download" />
          </Button>
          <Button
            onClick={() => onDelete(filename)}
            variant="danger"
            size="small"
            title="Delete"
            className={styles.actionButton}
          >
            <i className="fas fa-trash" />
          </Button>
        </div>
      </div>

      <div className={styles.audioControls}>
        <Button
          onClick={handlePlayPause}
          variant="primary"
          size="small"
          className={styles.playButton}
        >
          <i className={`fas ${isPlaying ? 'fa-pause' : 'fa-play'}`} />
        </Button>

        <div className={styles.progressContainer}>
          <div className={styles.timeDisplay}>
            <span>{formatTime(currentTime)}</span>
          </div>
          <div 
            className={styles.progressBar}
            onClick={handleSeek}
          >
            <div 
              className={styles.progressFill}
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
          <div className={styles.timeDisplay}>
            <span>{formatTime(duration || fileDuration)}</span>
          </div>
        </div>

        <div className={styles.volumeControl}>
          <i className="fas fa-volume-up" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={handleVolumeChange}
            className={styles.volumeSlider}
          />
        </div>
      </div>

      {text_preview && (
        <div className={styles.textPreview}>
          <p title={text_preview}>
            <i className="fas fa-quote-left" />
            {text_preview}
          </p>
        </div>
      )}

      <div className={styles.metadata}>
        <div className={styles.metadataItem}>
          <i className="fas fa-calendar" />
          <span>{formatDate(created_at)}</span>
        </div>
        <div className={styles.metadataItem}>
          <i className="fas fa-clock" />
          <span>{formatTime(duration || fileDuration)}</span>
        </div>
        {file_size && (
          <div className={styles.metadataItem}>
            <i className="fas fa-file" />
            <span>{formatFileSize(file_size)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioCard;

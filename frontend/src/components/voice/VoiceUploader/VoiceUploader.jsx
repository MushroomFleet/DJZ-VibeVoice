import React, { useState } from 'react';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import Button from '../../common/Button/Button';
import toast from 'react-hot-toast';
import styles from './VoiceUploader.module.css';

const VoiceUploader = ({ onVoiceAdded }) => {
  const { dispatch } = useApp();
  const [voiceName, setVoiceName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      // Auto-generate name from filename if none exists
      if (!voiceName) {
        const nameWithoutExt = file.name.replace(/\.[^/.]+$/, '');
        setVoiceName(nameWithoutExt);
      }
    }
  }, [voiceName]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error('Please select an audio file');
      return;
    }

    if (!voiceName.trim()) {
      toast.error('Please enter a voice name');
      return;
    }

    setIsUploading(true);
    
    try {
      dispatch({ 
        type: 'SET_LOADING', 
        payload: { loading: true, text: 'Uploading voice...' } 
      });

      const result = await voiceService.uploadVoice(selectedFile, voiceName.trim());
      
      if (result.success) {
        onVoiceAdded(result.voice);
        toast.success('Voice uploaded successfully');
        
        // Reset form
        setSelectedFile(null);
        setVoiceName('');
      } else {
        toast.error(result.message || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Upload failed');
    } finally {
      setIsUploading(false);
      dispatch({ type: 'SET_LOADING', payload: { loading: false } });
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={styles.uploader}>
      {/* Voice name input */}
      <div className={styles.nameSection}>
        <label htmlFor="voice-name" className={styles.label}>
          Voice Name *
        </label>
        <input
          id="voice-name"
          type="text"
          value={voiceName}
          onChange={(e) => setVoiceName(e.target.value)}
          placeholder="Enter a name for this voice..."
          className={styles.nameInput}
          disabled={isUploading}
        />
      </div>

      {/* File upload area */}
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${
          isDragActive ? styles.dragActive : ''
        } ${isDragReject ? styles.dragReject : ''} ${
          selectedFile ? styles.hasFile : ''
        }`}
      >
        <input {...getInputProps()} />
        
        {selectedFile ? (
          <div className={styles.fileSelected}>
            <div className={styles.fileIcon}>
              <i className="fas fa-file-audio" />
            </div>
            <div className={styles.fileInfo}>
              <div className={styles.fileName}>{selectedFile.name}</div>
              <div className={styles.fileSize}>
                {formatFileSize(selectedFile.size)}
              </div>
            </div>
            <Button
              onClick={(e) => {
                e.stopPropagation();
                clearFile();
              }}
              variant="secondary"
              size="small"
              className={styles.clearBtn}
            >
              <i className="fas fa-times" />
            </Button>
          </div>
        ) : (
          <div className={styles.dropzoneContent}>
            <div className={styles.uploadIcon}>
              <i className="fas fa-cloud-upload-alt" />
            </div>
            <div className={styles.uploadText}>
              {isDragActive ? (
                isDragReject ? (
                  <p>Invalid file type. Please upload an audio file.</p>
                ) : (
                  <p>Drop the audio file here...</p>
                )
              ) : (
                <>
                  <p>Drag & drop an audio file here</p>
                  <p className={styles.uploadSubtext}>or click to browse</p>
                </>
              )}
            </div>
            <div className={styles.supportedFormats}>
              Supported: WAV, MP3, M4A, OGG, WebM, FLAC (Max 50MB)
            </div>
          </div>
        )}
      </div>

      {/* Upload button */}
      <Button
        onClick={handleUpload}
        variant="primary"
        size="large"
        disabled={!selectedFile || !voiceName.trim() || isUploading}
        className={styles.uploadBtn}
      >
        <i className="fas fa-upload" />
        {isUploading ? 'Uploading...' : 'Upload Voice'}
      </Button>
    </div>
  );
};

export default VoiceUploader;

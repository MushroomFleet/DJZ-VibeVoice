import React, { useState, useEffect } from 'react';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import Button from '../../common/Button/Button';
import styles from './TextInput.module.css';

const MultiSpeakerHelper = ({ show, referencedSpeakers, speakerAssignments }) => {
  if (!show) return null;
  
  const missingAssignments = referencedSpeakers.filter(
    speakerId => !speakerAssignments[speakerId]
  );
  
  return (
    <div className={styles.syntaxHelper}>
      <div className={styles.helperHeader}>
        <h4><i className="fas fa-users" /> Multi-Speaker Syntax</h4>
      </div>
      
      <div className={styles.helperContent}>
        <p>
          Use <code>[1]</code>, <code>[2]</code>, <code>[3]</code>, <code>[4]</code> to control which speaker says what:
        </p>
        
        <div className={styles.example}>
          <strong>Example:</strong><br/>
          <span className={styles.exampleText}>
            <span className={styles.speakerMarker}>[1]</span> Hello everyone, I want to introduce someone<br/>
            <span className={styles.speakerMarker}>[2]</span> Hey, my name's Sandy - hello all<br/>
            <span className={styles.speakerMarker}>[1]</span> She is going to be helping out now and then<br/>
            <span className={styles.speakerMarker}>[2]</span> You will have to get used to me hehe
          </span>
        </div>
        
        {referencedSpeakers.length > 0 && (
          <div className={styles.speakerValidation}>
            <strong>Speakers found in text:</strong>
            <ul className={styles.speakerList}>
              {referencedSpeakers.map(speakerId => (
                <li key={speakerId} className={speakerAssignments[speakerId] ? styles.assigned : styles.missing}>
                  <span className={styles.speakerMarker}>[{speakerId}]</span>
                  {speakerAssignments[speakerId] ? (
                    <span className={styles.status}>
                      <i className="fas fa-check" /> Assigned
                    </span>
                  ) : (
                    <span className={styles.status}>
                      <i className="fas fa-exclamation-triangle" /> Not assigned
                    </span>
                  )}
                </li>
              ))}
            </ul>
            
            {missingAssignments.length > 0 && (
              <div className={styles.warning}>
                <i className="fas fa-exclamation-triangle" />
                Please assign voices to speakers {missingAssignments.join(', ')} before generating.
              </div>
            )}
          </div>
        )}
        
        <p className={styles.note}>
          <i className="fas fa-info-circle" />
          Make sure to assign voices to speaker slots in the Voice Assignment section above!
        </p>
      </div>
    </div>
  );
};

const TextInput = () => {
  const { state, dispatch } = useApp();
  const { textInput, multiSpeakerMode, speakerAssignments } = state;
  const [activeTab, setActiveTab] = useState('manual');
  const [selectedFile, setSelectedFile] = useState(null);
  const [referencedSpeakers, setReferencedSpeakers] = useState([]);

  const updateText = (value) => {
    dispatch({ type: 'SET_TEXT_INPUT', payload: value });
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/plain') {
      setSelectedFile(file);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        updateText(e.target.result);
      };
      reader.readAsText(file);
    } else {
      alert('Please select a valid text file (.txt)');
    }
  };

  const getWordCount = () => {
    return textInput.trim() ? textInput.trim().split(/\s+/).length : 0;
  };

  const getCharCount = () => {
    return textInput.length;
  };

  const clearText = () => {
    updateText('');
    setSelectedFile(null);
  };

  // Extract referenced speakers when text changes
  useEffect(() => {
    if (multiSpeakerMode && textInput) {
      const speakers = voiceService.extractReferencedSpeakers(textInput);
      setReferencedSpeakers(speakers);
    } else {
      setReferencedSpeakers([]);
    }
  }, [textInput, multiSpeakerMode]);

  const highlightSpeakerMarkers = (text) => {
    if (!multiSpeakerMode) return text;
    return text.replace(/\[([1-4])\]/g, (match) => 
      `<span class="${styles.speakerHighlight}">${match}</span>`
    );
  };

  return (
    <div className={styles.textInput}>
      <div className={styles.tabs}>
        <Button
          variant={activeTab === 'manual' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('manual')}
          size="small"
        >
          <i className="fas fa-pen" /> Manual Input
        </Button>
        <Button
          variant={activeTab === 'file' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('file')}
          size="small"
        >
          <i className="fas fa-file-alt" /> Upload File
        </Button>
      </div>

      {activeTab === 'manual' && (
        <div className={styles.manualTab}>
          <div className={styles.textareaContainer}>
            <textarea
              value={textInput}
              onChange={(e) => updateText(e.target.value)}
              placeholder={multiSpeakerMode 
                ? "Enter your text here... Use [1], [2], [3], [4] for different speakers"
                : "Enter your text here..."
              }
              className={`${styles.textarea} ${multiSpeakerMode ? styles.multiSpeakerMode : ''}`}
              rows={8}
            />
            {textInput && (
              <Button
                onClick={clearText}
                variant="secondary"
                size="small"
                className={styles.clearButton}
                title="Clear text"
              >
                <i className="fas fa-times" />
              </Button>
            )}
          </div>
          <div className={styles.stats}>
            <span>{getCharCount()} characters</span>
            <span>{getWordCount()} words</span>
            {multiSpeakerMode && referencedSpeakers.length > 0 && (
              <span className={styles.speakerCount}>
                <i className="fas fa-users" />
                {referencedSpeakers.length} speaker{referencedSpeakers.length !== 1 ? 's' : ''} referenced
              </span>
            )}
          </div>
          
          {multiSpeakerMode && (
            <MultiSpeakerHelper 
              show={true}
              referencedSpeakers={referencedSpeakers}
              speakerAssignments={speakerAssignments}
            />
          )}
        </div>
      )}

      {activeTab === 'file' && (
        <div className={styles.fileTab}>
          <div className={styles.fileUpload}>
            <i className="fas fa-file-upload" />
            <p>Upload a text file (.txt)</p>
            <input
              type="file"
              accept=".txt"
              onChange={handleFileSelect}
              className={styles.fileInput}
              id="text-file-input"
            />
            <Button onClick={() => document.getElementById('text-file-input').click()}>
              <i className="fas fa-folder-open" />
              Choose File
            </Button>
            {selectedFile && (
              <div className={styles.fileInfo}>
                <p className={styles.fileName}>
                  <i className="fas fa-file-alt" />
                  Selected: {selectedFile.name}
                </p>
                <Button
                  onClick={() => {
                    setSelectedFile(null);
                    updateText('');
                  }}
                  variant="secondary"
                  size="small"
                >
                  <i className="fas fa-times" />
                  Remove
                </Button>
              </div>
            )}
          </div>
          
          {textInput && (
            <div className={styles.previewSection}>
              <h4>File Content Preview:</h4>
              <div className={styles.preview}>
                {textInput.substring(0, 500)}
                {textInput.length > 500 && '...'}
              </div>
              <div className={styles.stats}>
                <span>{getCharCount()} characters</span>
                <span>{getWordCount()} words</span>
                {multiSpeakerMode && referencedSpeakers.length > 0 && (
                  <span className={styles.speakerCount}>
                    <i className="fas fa-users" />
                    {referencedSpeakers.length} speaker{referencedSpeakers.length !== 1 ? 's' : ''} referenced
                  </span>
                )}
              </div>
              
              {multiSpeakerMode && (
                <MultiSpeakerHelper 
                  show={true}
                  referencedSpeakers={referencedSpeakers}
                  speakerAssignments={speakerAssignments}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TextInput;

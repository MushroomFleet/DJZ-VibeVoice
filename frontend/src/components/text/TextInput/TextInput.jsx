import React, { useState, useEffect } from 'react';
import { useApp } from '../../../contexts/AppContext';
import Button from '../../common/Button/Button';
import styles from './TextInput.module.css';

const TextInput = () => {
  const { state, dispatch } = useApp();
  const { textInput } = state;
  const [activeTab, setActiveTab] = useState('manual');
  const [selectedFile, setSelectedFile] = useState(null);

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
              placeholder="Enter your text here..."
              className={styles.textarea}
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
          </div>
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
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TextInput;

import React, { useState } from 'react';
import { useApp } from '../../../contexts/AppContext';
import Button from '../../common/Button/Button';
import AudioLibrary from '../../audio/AudioLibrary/AudioLibrary';
import styles from './Header.module.css';

const Header = () => {
  const { state, dispatch } = useApp();
  const { theme, audioLibrary } = state;
  const [audioLibraryOpen, setAudioLibraryOpen] = useState(false);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    dispatch({ type: 'SET_THEME', payload: newTheme });
  };

  return (
    <>
      <header className={styles.header}>
        <div className={styles.content}>
          <div className={styles.logo}>
            <i className="fas fa-microphone-lines" />
            <h1>DJZ-VibeVoice</h1>
          </div>
          
          <div className={styles.actions}>
            <Button
              onClick={() => setAudioLibraryOpen(true)}
              variant="secondary"
              size="small"
              className={styles.libraryButton}
              title="Audio Library"
            >
              <i className="fas fa-folder-open" />
              {audioLibrary.length > 0 && (
                <span className={styles.badge}>{audioLibrary.length}</span>
              )}
            </Button>
            
            <Button
              onClick={toggleTheme}
              variant="secondary"
              size="small"
              title="Toggle Theme"
            >
              <i className={`fas ${theme === 'light' ? 'fa-moon' : 'fa-sun'}`} />
            </Button>
          </div>
        </div>
      </header>

      {/* Audio Library Modal */}
      <AudioLibrary 
        isOpen={audioLibraryOpen} 
        onClose={() => setAudioLibraryOpen(false)} 
      />
    </>
  );
};

export default Header;

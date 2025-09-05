import React, { useState, useEffect } from 'react';
import { useApp } from '../../../contexts/AppContext';
import { voiceService } from '../../../services/voiceService';
import VoiceCard from '../VoiceCard/VoiceCard';
import VoiceUploader from '../VoiceUploader/VoiceUploader';
import VoiceRecorder from '../VoiceRecorder/VoiceRecorder';
import Button from '../../common/Button/Button';
import toast from 'react-hot-toast';
import styles from './VoiceSelector.module.css';

const VoiceSelector = () => {
  const { state, dispatch } = useApp();
  const { voices, selectedVoiceId, loading } = state;
  const [activeTab, setActiveTab] = useState('select');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredVoices, setFilteredVoices] = useState([]);

  useEffect(() => {
    loadVoices();
  }, []);

  useEffect(() => {
    // Filter voices based on search query
    if (!searchQuery.trim()) {
      setFilteredVoices(voices);
    } else {
      const query = searchQuery.toLowerCase();
      const filtered = voices.filter(voice =>
        voice.name.toLowerCase().includes(query) ||
        voice.voice_type?.toLowerCase().includes(query)
      );
      setFilteredVoices(filtered);
    }
  }, [voices, searchQuery]);

  const loadVoices = async () => {
    try {
      dispatch({ 
        type: 'SET_LOADING', 
        payload: { loading: true, text: 'Loading voices...' } 
      });
      
      const voicesData = await voiceService.getVoices();
      dispatch({ type: 'SET_VOICES', payload: voicesData });
      
      // Auto-select first voice if none selected
      if (!selectedVoiceId && voicesData.length > 0) {
        dispatch({ type: 'SELECT_VOICE', payload: voicesData[0].id });
      }
    } catch (error) {
      console.error('Failed to load voices:', error);
      toast.error('Failed to load voices');
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { loading: false } });
    }
  };

  const handleVoiceSelect = (voiceId) => {
    dispatch({ type: 'SELECT_VOICE', payload: voiceId });
    toast.success('Voice selected');
  };

  const handleVoiceDelete = async (voiceId, voiceName) => {
    if (!window.confirm(`Are you sure you want to delete "${voiceName}"?`)) {
      return;
    }

    try {
      dispatch({ 
        type: 'SET_LOADING', 
        payload: { loading: true, text: 'Deleting voice...' } 
      });
      
      await voiceService.deleteVoice(voiceId);
      dispatch({ type: 'DELETE_VOICE', payload: voiceId });
      toast.success('Voice deleted successfully');
    } catch (error) {
      console.error('Failed to delete voice:', error);
      toast.error('Failed to delete voice');
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { loading: false } });
    }
  };

  const handleVoiceAdded = (newVoice) => {
    dispatch({ type: 'ADD_VOICE', payload: newVoice });
    dispatch({ type: 'SELECT_VOICE', payload: newVoice.id });
    setActiveTab('select');
  };

  return (
    <div className={styles.voiceSelector}>
      {/* Header with search and refresh */}
      <div className={styles.header}>
        <div className={styles.searchContainer}>
          <div className={styles.searchBox}>
            <i className="fas fa-search" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search voices..."
              className={styles.searchInput}
            />
          </div>
          <Button
            onClick={loadVoices}
            variant="secondary"
            size="small"
            disabled={loading}
          >
            <i className="fas fa-sync-alt" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        <Button
          variant={activeTab === 'select' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('select')}
          size="small"
        >
          <i className="fas fa-list" />
          Select Voice
        </Button>
        <Button
          variant={activeTab === 'upload' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('upload')}
          size="small"
        >
          <i className="fas fa-upload" />
          Upload Voice
        </Button>
        <Button
          variant={activeTab === 'record' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('record')}
          size="small"
        >
          <i className="fas fa-microphone" />
          Record Voice
        </Button>
      </div>

      {/* Tab Content */}
      <div className={styles.tabContent}>
        {activeTab === 'select' && (
          <div className={styles.selectTab}>
            {filteredVoices.length === 0 ? (
              <div className={styles.noResults}>
                <i className="fas fa-microphone-slash" />
                <p>
                  {voices.length === 0 
                    ? 'No voices found. Upload or record a voice to get started.'
                    : 'No voices match your search.'
                  }
                </p>
              </div>
            ) : (
              <div className={styles.voiceGrid}>
                {filteredVoices.map((voice) => (
                  <VoiceCard
                    key={voice.id}
                    voice={voice}
                    isSelected={voice.id === selectedVoiceId}
                    onSelect={() => handleVoiceSelect(voice.id)}
                    onDelete={() => handleVoiceDelete(voice.id, voice.name)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'upload' && (
          <VoiceUploader onVoiceAdded={handleVoiceAdded} />
        )}

        {activeTab === 'record' && (
          <VoiceRecorder onVoiceAdded={handleVoiceAdded} />
        )}
      </div>
    </div>
  );
};

export default VoiceSelector;

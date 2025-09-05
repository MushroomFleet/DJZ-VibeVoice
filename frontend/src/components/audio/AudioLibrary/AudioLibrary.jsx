import React, { useState, useEffect } from 'react';
import { audioService } from '../../../services/audioService';
import Button from '../../common/Button/Button';
import LoadingOverlay from '../../common/LoadingOverlay/LoadingOverlay';
import AudioCard from '../AudioCard/AudioCard';
import toast from 'react-hot-toast';
import styles from './AudioLibrary.module.css';

const AudioLibrary = ({ isOpen, onClose }) => {
  const [audioFiles, setAudioFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedFiles, setSelectedFiles] = useState(new Set());

  useEffect(() => {
    if (isOpen) {
      fetchAudioLibrary();
    }
  }, [isOpen]);

  const fetchAudioLibrary = async () => {
    setLoading(true);
    try {
      const files = await audioService.getAudioLibrary(searchQuery);
      setAudioFiles(files);
    } catch (error) {
      console.error('Failed to fetch audio library:', error);
      toast.error('Failed to load audio library');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    await fetchAudioLibrary();
  };

  const handleDeleteFile = async (filename) => {
    try {
      await audioService.deleteAudio(filename);
      setAudioFiles(prev => prev.filter(file => file.filename !== filename));
      setSelectedFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(filename);
        return newSet;
      });
      toast.success('Audio file deleted');
    } catch (error) {
      console.error('Failed to delete audio:', error);
      toast.error('Failed to delete audio file');
    }
  };

  const handleDownloadFile = (filename) => {
    audioService.downloadAudio(filename);
    toast.success('Download started');
  };

  const handleSelectFile = (filename) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(filename)) {
        newSet.delete(filename);
      } else {
        newSet.add(filename);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    if (selectedFiles.size === sortedFiles.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(sortedFiles.map(file => file.filename)));
    }
  };

  const handleBulkDelete = async () => {
    if (selectedFiles.size === 0) return;
    
    const confirmDelete = window.confirm(
      `Are you sure you want to delete ${selectedFiles.size} audio file(s)?`
    );
    
    if (!confirmDelete) return;

    setLoading(true);
    try {
      await Promise.all(
        Array.from(selectedFiles).map(filename => 
          audioService.deleteAudio(filename)
        )
      );
      
      setAudioFiles(prev => prev.filter(file => !selectedFiles.has(file.filename)));
      setSelectedFiles(new Set());
      toast.success(`Deleted ${selectedFiles.size} audio file(s)`);
    } catch (error) {
      console.error('Failed to delete files:', error);
      toast.error('Failed to delete some files');
    } finally {
      setLoading(false);
    }
  };

  const sortedFiles = [...audioFiles].sort((a, b) => {
    let aValue, bValue;
    
    switch (sortBy) {
      case 'name':
        aValue = a.voice_name || '';
        bValue = b.voice_name || '';
        break;
      case 'duration':
        aValue = a.duration || 0;
        bValue = b.duration || 0;
        break;
      case 'date':
      default:
        aValue = new Date(a.created_at || 0);
        bValue = new Date(b.created_at || 0);
        break;
    }
    
    if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
    return 0;
  });

  if (!isOpen) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>
            <i className="fas fa-folder-open" />
            Audio Library
            {audioFiles.length > 0 && (
              <span className={styles.count}>({audioFiles.length})</span>
            )}
          </h2>
          <Button
            onClick={onClose}
            variant="secondary"
            size="small"
            className={styles.closeButton}
          >
            <i className="fas fa-times" />
          </Button>
        </div>

        <div className={styles.controls}>
          <form onSubmit={handleSearch} className={styles.searchForm}>
            <div className={styles.searchInput}>
              <i className="fas fa-search" />
              <input
                type="text"
                placeholder="Search by voice name or text..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button type="submit" variant="primary" size="small">
              Search
            </Button>
          </form>

          <div className={styles.sortControls}>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className={styles.sortSelect}
            >
              <option value="date">Date</option>
              <option value="name">Voice Name</option>
              <option value="duration">Duration</option>
            </select>
            <Button
              onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              variant="secondary"
              size="small"
              title={`Sort ${sortOrder === 'asc' ? 'Descending' : 'Ascending'}`}
            >
              <i className={`fas fa-sort-${sortOrder === 'asc' ? 'up' : 'down'}`} />
            </Button>
          </div>

          <div className={styles.actionControls}>
            <Button
              onClick={fetchAudioLibrary}
              variant="secondary"
              size="small"
              disabled={loading}
              title="Refresh Library"
            >
              <i className="fas fa-refresh" />
            </Button>
            
            {sortedFiles.length > 0 && (
              <>
                <Button
                  onClick={handleSelectAll}
                  variant="secondary"
                  size="small"
                  title={selectedFiles.size === sortedFiles.length ? 'Deselect All' : 'Select All'}
                >
                  <i className={`fas fa-${selectedFiles.size === sortedFiles.length ? 'square' : 'check-square'}`} />
                  {selectedFiles.size > 0 && ` (${selectedFiles.size})`}
                </Button>
                
                {selectedFiles.size > 0 && (
                  <Button
                    onClick={handleBulkDelete}
                    variant="danger"
                    size="small"
                    disabled={loading}
                    title="Delete Selected"
                  >
                    <i className="fas fa-trash" />
                    Delete ({selectedFiles.size})
                  </Button>
                )}
              </>
            )}
          </div>
        </div>

        <div className={styles.content}>
          {loading && <LoadingOverlay text="Loading audio library..." />}
          
          {!loading && sortedFiles.length === 0 && (
            <div className={styles.emptyState}>
              <i className="fas fa-music" />
              <h3>No Audio Files</h3>
              <p>Generate some speech to see your audio library here.</p>
            </div>
          )}

          {!loading && sortedFiles.length > 0 && (
            <div className={styles.audioGrid}>
              {sortedFiles.map((audioFile) => (
                <AudioCard
                  key={audioFile.filename}
                  audioFile={audioFile}
                  onDelete={handleDeleteFile}
                  onDownload={handleDownloadFile}
                  onSelect={handleSelectFile}
                  isSelected={selectedFiles.has(audioFile.filename)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AudioLibrary;

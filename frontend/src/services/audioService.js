import { api } from './api';

class AudioService {
  async getAudioLibrary(searchQuery = '') {
    const params = searchQuery ? { search: searchQuery } : {};
    const response = await api.get('/audio/library', { params });
    return response.data.audio_files || [];
  }

  async deleteAudio(filename) {
    const response = await api.delete(`/audio/${filename}`);
    return response.data;
  }

  getAudioUrl(filename) {
    return `/api/audio/${filename}`;
  }

  downloadAudio(filename) {
    const link = document.createElement('a');
    link.href = this.getAudioUrl(filename);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Utility function to create audio blob from URL
  async fetchAudioAsBlob(url) {
    const response = await fetch(url);
    return await response.blob();
  }

  // Utility function to get audio duration
  async getAudioDuration(audioUrl) {
    return new Promise((resolve) => {
      const audio = new Audio(audioUrl);
      audio.addEventListener('loadedmetadata', () => {
        resolve(audio.duration);
      });
      audio.addEventListener('error', () => {
        resolve(0);
      });
    });
  }
}

export const audioService = new AudioService();

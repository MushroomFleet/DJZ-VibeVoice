import { api } from './api';

class VoiceService {
  async getVoices(searchQuery = '') {
    const params = searchQuery ? { search: searchQuery } : {};
    const response = await api.get('/voices', { params });
    return response.data;
  }

  async uploadVoice(file, name) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);

    const response = await api.post('/voices/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async saveRecording(name, blob) {
    // Convert blob to base64
    const base64Data = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.readAsDataURL(blob);
    });

    const response = await api.post('/voices/record', {
      name,
      audio_data: base64Data,
      format: 'webm'
    });
    return response.data;
  }

  async deleteVoice(voiceId) {
    const response = await api.delete(`/voices/${voiceId}`);
    return response.data;
  }

  async generateSpeech(request) {
    const response = await api.post('/generate', request);
    return response.data;
  }

  async generateFromFile(file, voiceId, cfgScale = 1.3) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('voice_id', voiceId);
    formData.append('cfg_scale', cfgScale);

    const response = await api.post('/generate/file', formData);
    return response.data;
  }

  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  }
}

export const voiceService = new VoiceService();

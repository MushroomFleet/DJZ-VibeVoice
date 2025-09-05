import { api } from './api';

class VoiceService {
  constructor() {
    this.backendReady = false;
    this.checkingBackend = false;
  }

  async waitForBackend(maxRetries = 10, delayMs = 1000) {
    if (this.backendReady) return true;
    if (this.checkingBackend) {
      // Wait for ongoing check
      while (this.checkingBackend) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return this.backendReady;
    }

    this.checkingBackend = true;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        await api.get('/health', { timeout: 2000 });
        this.backendReady = true;
        this.checkingBackend = false;
        return true;
      } catch (error) {
        console.log(`Backend not ready, attempt ${i + 1}/${maxRetries}...`);
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }
    
    this.checkingBackend = false;
    return false;
  }

  async withRetry(operation, retries = 3) {
    // First ensure backend is ready
    const backendReady = await this.waitForBackend();
    if (!backendReady) {
      throw new Error('Backend is not ready. Please wait a moment and try again.');
    }

    for (let i = 0; i < retries; i++) {
      try {
        return await operation();
      } catch (error) {
        console.log(`Operation failed, attempt ${i + 1}/${retries}:`, error.message);
        
        // If it's a connection error and not the last retry, wait and try again
        if (error.code === 'ECONNABORTED' || error.message.includes('Network Error') || error.response?.status >= 500) {
          if (i < retries - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            continue;
          }
        }
        throw error;
      }
    }
  }

  async getVoices(searchQuery = '') {
    return this.withRetry(async () => {
      const params = searchQuery ? { search: searchQuery } : {};
      const response = await api.get('/voices', { params });
      return response.data;
    });
  }

  async uploadVoice(file, name) {
    return this.withRetry(async () => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', name);

      const response = await api.post('/voices/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    });
  }

  async saveRecording(name, blob) {
    return this.withRetry(async () => {
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
    });
  }

  async deleteVoice(voiceId) {
    return this.withRetry(async () => {
      const response = await api.delete(`/voices/${voiceId}`);
      return response.data;
    });
  }

  async generateSpeech(request) {
    // Don't use retry logic for speech generation as it can take a long time
    const backendReady = await this.waitForBackend();
    if (!backendReady) {
      throw new Error('Backend is not ready. Please wait a moment and try again.');
    }
    
    const response = await api.post('/generate', request, {
      timeout: 120000, // 2 minutes timeout for CPU generation
    });
    return response.data;
  }

  async generateFromFile(file, voiceId, cfgScale = 1.3) {
    return this.withRetry(async () => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('voice_id', voiceId);
      formData.append('cfg_scale', cfgScale);

      const response = await api.post('/generate/file', formData);
      return response.data;
    });
  }

  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  }

  // Multi-Speaker Methods

  async assignVoiceToSpeaker(speakerId, voiceId) {
    return this.withRetry(async () => {
      const response = await api.post(`/voices/assign-speaker/${speakerId}`, { 
        voice_id: voiceId 
      });
      return response.data;
    });
  }

  async getSpeakerAssignments() {
    return this.withRetry(async () => {
      const response = await api.get('/voices/speaker-assignments');
      return response.data;
    });
  }

  async clearSpeakerAssignment(speakerId) {
    return this.withRetry(async () => {
      const response = await api.delete(`/voices/speaker-assignments/${speakerId}`);
      return response.data;
    });
  }

  async clearAllSpeakerAssignments() {
    return this.withRetry(async () => {
      const response = await api.delete('/voices/speaker-assignments');
      return response.data;
    });
  }

  async generateMultiSpeakerSpeech({ text, speaker_assignments, cfg_scale = 1.3 }) {
    // Don't use retry logic for speech generation as it can take a long time
    const backendReady = await this.waitForBackend();
    if (!backendReady) {
      throw new Error('Backend is not ready. Please wait a moment and try again.');
    }
    
    try {
      const response = await api.post('/voices/generate-multi-speaker', {
        text,
        speaker_assignments,
        cfg_scale
      }, {
        timeout: 120000, // 2 minutes timeout for CPU generation
      });
      
      if (response.data.success) {
        return {
          success: true,
          audio_url: response.data.audio_url,
          duration: response.data.duration,
          referenced_speakers: response.data.referenced_speakers,
          message: response.data.message
        };
      } else {
        throw new Error(response.data.message || 'Multi-speaker generation failed');
      }
    } catch (error) {
      console.error('Multi-speaker generation error:', error);
      throw error;
    }
  }

  // Helper method to extract referenced speakers from text
  extractReferencedSpeakers(text) {
    const matches = text.match(/\[([1-4])\]/g);
    if (!matches) return [];
    return [...new Set(matches.map(match => parseInt(match.slice(1, -1))))];
  }

  // Helper method to validate speaker assignments
  validateSpeakerAssignments(text, speakerAssignments) {
    const referencedSpeakers = this.extractReferencedSpeakers(text);
    const missingAssignments = referencedSpeakers.filter(
      speakerId => !speakerAssignments[speakerId]
    );
    
    return {
      isValid: missingAssignments.length === 0,
      referencedSpeakers,
      missingAssignments
    };
  }
}

export const voiceService = new VoiceService();

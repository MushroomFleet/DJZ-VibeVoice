import React, { createContext, useContext, useReducer } from 'react';

const AppContext = createContext();

const initialState = {
  loading: false,
  loadingText: '',
  theme: 'light',
  selectedVoiceId: null,
  voices: [],
  audioLibrary: [],
  generatedAudio: null,
  textInput: '',
  generationSettings: {
    cfgScale: 1.3,
    numSpeakers: 1,
    outputFormat: 'wav'
  }
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { 
        ...state, 
        loading: action.payload.loading,
        loadingText: action.payload.text || ''
      };
    
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    
    case 'SET_VOICES':
      return { ...state, voices: action.payload };
    
    case 'ADD_VOICE':
      return { 
        ...state, 
        voices: [...state.voices, action.payload] 
      };
    
    case 'DELETE_VOICE':
      return {
        ...state,
        voices: state.voices.filter(v => v.id !== action.payload),
        selectedVoiceId: state.selectedVoiceId === action.payload ? null : state.selectedVoiceId
      };
    
    case 'SELECT_VOICE':
      return { ...state, selectedVoiceId: action.payload };
    
    case 'SET_AUDIO_LIBRARY':
      return { ...state, audioLibrary: action.payload };
    
    case 'ADD_AUDIO_TO_LIBRARY':
      return { 
        ...state, 
        audioLibrary: [...state.audioLibrary, action.payload] 
      };
    
    case 'DELETE_AUDIO_FROM_LIBRARY':
      return {
        ...state,
        audioLibrary: state.audioLibrary.filter(audio => audio.filename !== action.payload)
      };
    
    case 'SET_GENERATED_AUDIO':
      return { ...state, generatedAudio: action.payload };
    
    case 'SET_TEXT_INPUT':
      return { ...state, textInput: action.payload };
    
    case 'UPDATE_GENERATION_SETTINGS':
      return { 
        ...state, 
        generationSettings: { 
          ...state.generationSettings, 
          ...action.payload 
        } 
      };
    
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Initialize theme from localStorage
  React.useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    dispatch({ type: 'SET_THEME', payload: savedTheme });
  }, []);

  // Save theme to localStorage when it changes
  React.useEffect(() => {
    localStorage.setItem('theme', state.theme);
    document.body.setAttribute('data-theme', state.theme);
  }, [state.theme]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};

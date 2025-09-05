import React from 'react';
import { useApp } from '../../../contexts/AppContext';
import styles from './GenerationSettings.module.css';

const GenerationSettings = () => {
  const { state, dispatch } = useApp();
  const { generationSettings } = state;

  const updateSetting = (key, value) => {
    dispatch({
      type: 'UPDATE_GENERATION_SETTINGS',
      payload: { [key]: value }
    });
  };

  return (
    <div className={styles.settings}>
      <h3 className={styles.title}>Generation Settings</h3>
      
      <div className={styles.grid}>
        <div className={styles.setting}>
          <label htmlFor="cfg-scale" className={styles.label}>
            Voice Strength
            <span className={styles.tooltip}>
              <i className="fas fa-info-circle" />
              <span className={styles.tooltipText}>
                Controls how closely the generated speech follows the voice characteristics. 
                Higher values produce more distinctive voice qualities.
              </span>
            </span>
          </label>
          <div className={styles.rangeInput}>
            <input
              id="cfg-scale"
              type="range"
              min="1.0"
              max="2.0"
              step="0.1"
              value={generationSettings.cfgScale}
              onChange={(e) => updateSetting('cfgScale', parseFloat(e.target.value))}
              className={styles.slider}
            />
            <span className={styles.value}>{generationSettings.cfgScale}</span>
          </div>
          <div className={styles.rangeLabels}>
            <span>Natural</span>
            <span>Distinctive</span>
          </div>
        </div>

        <div className={styles.setting}>
          <label htmlFor="num-speakers" className={styles.label}>
            Number of Speakers
            <span className={styles.tooltip}>
              <i className="fas fa-info-circle" />
              <span className={styles.tooltipText}>
                For multi-speaker conversations. The AI will automatically 
                assign different voices to different speakers in the text.
              </span>
            </span>
          </label>
          <select
            id="num-speakers"
            value={generationSettings.numSpeakers}
            onChange={(e) => updateSetting('numSpeakers', parseInt(e.target.value))}
            className={styles.select}
          >
            <option value={1}>1 Speaker (Monologue)</option>
            <option value={2}>2 Speakers (Dialogue)</option>
            <option value={3}>3 Speakers (Group)</option>
            <option value={4}>4 Speakers (Panel)</option>
          </select>
        </div>

        <div className={styles.setting}>
          <label htmlFor="output-format" className={styles.label}>
            Output Format
            <span className={styles.tooltip}>
              <i className="fas fa-info-circle" />
              <span className={styles.tooltipText}>
                Audio file format for the generated speech. 
                WAV provides highest quality, MP3 is more compressed.
              </span>
            </span>
          </label>
          <select
            id="output-format"
            value={generationSettings.outputFormat}
            onChange={(e) => updateSetting('outputFormat', e.target.value)}
            className={styles.select}
          >
            <option value="wav">WAV (High Quality)</option>
            <option value="mp3">MP3 (Compressed)</option>
          </select>
        </div>
      </div>

      <div className={styles.presets}>
        <h4 className={styles.presetsTitle}>Quick Presets</h4>
        <div className={styles.presetButtons}>
          <button
            className={styles.presetBtn}
            onClick={() => {
              updateSetting('cfgScale', 1.1);
              updateSetting('numSpeakers', 1);
            }}
          >
            <i className="fas fa-microphone" />
            <span>Natural</span>
          </button>
          <button
            className={styles.presetBtn}
            onClick={() => {
              updateSetting('cfgScale', 1.3);
              updateSetting('numSpeakers', 1);
            }}
          >
            <i className="fas fa-star" />
            <span>Balanced</span>
          </button>
          <button
            className={styles.presetBtn}
            onClick={() => {
              updateSetting('cfgScale', 1.7);
              updateSetting('numSpeakers', 1);
            }}
          >
            <i className="fas fa-fire" />
            <span>Dramatic</span>
          </button>
          <button
            className={styles.presetBtn}
            onClick={() => {
              updateSetting('cfgScale', 1.3);
              updateSetting('numSpeakers', 2);
            }}
          >
            <i className="fas fa-users" />
            <span>Conversation</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default GenerationSettings;

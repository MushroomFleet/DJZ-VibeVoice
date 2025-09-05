import React from 'react';
import styles from './LoadingOverlay.module.css';

const LoadingOverlay = ({ text = 'Loading...', isVisible = false }) => {
  if (!isVisible) return null;

  return (
    <div className={styles.overlay}>
      <div className={styles.content}>
        <div className={styles.spinner}>
          <div className={styles.spinnerInner}></div>
        </div>
        <p className={styles.text}>{text}</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;

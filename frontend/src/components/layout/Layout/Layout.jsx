import React from 'react';
import Header from '../Header/Header';
import LoadingOverlay from '../../common/LoadingOverlay/LoadingOverlay';
import { useApp } from '../../../contexts/AppContext';
import styles from './Layout.module.css';

const Layout = ({ children }) => {
  const { state } = useApp();
  const { loading, loadingText } = state;

  return (
    <div className={styles.layout}>
      <Header />
      <main className={styles.main}>
        <div className={styles.container}>
          {children}
        </div>
      </main>
      
      <LoadingOverlay isVisible={loading} text={loadingText} />
    </div>
  );
};

export default Layout;

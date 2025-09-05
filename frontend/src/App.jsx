import React, { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import { AppProvider } from './contexts/AppContext';
import Layout from './components/layout/Layout/Layout';
import MainPage from './components/pages/MainPage/MainPage';
import './styles/globals.css';

function App() {
  useEffect(() => {
    // Initialize theme on app load
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
  }, []);

  return (
    <AppProvider>
      <Layout>
        <MainPage />
      </Layout>
      
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-color)'
          }
        }}
      />
    </AppProvider>
  );
}

export default App;

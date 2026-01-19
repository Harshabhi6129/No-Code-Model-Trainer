// src/theme.ts
export const theme = {
  colors: {
    // Dark mode primary
    background: {
      primary: '#0a0a0a',      // Deep black
      secondary: '#141414',     // Card backgrounds
      tertiary: '#1a1a1a',      // Elevated surfaces
      glass: 'rgba(255, 255, 255, 0.05)',  // Glass panels
    },
    
    // Accent colors
    accent: {
      primary: '#6366f1',       // Indigo
      secondary: '#8b5cf6',     // Purple
      success: '#10b981',       // Green
      warning: '#f59e0b',       // Amber
      error: '#ef4444',         // Red
      info: '#3b82f6',          // Blue
    },
    
    // Text
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
      tertiary: 'rgba(255, 255, 255, 0.5)',
    },
    
    // Borders & dividers
    border: {
      default: 'rgba(255, 255, 255, 0.1)',
      focus: 'rgba(99, 102, 241, 0.5)',
    }
  },
  
  // Glassmorphism effects
  glass: {
    light: {
      background: 'rgba(255, 255, 255, 0.05)',
      backdrop: 'blur(12px)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      shadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
    },
    medium: {
      background: 'rgba(255, 255, 255, 0.08)',
      backdrop: 'blur(16px)',
      border: '1px solid rgba(255, 255, 255, 0.15)',
      shadow: '0 12px 48px 0 rgba(0, 0, 0, 0.5)',
    },
    strong: {
      background: 'rgba(255, 255, 255, 0.12)',
      backdrop: 'blur(20px)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      shadow: '0 16px 64px 0 rgba(0, 0, 0, 0.6)',
    }
  },
  
  // Animation timings
  animation: {
    fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
    normal: '300ms cubic-bezier(0.4, 0, 0.2, 1)',
    slow: '500ms cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'spring(1, 100, 10, 0)',
  },
  
  // Shadows
  shadow: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
    glow: '0 0 20px rgba(99, 102, 241, 0.3)',
  }
};

export const glassMorphismClasses = {
  light: 'backdrop-blur-xl bg-white/5 border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.37)]',
  medium: 'backdrop-blur-xl bg-white/8 border border-white/15 shadow-[0_12px_48px_0_rgba(0,0,0,0.5)]',
  strong: 'backdrop-blur-xl bg-white/12 border border-white/20 shadow-[0_16px_64px_0_rgba(0,0,0,0.6)]'
};
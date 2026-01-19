// src/components/GlassCard.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'light' | 'medium' | 'strong';
  animate?: boolean;
  hover?: boolean;
}

const glassVariants = {
  light: 'backdrop-blur-xl bg-white/5 border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.37)]',
  medium: 'backdrop-blur-xl bg-white/8 border border-white/15 shadow-[0_12px_48px_0_rgba(0,0,0,0.5)]',
  strong: 'backdrop-blur-xl bg-white/12 border border-white/20 shadow-[0_16px_64px_0_rgba(0,0,0,0.6)]'
};

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  variant = 'medium',
  animate = true,
  hover = true
}) => {
  const baseClasses = `${glassVariants[variant]} rounded-2xl p-6 ${className}`;
  
  const card = (
    <div className={`${baseClasses} ${hover ? 'hover:bg-white/10 transition-all duration-300' : ''}`}>
      {children}
    </div>
  );

  return animate ? (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={hover ? { y: -5, scale: 1.02 } : undefined}
    >
      {card}
    </motion.div>
  ) : card;
};
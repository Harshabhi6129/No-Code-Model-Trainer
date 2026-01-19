// src/components/ModernDashboard.tsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './GlassCard';
import {
  Plus,
  Clock,
  Layers,
  TrendingUp,
  Loader,
  Play,
  Pause,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'training' | 'completed' | 'failed' | 'paused';
  progress: number;
  lastUpdated: string;
  model: string;
  accuracy?: number;
}

export const ModernDashboard: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([
    {
      id: '1',
      name: 'Sentiment Analysis',
      description: 'Customer review classification',
      status: 'training',
      progress: 65,
      lastUpdated: '2 hours ago',
      model: 'BERT-base',
      accuracy: 0.89
    },
    {
      id: '2', 
      name: 'Named Entity Recognition',
      description: 'Extract entities from news articles',
      status: 'completed',
      progress: 100,
      lastUpdated: '1 day ago',
      model: 'RoBERTa-large',
      accuracy: 0.94
    }
  ]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] relative overflow-hidden">
      {/* Animated gradient background */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 8, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.3, 0.5],
          }}
          transition={{ duration: 10, repeat: Infinity }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
            Your ML Projects
          </h1>
          <p className="text-gray-400 text-lg">
            Fine-tune models without writing code
          </p>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <button className="group relative px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl font-semibold overflow-hidden">
            <span className="relative z-10 flex items-center gap-2">
              <Plus className="w-5 h-5" /> New Project
            </span>
            {/* Hover shimmer effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          </button>
        </motion.div>

        {/* Project Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project, index) => (
            <ProjectCard key={project.id} project={project} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
};

// Glassmorphic Project Card Component
const ProjectCard: React.FC<{ project: Project; index: number }> = ({ project, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={{ scale: 1.02, y: -5 }}
      className="group relative"
    >
      {/* Glass card */}
      <div className="relative backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl overflow-hidden">
        {/* Gradient border on hover */}
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl" />
        
        {/* Content */}
        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-white mb-2">
                {project.name}
              </h3>
              <p className="text-gray-400 text-sm">{project.description}</p>
            </div>
            <StatusBadge status={project.status} />
          </div>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-2">
              <span>Progress</span>
              <span>{project.progress}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${project.progress}%` }}
                transition={{ duration: 1, delay: index * 0.1 + 0.3 }}
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
              />
            </div>
          </div>

          {/* Metadata */}
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              {project.lastUpdated}
            </span>
            <span className="flex items-center gap-1">
              <Layers className="w-4 h-4" />
              {project.model}
            </span>
          </div>

          {/* Accuracy if available */}
          {project.accuracy && (
            <div className="mt-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400 font-semibold">
                {(project.accuracy * 100).toFixed(1)}% accuracy
              </span>
            </div>
          )}
        </div>

        {/* Hover glow effect */}
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-32 bg-indigo-500/30 rounded-full blur-3xl" />
        </div>
      </div>
    </motion.div>
  );
};

// Animated status badge
const StatusBadge: React.FC<{ status: Project['status'] }> = ({ status }) => {
  const configs = {
    training: {
      color: 'from-yellow-500 to-orange-500',
      icon: <Loader className="w-3 h-3" />,
      animate: true
    },
    completed: {
      color: 'from-green-500 to-emerald-500',
      icon: <CheckCircle className="w-3 h-3" />,
      animate: false
    },
    failed: {
      color: 'from-red-500 to-pink-500',
      icon: <AlertCircle className="w-3 h-3" />,
      animate: false
    },
    paused: {
      color: 'from-gray-500 to-gray-600',
      icon: <Pause className="w-3 h-3" />,
      animate: false
    }
  };

  const config = configs[status];

  return (
    <div className={`px-3 py-1 rounded-full bg-gradient-to-r ${config.color} text-white text-xs font-semibold flex items-center gap-1`}>
      {config.animate ? (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        >
          {config.icon}
        </motion.div>
      ) : config.icon}
      {status}
    </div>
  );
};
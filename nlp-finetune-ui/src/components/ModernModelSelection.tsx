// src/components/ModernModelSelection.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './GlassCard';
import {
  FileText,
  Tag,
  HelpCircle,
  AlignLeft,
  CheckCircle,
  Sparkles,
  Zap,
  Clock,
  Target
} from 'lucide-react';

interface Task {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

interface Model {
  id: string;
  name: string;
  params: string;
  speed: 'Very Fast' | 'Fast' | 'Medium' | 'Slow';
  accuracy: 'Good' | 'High' | 'Very High' | 'Excellent';
  recommended: boolean;
  description: string;
}

const tasks: Task[] = [
  {
    id: 'classification',
    name: 'Text Classification',
    description: 'Categorize text into predefined classes',
    icon: <FileText className="w-6 h-6" />
  },
  {
    id: 'ner',
    name: 'Named Entity Recognition',
    description: 'Extract entities like names, locations, dates',
    icon: <Tag className="w-6 h-6" />
  },
  {
    id: 'qa',
    name: 'Question Answering',
    description: 'Answer questions based on context',
    icon: <HelpCircle className="w-6 h-6" />
  },
  {
    id: 'summarization',
    name: 'Text Summarization',
    description: 'Generate concise summaries of long text',
    icon: <AlignLeft className="w-6 h-6" />
  }
];

const models: Model[] = [
  {
    id: 'bert-base',
    name: 'BERT Base',
    params: '110M',
    speed: 'Fast',
    accuracy: 'High',
    recommended: true,
    description: 'Balanced performance and efficiency for most tasks'
  },
  {
    id: 'roberta-large',
    name: 'RoBERTa Large',
    params: '355M',
    speed: 'Medium',
    accuracy: 'Very High',
    recommended: false,
    description: 'Superior accuracy for complex tasks'
  },
  {
    id: 'distilbert',
    name: 'DistilBERT',
    params: '66M',
    speed: 'Very Fast',
    accuracy: 'Good',
    recommended: false,
    description: 'Lightweight model for fast inference'
  }
];

export const ModernModelSelection: React.FC = () => {
  const [selectedTask, setSelectedTask] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-1/4 left-1/3 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.2, 0.4, 0.2],
          }}
          transition={{ duration: 12, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-1/4 right-1/3 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1.3, 1, 1.3],
            opacity: [0.4, 0.2, 0.4],
          }}
          transition={{ duration: 15, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center"
        >
          <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-gray-300 to-gray-500">
            Choose Your Model
          </h1>
          <p className="text-gray-400 text-xl max-w-2xl mx-auto">
            AI-powered recommendations based on your dataset and task requirements
          </p>
        </motion.div>

        {/* Task Selection */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-16"
        >
          <div className="flex items-center gap-3 mb-8">
            <Target className="w-6 h-6 text-indigo-400" />
            <h2 className="text-2xl font-bold">Select Your Task</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {tasks.map((task, index) => (
              <TaskCard
                key={task.id}
                task={task}
                selected={selectedTask === task.id}
                onClick={() => setSelectedTask(task.id)}
                index={index}
              />
            ))}
          </div>
        </motion.section>

        {/* Model Recommendations */}
        <AnimatePresence>
          {selectedTask && (
            <motion.section
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              transition={{ delay: 0.3 }}
            >
              <div className="flex items-center gap-3 mb-8">
                <Sparkles className="w-6 h-6 text-indigo-400" />
                <h2 className="text-2xl font-bold">Recommended Models</h2>
                <div className="px-3 py-1 bg-indigo-500/20 border border-indigo-500/30 rounded-full text-sm text-indigo-300">
                  AI Powered
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {models.map((model, index) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    selected={selectedModel === model.id}
                    onClick={() => setSelectedModel(model.id)}
                    index={index}
                  />
                ))}
              </div>

              {/* Continue Button */}
              {selectedModel && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-12 text-center"
                >
                  <button className="group relative px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl font-semibold text-lg overflow-hidden">
                    <span className="relative z-10 flex items-center gap-2">
                      Continue to Training Setup
                      <Zap className="w-5 h-5" />
                    </span>
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                  </button>
                </motion.div>
              )}
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Task Selection Card
const TaskCard: React.FC<{
  task: Task;
  selected: boolean;
  onClick: () => void;
  index: number;
}> = ({ task, selected, onClick, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={{ scale: 1.05, y: -5 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className="group cursor-pointer"
    >
      <div className={`relative backdrop-blur-xl rounded-2xl p-6 border-2 transition-all duration-300 ${
        selected
          ? 'border-indigo-500 bg-indigo-500/10 shadow-[0_0_30px_rgba(99,102,241,0.3)]'
          : 'border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/8'
      }`}>
        {/* Selection indicator */}
        {selected && (
          <motion.div
            layoutId="task-selected"
            className="absolute top-4 right-4"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
          >
            <CheckCircle className="w-6 h-6 text-indigo-400" />
          </motion.div>
        )}

        {/* Icon */}
        <div className={`w-14 h-14 rounded-xl mb-4 flex items-center justify-center transition-colors ${
          selected ? 'bg-indigo-500' : 'bg-white/10 group-hover:bg-white/15'
        }`}>
          {task.icon}
        </div>

        {/* Content */}
        <h3 className="text-xl font-semibold mb-2">{task.name}</h3>
        <p className="text-gray-400 text-sm leading-relaxed">{task.description}</p>

        {/* Hover glow */}
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-indigo-500/20 rounded-full blur-2xl" />
        </div>
      </div>
    </motion.div>
  );
};

// Model Card Component
const ModelCard: React.FC<{
  model: Model;
  selected: boolean;
  onClick: () => void;
  index: number;
}> = ({ model, selected, onClick, index }) => {
  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'Very Fast': return 'from-green-500 to-emerald-500';
      case 'Fast': return 'from-blue-500 to-cyan-500';
      case 'Medium': return 'from-yellow-500 to-orange-500';
      case 'Slow': return 'from-red-500 to-pink-500';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  const getAccuracyColor = (accuracy: string) => {
    switch (accuracy) {
      case 'Excellent': return 'from-purple-500 to-indigo-500';
      case 'Very High': return 'from-green-500 to-emerald-500';
      case 'High': return 'from-blue-500 to-cyan-500';
      case 'Good': return 'from-yellow-500 to-orange-500';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.15 }}
      whileHover={{ y: -10 }}
      className="group relative"
    >
      {/* Recommended badge */}
      {model.recommended && (
        <motion.div
          initial={{ scale: 0, rotate: -10 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ delay: index * 0.15 + 0.3, type: 'spring' }}
          className="absolute -top-3 -right-3 z-10 px-3 py-1 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full text-xs font-bold text-black shadow-lg"
        >
          ⭐ Recommended
        </motion.div>
      )}

      {/* Card */}
      <div className={`relative backdrop-blur-xl rounded-2xl p-6 h-full overflow-hidden border-2 transition-all duration-300 ${
        selected
          ? 'border-indigo-500 bg-indigo-500/10 shadow-[0_0_30px_rgba(99,102,241,0.3)]'
          : 'border-white/10 bg-white/5 group-hover:border-indigo-500/50 group-hover:bg-white/8'
      }`}>
        {/* Hover glow */}
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-40 h-40 bg-indigo-500/20 rounded-full blur-3xl" />
        </div>

        {/* Content */}
        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-2xl font-bold mb-1">{model.name}</h3>
              <p className="text-gray-400 text-sm">{model.params} parameters</p>
            </div>
            {selected && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring' }}
              >
                <CheckCircle className="w-6 h-6 text-indigo-400" />
              </motion.div>
            )}
          </div>

          <p className="text-gray-300 text-sm mb-6 leading-relaxed">
            {model.description}
          </p>

          {/* Specs */}
          <div className="space-y-4 mb-6">
            <SpecRow 
              label="Speed" 
              value={model.speed} 
              gradient={getSpeedColor(model.speed)}
              icon={<Clock className="w-4 h-4" />}
            />
            <SpecRow 
              label="Accuracy" 
              value={model.accuracy} 
              gradient={getAccuracyColor(model.accuracy)}
              icon={<Target className="w-4 h-4" />}
            />
          </div>

          {/* Select button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClick}
            className={`w-full py-3 rounded-xl font-semibold transition-all duration-300 ${
              selected
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 shadow-lg shadow-indigo-500/50'
                : 'bg-white/10 hover:bg-white/15 border border-white/20'
            }`}
          >
            {selected ? 'Selected' : 'Select Model'}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

// Spec Row Component
const SpecRow: React.FC<{
  label: string;
  value: string;
  gradient: string;
  icon: React.ReactNode;
}> = ({ label, value, gradient, icon }) => {
  return (
    <div>
      <div className="flex justify-between items-center text-sm mb-2">
        <span className="flex items-center gap-2 text-gray-400">
          <span className="text-indigo-400">{icon}</span>
          {label}
        </span>
        <span className="font-semibold">{value}</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: '80%' }}
          transition={{ duration: 1, delay: 0.5 }}
          className={`h-full bg-gradient-to-r ${gradient} rounded-full relative`}
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
            animate={{ x: ['-100%', '100%'] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          />
        </motion.div>
      </div>
    </div>
  );
};
// src/components/ModernTrainingDashboard.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import { GlassCard } from './GlassCard';
import {
  Play,
  Pause,
  Square,
  Settings,
  Brain,
  Cpu,
  HardDrive,
  Zap,
  TrendingUp,
  Clock,
  Layers
} from 'lucide-react';

interface TrainingState {
  status: 'running' | 'paused' | 'stopped';
  epoch: number;
  totalEpochs: number;
  batch: number;
  totalBatches: number;
  elapsedTime: number;
  estimatedRemaining: number;
}

interface Metrics {
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  learningRate: number;
  gradNorm: number;
}

export const ModernTrainingDashboard: React.FC = () => {
  const [trainingState, setTrainingState] = useState<TrainingState>({
    status: 'running',
    epoch: 15,
    totalEpochs: 50,
    batch: 234,
    totalBatches: 500,
    elapsedTime: 1425, // seconds
    estimatedRemaining: 3150
  });

  const [metrics, setMetrics] = useState<Metrics>({
    trainLoss: 0.234,
    valLoss: 0.312,
    trainAcc: 0.89,
    valAcc: 0.85,
    learningRate: 2e-5,
    gradNorm: 1.23
  });

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Animated grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px] [mask-image:radial-gradient(ellipse_at_center,black_50%,transparent_100%)]" />

      <div className="relative z-10">
        {/* Header */}
        <TrainingHeader 
          state={trainingState}
          onPause={() => setTrainingState(prev => ({ ...prev, status: 'paused' }))}
          onResume={() => setTrainingState(prev => ({ ...prev, status: 'running' }))}
          onStop={() => setTrainingState(prev => ({ ...prev, status: 'stopped' }))}
        />

        {/* Main Layout */}
        <div className="container mx-auto px-6 py-6">
          <div className="grid grid-cols-12 gap-6">
            {/* Left Sidebar */}
            <div className="col-span-3 space-y-6">
              <OverviewCard state={trainingState} />
              <QuickMetricsCard metrics={metrics} />
              <ProgressRing progress={(trainingState.epoch / trainingState.totalEpochs) * 100} />
              <ResourceMonitor />
            </div>

            {/* Main Content */}
            <div className="col-span-6 space-y-6">
              <LossCurveChart />
              <AccuracyMetricsChart />
            </div>

            {/* Right Sidebar */}
            <div className="col-span-3 space-y-6">
              <TrainingControlsPanel />
              <AIAssistantCard />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Training Header Component
const TrainingHeader: React.FC<{
  state: TrainingState;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}> = ({ state, onPause, onResume, onStop }) => {
  return (
    <div className="backdrop-blur-xl bg-white/5 border-b border-white/10 px-6 py-4">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">BERT Sentiment Analysis</h1>
          <StatusIndicator status={state.status} />
        </div>
        
        <div className="flex items-center gap-3">
          {state.status === 'running' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onPause}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-500/20 border border-yellow-500/30 rounded-lg text-yellow-400 hover:bg-yellow-500/30 transition-colors"
            >
              <Pause className="w-4 h-4" />
              Pause
            </motion.button>
          )}
          
          {state.status === 'paused' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onResume}
              className="flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-lg text-green-400 hover:bg-green-500/30 transition-colors"
            >
              <Play className="w-4 h-4" />
              Resume
            </motion.button>
          )}
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onStop}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 hover:bg-red-500/30 transition-colors"
          >
            <Square className="w-4 h-4" />
            Stop
          </motion.button>
        </div>
      </div>
    </div>
  );
};

// Status Indicator
const StatusIndicator: React.FC<{ status: TrainingState['status'] }> = ({ status }) => {
  const configs = {
    running: { color: 'bg-green-500', text: 'Training', pulse: true },
    paused: { color: 'bg-yellow-500', text: 'Paused', pulse: false },
    stopped: { color: 'bg-red-500', text: 'Stopped', pulse: false }
  };
  
  const config = configs[status];
  
  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`} />
      <span className="text-sm font-medium text-gray-300">{config.text}</span>
    </div>
  );
};

// Overview Card
const OverviewCard: React.FC<{ state: TrainingState }> = ({ state }) => {
  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <GlassCard>
      <h3 className="text-sm font-medium text-gray-400 mb-4">Training Overview</h3>
      
      <div className="space-y-4">
        <MetricRow 
          label="Current Epoch" 
          value={`${state.epoch}/${state.totalEpochs}`}
          icon={<Layers className="w-4 h-4" />}
        />
        <MetricRow 
          label="Time Elapsed" 
          value={formatTime(state.elapsedTime)}
          icon={<Clock className="w-4 h-4" />}
        />
        <MetricRow 
          label="Est. Remaining" 
          value={formatTime(state.estimatedRemaining)}
          icon={<TrendingUp className="w-4 h-4" />}
        />
      </div>
    </GlassCard>
  );
};

// Metric Row Component
const MetricRow: React.FC<{ label: string; value: string; icon: React.ReactNode }> = ({ label, value, icon }) => {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-2 text-gray-400 text-sm">
        <span className="text-indigo-400">{icon}</span>
        {label}
      </span>
      <motion.span 
        key={value}
        initial={{ scale: 1.2, color: '#6366f1' }}
        animate={{ scale: 1, color: '#ffffff' }}
        className="font-semibold"
      >
        {value}
      </motion.span>
    </div>
  );
};

// Quick Metrics Card
const QuickMetricsCard: React.FC<{ metrics: Metrics }> = ({ metrics }) => {
  const metricsList = [
    { label: 'Train Loss', value: metrics.trainLoss, target: 0.1, color: 'indigo' },
    { label: 'Val Loss', value: metrics.valLoss, target: 0.15, color: 'purple' },
    { label: 'Accuracy', value: metrics.trainAcc, target: 1.0, color: 'green' },
    { label: 'Learning Rate', value: metrics.learningRate, target: 1e-4, color: 'blue' },
  ];

  return (
    <GlassCard>
      <h3 className="text-sm font-medium text-gray-400 mb-4">Live Metrics</h3>
      
      <div className="space-y-4">
        {metricsList.map((metric, index) => (
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-400">{metric.label}</span>
              <span className="text-lg font-bold">
                {metric.label === 'Learning Rate' ? metric.value.toExponential(2) : metric.value.toFixed(3)}
              </span>
            </div>
            
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((metric.value / metric.target) * 100, 100)}%` }}
                transition={{ duration: 1, delay: index * 0.1 + 0.2 }}
                className={`h-full bg-gradient-to-r from-${metric.color}-500 to-${metric.color}-600 rounded-full relative`}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
              </motion.div>
            </div>
          </motion.div>
        ))}
      </div>
    </GlassCard>
  );
};

// Circular Progress Ring
const ProgressRing: React.FC<{ progress: number }> = ({ progress }) => {
  return (
    <GlassCard className="flex flex-col items-center">
      <h3 className="text-sm font-medium text-gray-400 mb-4 self-start">
        Overall Progress
      </h3>
      
      <div className="relative w-40 h-40">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="80"
            cy="80"
            r="70"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="12"
            fill="none"
          />
          <motion.circle
            cx="80"
            cy="80"
            r="70"
            stroke="url(#gradient)"
            strokeWidth="12"
            fill="none"
            strokeLinecap="round"
            initial={{ strokeDasharray: '0 440' }}
            animate={{ strokeDasharray: `${(progress / 100) * 440} 440` }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
        </svg>
        
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span 
            className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5, type: 'spring' }}
          >
            {Math.round(progress)}%
          </motion.span>
          <span className="text-sm text-gray-400">Complete</span>
        </div>
      </div>
    </GlassCard>
  );
};

// Resource Monitor
const ResourceMonitor: React.FC = () => {
  const resources = [
    { label: 'CPU', value: 45, color: 'blue', icon: <Cpu className="w-4 h-4" /> },
    { label: 'Memory', value: 67, color: 'green', icon: <HardDrive className="w-4 h-4" /> },
    { label: 'GPU', value: 89, color: 'purple', icon: <Zap className="w-4 h-4" /> },
  ];

  return (
    <GlassCard>
      <h3 className="text-sm font-medium text-gray-400 mb-4">System Resources</h3>
      
      <div className="space-y-4">
        {resources.map((resource) => (
          <ResourceGauge key={resource.label} {...resource} />
        ))}
      </div>
    </GlassCard>
  );
};

// Resource Gauge Component
const ResourceGauge: React.FC<{ 
  label: string; 
  value: number; 
  color: string; 
  icon: React.ReactNode;
}> = ({ label, value, color, icon }) => {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="flex items-center gap-2 text-sm text-gray-400">
          <span className="text-indigo-400">{icon}</span>
          {label}
        </span>
        <span className="text-sm font-semibold">{value}%</span>
      </div>
      <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          className={`h-full bg-gradient-to-r from-${color}-500 to-${color}-600 rounded-full`}
        />
        {value > 85 && (
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1, repeat: Infinity }}
            className="absolute inset-0 bg-red-500/20"
          />
        )}
      </div>
    </div>
  );
};

// Loss Curve Chart
const LossCurveChart: React.FC = () => {
  const chartData = {
    labels: Array.from({ length: 15 }, (_, i) => i + 1),
    datasets: [
      {
        label: 'Training Loss',
        data: [0.8, 0.65, 0.52, 0.45, 0.38, 0.34, 0.31, 0.28, 0.26, 0.25, 0.24, 0.235, 0.234, 0.233, 0.234],
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
      },
      {
        label: 'Validation Loss',
        data: [0.85, 0.7, 0.58, 0.5, 0.44, 0.4, 0.37, 0.35, 0.33, 0.32, 0.315, 0.313, 0.312, 0.311, 0.312],
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: { size: 12 },
          usePointStyle: true,
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
          font: { size: 11 },
        },
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
          font: { size: 11 },
        },
      },
    },
  };

  return (
    <GlassCard className="h-80">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Loss Curves</h3>
      </div>
      <div className="h-64">
        <Line data={chartData} options={chartOptions} />
      </div>
    </GlassCard>
  );
};

// Accuracy Metrics Chart
const AccuracyMetricsChart: React.FC = () => {
  const chartData = {
    labels: Array.from({ length: 15 }, (_, i) => i + 1),
    datasets: [
      {
        label: 'Training Accuracy',
        data: [0.6, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.885, 0.89, 0.892, 0.894, 0.895, 0.896, 0.897, 0.89],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
      },
      {
        label: 'Validation Accuracy',
        data: [0.58, 0.68, 0.74, 0.78, 0.81, 0.83, 0.84, 0.845, 0.848, 0.85, 0.851, 0.852, 0.851, 0.85, 0.85],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
      },
    ],
  };

  return (
    <GlassCard className="h-80">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Accuracy Metrics</h3>
      </div>
      <div className="h-64">
        <Line data={chartData} options={chartOptions} />
      </div>
    </GlassCard>
  );
};

// Training Controls Panel
const TrainingControlsPanel: React.FC = () => {
  return (
    <GlassCard>
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Settings className="w-5 h-5" />
        Training Controls
      </h3>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-400 mb-2 block">
            Learning Rate: <span className="text-white font-mono">2.00e-05</span>
          </label>
          <div className="h-2 bg-white/5 rounded-full">
            <div className="h-full w-1/3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full" />
          </div>
        </div>
        
        <div>
          <label className="text-sm text-gray-400 mb-2 block">
            Dropout: <span className="text-white font-mono">0.10</span>
          </label>
          <div className="h-2 bg-white/5 rounded-full">
            <div className="h-full w-1/5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full" />
          </div>
        </div>
      </div>
    </GlassCard>
  );
};

// AI Assistant Card
const AIAssistantCard: React.FC = () => {
  return (
    <GlassCard>
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5" />
        AI Assistant
      </h3>
      
      <div className="space-y-3">
        <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 bg-green-500 rounded-full" />
            <span className="text-sm font-medium text-green-400">Healthy Training</span>
          </div>
          <p className="text-sm text-gray-300">
            Training is progressing well. Validation loss is decreasing steadily.
          </p>
        </div>
        
        <div className="text-sm text-gray-400">
          Next analysis in 3 epochs
        </div>
      </div>
    </GlassCard>
  );
};
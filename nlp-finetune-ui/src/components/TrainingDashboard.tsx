import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Zap, Target, Clock, Download, CheckCircle } from 'lucide-react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface TrainingDashboardProps {
  config: any
  onNext: (data: any) => void
  onBack?: () => void
  canGoBack?: boolean
  canGoForward?: boolean
}

export default function TrainingDashboard({ config, onNext, onBack, canGoBack, canGoForward, showBackArrow, showForwardArrow }: TrainingDashboardProps) {
  const [status, setStatus] = useState<any>({})
  const [metrics, setMetrics] = useState<any[]>([])
  const [ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const websocket = new WebSocket(`ws://localhost:8000/ws/${config.sessionId}`)
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setStatus(data)
      
      if (data.loss && data.accuracy) {
        setMetrics(prev => [...prev, {
          step: prev.length + 1,
          loss: data.loss,
          accuracy: data.accuracy,
          epoch: data.epoch
        }])
      }
      
      if (data.status === 'completed') {
        setTimeout(() => onNext({ sessionId: config.sessionId }), 2000)
      }
    }
    
    setWs(websocket)
    
    return () => websocket.close()
  }, [config.sessionId])

  const chartData = {
    labels: metrics.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Loss',
        data: metrics.map(m => m.loss),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        yAxisID: 'y',
      },
      {
        label: 'Accuracy',
        data: metrics.map(m => m.accuracy),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        yAxisID: 'y1',
      },
    ],
  }

  const chartOptions = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        labels: {
          color: 'white'
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: 'white' }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: 'white' }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        grid: { drawOnChartArea: false },
        ticks: { color: 'white' }
      },
    },
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="max-w-7xl mx-auto relative">
      {/* Navigation Arrows */}
      {showBackArrow && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={onBack}
          className="absolute left-4 bottom-8 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
        >
          <ArrowLeft className="w-6 h-6 text-white" />
        </motion.button>
      )}

      {showForwardArrow && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => onNext({})}
          className={`absolute right-0 bottom-8 p-3 rounded-full transition-colors z-10 ${
            canGoForward ? 'bg-white/10 hover:bg-white/20' : 'bg-gray-600/50 cursor-not-allowed'
          }`}
          disabled={!canGoForward}
        >
          <ArrowRight className="w-6 h-6 text-white" />
        </motion.button>
      )}

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-12"
      >
        <div className="flex items-center justify-center mb-6">
          <div className="flex items-center">
            <Sparkles className="w-8 h-8 text-purple-400 mr-3" />
            <h1 className="text-4xl font-bold gradient-text">
              Training in Progress
            </h1>
          </div>
        </div>
        <p className="text-xl text-gray-300 text-center">
          {config.model_id} • {config.dataset_info.task_type}
        </p>
      </motion.div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass p-6 text-center"
        >
          <Activity className="w-8 h-8 text-blue-400 mx-auto mb-3" />
          <div className="text-2xl font-bold text-white">
            {status.progress?.toFixed(1) || 0}%
          </div>
          <div className="text-sm text-gray-400">Progress</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="glass p-6 text-center"
        >
          <Zap className="w-8 h-8 text-yellow-400 mx-auto mb-3" />
          <div className="text-2xl font-bold text-white">
            {status.epoch || 0}
          </div>
          <div className="text-sm text-gray-400">Epoch</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass p-6 text-center"
        >
          <Target className="w-8 h-8 text-green-400 mx-auto mb-3" />
          <div className="text-2xl font-bold text-white">
            {(status.accuracy * 100)?.toFixed(1) || 0}%
          </div>
          <div className="text-sm text-gray-400">Accuracy</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="glass p-6 text-center"
        >
          <Clock className="w-8 h-8 text-purple-400 mx-auto mb-3" />
          <div className="text-2xl font-bold text-white">
            {status.step || 0}
          </div>
          <div className="text-sm text-gray-400">Steps</div>
        </motion.div>
      </div>

      {/* Progress Bar */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass p-6 mb-8"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Training Progress</h3>
          <span className={`px-3 py-1 rounded-full text-sm ${
            status.status === 'completed' ? 'bg-green-500/20 text-green-400' :
            status.status === 'training' ? 'bg-blue-500/20 text-blue-400' :
            'bg-gray-500/20 text-gray-400'
          }`}>
            {status.status || 'Starting...'}
          </span>
        </div>
        
        <div className="w-full bg-gray-700 rounded-full h-3 mb-4">
          <motion.div
            className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${status.progress || 0}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        
        <div className="flex justify-between text-sm text-gray-400">
          <span>Loss: {status.loss?.toFixed(4) || 'N/A'}</span>
          <span>Accuracy: {(status.accuracy * 100)?.toFixed(2) || 0}%</span>
        </div>
      </motion.div>

      {/* Live Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass p-6 mb-8"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Live Metrics</h3>
        <div className="h-80">
          {metrics.length > 0 ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              Waiting for training data...
            </div>
          )}
        </div>
      </motion.div>

      {/* Completion Message */}
      {status.status === 'completed' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass p-8 text-center"
        >
          <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-white mb-2">Training Complete!</h3>
          <p className="text-gray-300 mb-6">
            Your model has been successfully trained and is ready for export.
          </p>
          <div className="text-sm text-gray-400">
            Redirecting to results in a moment...
          </div>
        </motion.div>
      )}
    </div>
  )
}
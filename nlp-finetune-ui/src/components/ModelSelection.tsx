import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Zap, Settings, Play, Sparkles, ArrowLeft, ArrowRight } from 'lucide-react'

interface ModelSelectionProps {
  datasetInfo: any
  onNext: (data: any) => void
  onBack?: () => void
  canGoBack?: boolean
  canGoForward?: boolean
  showBackArrow?: boolean
  showForwardArrow?: boolean
}

export default function ModelSelection({ datasetInfo, onNext, onBack, canGoBack, canGoForward, showBackArrow, showForwardArrow }: ModelSelectionProps) {
  const [models, setModels] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState<any>(null)
  const [params, setParams] = useState<any>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_type: datasetInfo.task_type,
          dataset_size: datasetInfo.rows
        })
      })
      
      const data = await response.json()
      setModels(data.models)
      setSelectedModel(data.models[0])
      setParams(data.models[0].params)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleParamChange = (key: string, value: any) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const startTraining = async () => {
    const config = {
      model_id: selectedModel.id,
      dataset_info: datasetInfo,
      parameters: Object.fromEntries(
        Object.entries(params).map(([key, param]: [string, any]) => [
          key, param.value || param.default
        ])
      )
    }

    try {
      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      const result = await response.json()
      onNext({ 
        trainingConfig: config,
        sessionId: result.session_id 
      })
    } catch (error) {
      console.error('Failed to start training:', error)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-300">AI is selecting the best models for your data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto relative">
      {/* Navigation Arrows */}
      {showBackArrow && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={onBack}
          className="absolute left-0 top-1/2 -translate-y-1/2 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
        >
          <ArrowLeft className="w-6 h-6 text-white" />
        </motion.button>
      )}

      {showForwardArrow && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => onNext({})}
          className={`absolute right-0 top-1/2 -translate-y-1/2 p-3 rounded-full transition-colors z-10 ${
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
        <div className="relative flex items-center justify-center mb-6">
          {canGoBack && (
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={onBack}
              className="absolute left-0 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
            >
              <ArrowLeft className="w-6 h-6 text-white" />
            </motion.button>
          )}
          <div className="flex items-center">
            <Sparkles className="w-8 h-8 text-purple-400 mr-3" />
            <h1 className="text-4xl font-bold gradient-text">
              AI Model Recommendations
            </h1>
          </div>
          {canGoForward && (
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => onNext({ selectedModel })}
              className="absolute right-0 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
            >
              <ArrowRight className="w-6 h-6 text-white" />
            </motion.button>
          )}
        </div>
        <p className="text-xl text-gray-300 text-center">
          Based on your {datasetInfo.task_type} task with {datasetInfo.rows.toLocaleString()} samples
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Model Selection */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-4"
        >
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2" />
            Choose Your Model
          </h3>
          
          {models.map((model, i) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`glass p-6 cursor-pointer transition-all duration-300 ${
                selectedModel?.id === model.id 
                  ? 'ring-2 ring-purple-500 bg-purple-500/10' 
                  : 'hover:bg-white/5'
              }`}
              onClick={() => {
                setSelectedModel(model)
                setParams(model.params)
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-white">{model.name}</h4>
                {i === 0 && (
                  <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded-full text-xs">
                    Recommended
                  </span>
                )}
              </div>
              
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center">
                  <Zap className="w-4 h-4 text-yellow-400 mr-1" />
                  <span className="text-gray-400">Speed: {model.speed}</span>
                </div>
                <div className="flex items-center">
                  <Brain className="w-4 h-4 text-blue-400 mr-1" />
                  <span className="text-gray-400">Accuracy: {model.accuracy}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Parameters */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass p-6"
        >
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Training Parameters
          </h3>
          
          <div className="space-y-6">
            {selectedModel && Object.entries(selectedModel.params).map(([key, param]: [string, any]) => (
              <div key={key}>
                <label className="block text-sm font-medium text-gray-300 mb-2 capitalize">
                  {key.replace('_', ' ')}
                </label>
                
                {param.type === 'float' ? (
                  <div className="space-y-2">
                    <input
                      type="range"
                      min={param.min}
                      max={param.max}
                      step={(param.max - param.min) / 100}
                      defaultValue={param.default}
                      onChange={(e) => handleParamChange(key, { ...param, value: parseFloat(e.target.value) })}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{param.min}</span>
                      <span className="text-white">{params[key]?.value || param.default}</span>
                      <span>{param.max}</span>
                    </div>
                  </div>
                ) : param.type === 'int' && param.options ? (
                  <select
                    defaultValue={param.default}
                    onChange={(e) => handleParamChange(key, { ...param, value: parseInt(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white"
                  >
                    {param.options.map((option: number) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    min={param.min}
                    max={param.max}
                    defaultValue={param.default}
                    onChange={(e) => handleParamChange(key, { ...param, value: parseInt(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white"
                  />
                )}
              </div>
            ))}
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={startTraining}
            className="btn-primary w-full mt-8 flex items-center justify-center text-lg py-4"
          >
            <Play className="w-5 h-5 mr-2" />
            Start Training
          </motion.button>
        </motion.div>
      </div>
    </div>
  )
}
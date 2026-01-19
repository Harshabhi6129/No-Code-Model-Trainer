import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Download, FileText, Code, BarChart3, CheckCircle, RefreshCw, ArrowLeft, ArrowRight, Sparkles } from 'lucide-react'

interface ExportResultsProps {
  sessionId: string
  onBack?: () => void
  canGoBack?: boolean
  canGoForward?: boolean
  showBackArrow?: boolean
  showForwardArrow?: boolean
}

export default function ExportResults({ sessionId, onBack, canGoBack, canGoForward, showBackArrow, showForwardArrow }: ExportResultsProps) {
  const [exportData, setExportData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchExportData()
  }, [sessionId])

  const fetchExportData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/export/${sessionId}`)
      const data = await response.json()
      setExportData(data)
    } catch (error) {
      console.error('Failed to fetch export data:', error)
    } finally {
      setLoading(false)
    }
  }

  const downloadModel = () => {
    // Create downloadable content
    const modelData = {
      ...exportData,
      inference_code: `
# Generated inference code for your trained model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your trained model
tokenizer = AutoTokenizer.from_pretrained("${exportData?.config?.model_id}")
model = AutoModelForSequenceClassification.from_pretrained("./model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.numpy()

# Example usage
result = predict("Your text here")
print(f"Prediction: {result}")
      `
    }
    
    const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `model_${sessionId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const restartTraining = () => {
    window.location.reload()
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-300">Preparing your trained model...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto relative">
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
          onClick={() => {}}
          className={`absolute right-0 top-1/2 -translate-y-1/2 p-3 rounded-full transition-colors z-10 ${
            canGoForward ? 'bg-white/10 hover:bg-white/20' : 'bg-gray-600/50 cursor-not-allowed'
          }`}
          disabled={!canGoForward}
        >
          <ArrowRight className="w-6 h-6 text-white" />
        </motion.button>
      )}

      {/* Success Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-12"
      >
        <div className="flex items-center justify-center mb-6">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring" }}
            className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center"
          >
            <CheckCircle className="w-12 h-12 text-green-400" />
          </motion.div>
        </div>

        <h2 className="text-4xl font-bold gradient-text mb-4 text-center">
          Training Complete!
        </h2>
        <p className="text-xl text-gray-300 text-center">
          Your model is ready for deployment
        </p>
      </motion.div>

      {/* Results Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass p-8 mb-8"
      >
        <h3 className="text-2xl font-semibold text-white mb-6">Training Results</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">
              {(exportData?.final_metrics?.accuracy * 100)?.toFixed(1) || 'N/A'}%
            </div>
            <div className="text-gray-400">Final Accuracy</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {exportData?.final_metrics?.loss?.toFixed(4) || 'N/A'}
            </div>
            <div className="text-gray-400">Final Loss</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">
              {exportData?.final_metrics?.epochs || 'N/A'}
            </div>
            <div className="text-gray-400">Epochs Trained</div>
          </div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="font-semibold text-white mb-2">Model Configuration</h4>
          <div className="text-sm text-gray-400 space-y-1">
            <div>Model: {exportData?.config?.model_id}</div>
            <div>Task: {exportData?.config?.dataset_info?.task_type}</div>
            <div>Dataset: {exportData?.config?.dataset_info?.rows} samples</div>
          </div>
        </div>
      </motion.div>

      {/* Export Options */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8"
      >
        <div className="glass p-6">
          <div className="flex items-center mb-4">
            <Download className="w-6 h-6 text-blue-400 mr-3" />
            <h4 className="text-lg font-semibold text-white">Download Model</h4>
          </div>
          <p className="text-gray-400 mb-4">
            Get your trained model with weights, config, and inference code
          </p>
          <button
            onClick={downloadModel}
            className="btn-primary w-full flex items-center justify-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Package
          </button>
        </div>

        <div className="glass p-6">
          <div className="flex items-center mb-4">
            <Code className="w-6 h-6 text-green-400 mr-3" />
            <h4 className="text-lg font-semibold text-white">Inference Code</h4>
          </div>
          <p className="text-gray-400 mb-4">
            Ready-to-use Python code for making predictions
          </p>
          <button
            onClick={() => {
              const code = `# Your trained model inference code
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("${exportData?.config?.model_id}")
model = AutoModelForSequenceClassification.from_pretrained("./model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.numpy()

# Example usage
result = predict("Your text here")
print(f"Prediction: {result}")`
              
              navigator.clipboard.writeText(code)
              alert('Code copied to clipboard!')
            }}
            className="btn-secondary w-full flex items-center justify-center"
          >
            <Code className="w-4 h-4 mr-2" />
            Copy Code
          </button>
        </div>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="flex flex-col sm:flex-row gap-4 justify-center"
      >
        <button
          onClick={restartTraining}
          className="btn-secondary flex items-center justify-center px-8 py-3"
        >
          <RefreshCw className="w-5 h-5 mr-2" />
          Train Another Model
        </button>
        
        <button
          onClick={downloadModel}
          className="btn-primary flex items-center justify-center px-8 py-3"
        >
          <Download className="w-5 h-5 mr-2" />
          Download Complete Package
        </button>
      </motion.div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="text-center mt-12 text-gray-500"
      >
        <p>Model trained successfully • Ready for production deployment</p>
      </motion.div>
    </div>
  )
}
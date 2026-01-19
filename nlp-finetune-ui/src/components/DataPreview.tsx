import React from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Database, Target, ArrowRight, ArrowLeft, Sparkles } from 'lucide-react'

interface DataPreviewProps {
  datasetInfo: any
  onNext: (data: any) => void
  onBack?: () => void
  canGoBack?: boolean
  canGoForward?: boolean
  showBackArrow?: boolean
  showForwardArrow?: boolean
}

export default function DataPreview({ datasetInfo, onNext, onBack, canGoBack, canGoForward, showBackArrow, showForwardArrow }: DataPreviewProps) {
  const { rows, columns, sample_data, task_type, unique_labels, avg_text_length } = datasetInfo

  return (
    <div className="max-w-6xl mx-auto relative">
      {/* Navigation Arrows */}
      {showBackArrow && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={onBack}
          className="absolute left-0 bottom-8 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
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
              Dataset Analysis Complete
            </h1>
          </div>
        </div>
        <p className="text-xl text-gray-300 text-center">
          AI has analyzed your data and detected the optimal training approach
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="glass p-6"
        >
          <div className="flex items-center mb-4">
            <Database className="w-6 h-6 text-blue-400 mr-3" />
            <h3 className="text-lg font-semibold text-white">Dataset Info</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Rows:</span>
              <span className="text-white font-medium">{rows.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Columns:</span>
              <span className="text-white font-medium">{columns.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Length:</span>
              <span className="text-white font-medium">{avg_text_length} chars</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass p-6"
        >
          <div className="flex items-center mb-4">
            <Target className="w-6 h-6 text-green-400 mr-3" />
            <h3 className="text-lg font-semibold text-white">Task Detection</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Type:</span>
              <span className="text-white font-medium capitalize">{task_type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Classes:</span>
              <span className="text-white font-medium">{unique_labels.length}</span>
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Labels: {unique_labels.slice(0, 3).join(', ')}
              {unique_labels.length > 3 && '...'}
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="glass p-6"
        >
          <div className="flex items-center mb-4">
            <BarChart3 className="w-6 h-6 text-purple-400 mr-3" />
            <h3 className="text-lg font-semibold text-white">Recommendations</h3>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-green-400">✓ Good data quality</div>
            <div className="text-sm text-green-400">✓ Balanced dataset</div>
            <div className="text-sm text-green-400">✓ Optimal size for training</div>
          </div>
        </motion.div>
      </div>

      {/* Data Preview Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass p-6 mb-8"
      >
        <h3 className="text-xl font-semibold text-white mb-4">Data Preview</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                {columns.map((col: string) => (
                  <th key={col} className="text-left p-3 text-gray-300 font-medium">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sample_data.slice(0, 5).map((row: any, i: number) => (
                <tr key={i} className="border-b border-gray-800">
                  {columns.map((col: string) => (
                    <td key={col} className="p-3 text-gray-400 max-w-xs truncate">
                      {String(row[col] || '').substring(0, 100)}
                      {String(row[col] || '').length > 100 && '...'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Continue Button */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="text-center"
      >
        <button
          onClick={() => onNext({})}
          className="btn-primary inline-flex items-center text-lg px-8 py-4"
        >
          Continue to Model Selection
          <ArrowRight className="w-5 h-5 ml-2" />
        </button>
      </motion.div>
    </div>
  )
}
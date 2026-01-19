import React, { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, Sparkles, ArrowLeft, ArrowRight } from 'lucide-react'

interface UploadStepProps {
  onNext: (data: any) => void
  onBack?: () => void
  canGoBack?: boolean
  canGoForward?: boolean
}

export default function UploadStep({ onNext, onBack, canGoBack, canGoForward, showBackArrow, showForwardArrow }: UploadStepProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      await uploadFile(files[0])
    }
  }, [])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      await uploadFile(files[0])
    }
  }

  const uploadFile = async (file: File) => {
    setIsUploading(true)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      })
      
      const datasetInfo = await response.json()
      onNext({ datasetInfo })
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto text-center relative">
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
        transition={{ duration: 0.6 }}
        className="mb-12"
      >
        <div className="flex items-center justify-center mb-6">
          <div className="flex items-center">
            <Sparkles className="w-8 h-8 text-purple-400 mr-3" />
            <h1 className="text-4xl font-bold gradient-text">
              AI Model Training
            </h1>
          </div>
        </div>
        <p className="text-xl text-gray-300">
          Upload your dataset to get started with intelligent model training
        </p>
      </motion.div>

      {/* Upload Zone */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className={`glass p-12 transition-all duration-300 ${
          isDragging ? 'scale-105 pulse-glow' : ''
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
      >
        {isUploading ? (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-lg text-gray-300">Analyzing your dataset...</p>
          </div>
        ) : (
          <>
            <Upload className="w-16 h-16 text-purple-400 mx-auto mb-6 float" />
            <h3 className="text-2xl font-semibold text-white mb-4">
              Drop your CSV file here
            </h3>
            <p className="text-gray-400 mb-8">
              Or click to browse and select your dataset
            </p>
            
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="btn-primary inline-block cursor-pointer"
            >
              Choose File
            </label>
          </>
        )}
      </motion.div>

      {/* Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {[
          { icon: FileText, title: "Smart Analysis", desc: "AI-powered dataset insights" },
          { icon: Sparkles, title: "Auto Detection", desc: "Automatic task type detection" },
          { icon: Upload, title: "Quick Setup", desc: "One-click model recommendations" }
        ].map((feature, i) => (
          <div key={i} className="glass-dark p-6 text-center">
            <feature.icon className="w-8 h-8 text-purple-400 mx-auto mb-3" />
            <h4 className="font-semibold text-white mb-2">{feature.title}</h4>
            <p className="text-sm text-gray-400">{feature.desc}</p>
          </div>
        ))}
      </motion.div>
    </div>
  )
}
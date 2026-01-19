import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import UploadStep from './components/UploadStep'
import DataPreview from './components/DataPreview'
import ModelSelection from './components/ModelSelection'
import TrainingDashboard from './components/TrainingDashboard'
import ExportResults from './components/ExportResults'
import './App.css'

type Step = 'upload' | 'preview' | 'model' | 'training' | 'export'

interface AppState {
  step: Step
  datasetInfo: any
  selectedModel: any
  trainingConfig: any
  sessionId: string
}

export default function App() {
  const [direction, setDirection] = useState<'forward' | 'backward'>('forward')
  const [state, setState] = useState<AppState>({
    step: 'upload',
    datasetInfo: null,
    selectedModel: null,
    trainingConfig: null,
    sessionId: ''
  })

  const steps: Step[] = ['upload', 'preview', 'model', 'training', 'export']
  const currentIndex = steps.indexOf(state.step)
  const canGoBack = currentIndex > 0
  const canGoForward = currentIndex < steps.length - 1 && (
    (state.step === 'upload') ||
    (state.step === 'preview' && state.datasetInfo) ||
    (state.step === 'model' && state.selectedModel) ||
    (state.step === 'training' && state.trainingConfig)
  )

  const nextStep = (data: any) => {
    if (currentIndex < steps.length - 1) {
      setDirection('forward')
      setState(prev => ({
        ...prev,
        step: steps[currentIndex + 1],
        ...data
      }))
    }
  }

  const prevStep = () => {
    if (currentIndex > 0) {
      setDirection('backward')
      setState(prev => ({
        ...prev,
        step: steps[currentIndex - 1]
      }))
    }
  }

  const pageVariants = {
    initial: (direction: 'forward' | 'backward') => ({
      opacity: 0,
      x: direction === 'forward' ? 100 : -100
    }),
    in: { opacity: 1, x: 0 },
    out: (direction: 'forward' | 'backward') => ({
      opacity: 0,
      x: direction === 'forward' ? -100 : 100
    })
  }

  const pageTransition = {
    type: "tween",
    ease: "anticipate",
    duration: 0.5
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Progress Bar */}
      <div className="fixed top-0 left-0 right-0 z-50">
        <div className="h-1 bg-black/20">
          <motion.div 
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
            initial={{ width: "0%" }}
            animate={{ 
              width: state.step === 'upload' ? '20%' : 
                     state.step === 'preview' ? '40%' :
                     state.step === 'model' ? '60%' :
                     state.step === 'training' ? '80%' : '100%'
            }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Main Content */}
      <AnimatePresence mode="wait" custom={direction}>
        <motion.div
          key={state.step}
          initial="initial"
          animate="in"
          exit="out"
          variants={pageVariants}
          transition={pageTransition}
          custom={direction}
          className="min-h-screen flex items-center justify-center p-4"
        >
          {state.step === 'upload' && (
            <UploadStep
              onNext={nextStep}
              onBack={prevStep}
              canGoBack={canGoBack}
              canGoForward={canGoForward}
              showBackArrow={currentIndex > 0}
              showForwardArrow={currentIndex < steps.length - 1}
            />
          )}

          {state.step === 'preview' && (
            <DataPreview
              datasetInfo={state.datasetInfo}
              onNext={nextStep}
              onBack={prevStep}
              canGoBack={canGoBack}
              canGoForward={canGoForward}
              showBackArrow={currentIndex > 0}
              showForwardArrow={currentIndex < steps.length - 1}
            />
          )}

          {state.step === 'model' && (
            <ModelSelection
              datasetInfo={state.datasetInfo}
              onNext={nextStep}
              onBack={prevStep}
              canGoBack={canGoBack}
              canGoForward={canGoForward}
              showBackArrow={currentIndex > 0}
              showForwardArrow={currentIndex < steps.length - 1}
            />
          )}

          {state.step === 'training' && (
            <TrainingDashboard
              config={state.trainingConfig}
              onNext={nextStep}
              onBack={prevStep}
              canGoBack={canGoBack}
              canGoForward={canGoForward}
              showBackArrow={currentIndex > 0}
              showForwardArrow={currentIndex < steps.length - 1}
            />
          )}

          {state.step === 'export' && (
            <ExportResults
              sessionId={state.sessionId}
              onBack={prevStep}
              canGoBack={canGoBack}
              canGoForward={canGoForward}
              showBackArrow={currentIndex > 0}
              showForwardArrow={currentIndex < steps.length - 1}
            />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

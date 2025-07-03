import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
  ClockIcon,
  CpuChipIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Training = () => {
  const [trainingState, setTrainingState] = useState('idle'); // idle, running, paused, completed
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(15);
  const [trainingData, setTrainingData] = useState([]);
  const [config, setConfig] = useState({
    classification: {
      epochs: 15,
      batchSize: 32,
      learningRate: 0.001,
      model: 'MobileNetV3Small'
    },
    segmentation: {
      epochs: 20,
      batchSize: 32,
      learningRate: 0.001,
      model: 'U-Net EfficientNetB0'
    }
  });

  const [logs, setLogs] = useState([
    { time: '14:35:21', type: 'info', message: 'Training pipeline initialized' },
    { time: '14:35:22', type: 'info', message: 'Loading classification dataset...' },
    { time: '14:35:25', type: 'success', message: 'Dataset loaded: 8,247 samples' },
    { time: '14:35:26', type: 'info', message: 'Starting classification model training...' }
  ]);

  // Simulate training progress
  useEffect(() => {
    let interval;
    if (trainingState === 'running') {
      interval = setInterval(() => {
        setCurrentEpoch(prev => {
          if (prev >= totalEpochs) {
            setTrainingState('completed');
            return prev;
          }
          return prev + 0.1;
        });

        // Add training data point
        setTrainingData(prev => {
          const epoch = Math.floor(currentEpoch);
          const newPoint = {
            epoch: epoch,
            trainLoss: Math.max(0.1, 1.2 - epoch * 0.05 + Math.random() * 0.1),
            valLoss: Math.max(0.1, 1.3 - epoch * 0.04 + Math.random() * 0.1),
            trainAcc: Math.min(99, 30 + epoch * 4 + Math.random() * 2),
            valAcc: Math.min(99, 25 + epoch * 4 + Math.random() * 2)
          };
          return [...prev.slice(-50), newPoint];
        });

        // Add log messages
        if (Math.random() > 0.7) {
          const messages = [
            'Epoch completed, saving checkpoint...',
            'Learning rate adjusted',
            'Validation accuracy improved',
            'Processing batch...'
          ];
          const types = ['info', 'success', 'warning', 'info'];
          const randomIndex = Math.floor(Math.random() * messages.length);
          
          setLogs(prev => [{
            time: new Date().toLocaleTimeString(),
            type: types[randomIndex],
            message: messages[randomIndex]
          }, ...prev.slice(0, 49)]);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [trainingState, currentEpoch, totalEpochs]);

  const startTraining = () => {
    setTrainingState('running');
    setLogs(prev => [{
      time: new Date().toLocaleTimeString(),
      type: 'success',
      message: 'Training started'
    }, ...prev]);
  };

  const pauseTraining = () => {
    setTrainingState('paused');
    setLogs(prev => [{
      time: new Date().toLocaleTimeString(),
      type: 'warning',
      message: 'Training paused'
    }, ...prev]);
  };

  const stopTraining = () => {
    setTrainingState('idle');
    setCurrentEpoch(0);
    setTrainingData([]);
    setLogs(prev => [{
      time: new Date().toLocaleTimeString(),
      type: 'error',
      message: 'Training stopped'
    }, ...prev]);
  };

  const getLogColor = (type) => {
    switch (type) {
      case 'success': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-text-primary';
    }
  };

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-text-primary mb-2">Model Training</h1>
          <p className="text-text-muted">Train and monitor your food detection models</p>
        </div>
        <div className="flex items-center space-x-3">
          {trainingState === 'idle' && (
            <button onClick={startTraining} className="button-primary">
              <PlayIcon className="w-5 h-5 mr-2" />
              Start Training
            </button>
          )}
          {trainingState === 'running' && (
            <>
              <button onClick={pauseTraining} className="button-secondary">
                <PauseIcon className="w-5 h-5 mr-2" />
                Pause
              </button>
              <button onClick={stopTraining} className="button-secondary">
                <StopIcon className="w-5 h-5 mr-2" />
                Stop
              </button>
            </>
          )}
          {trainingState === 'paused' && (
            <>
              <button onClick={startTraining} className="button-primary">
                <PlayIcon className="w-5 h-5 mr-2" />
                Resume
              </button>
              <button onClick={stopTraining} className="button-secondary">
                <StopIcon className="w-5 h-5 mr-2" />
                Stop
              </button>
            </>
          )}
        </div>
      </div>

      {/* Training Progress */}
      <motion.div
        className="card"
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-text-primary">Training Progress</h2>
          <div className="flex items-center space-x-2">
            <div className={`status-indicator ${
              trainingState === 'running' ? 'status-processing' : 
              trainingState === 'completed' ? 'status-success' : 'status-warning'
            }`}></div>
            <span className="text-sm text-text-muted capitalize">{trainingState}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div className="text-center">
            <CpuChipIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {Math.floor(currentEpoch)}/{totalEpochs}
            </p>
            <p className="text-text-muted text-sm">Epochs</p>
          </div>
          <div className="text-center">
            <ClockIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {trainingState === 'running' ? 
                `${Math.floor((Date.now() - Date.now()) / 60000)}m` : '0m'}
            </p>
            <p className="text-text-muted text-sm">Elapsed Time</p>
          </div>
          <div className="text-center">
            <ChartBarIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {trainingData.length > 0 ? trainingData[trainingData.length - 1].trainLoss.toFixed(3) : '0.000'}
            </p>
            <p className="text-text-muted text-sm">Training Loss</p>
          </div>
          <div className="text-center">
            <AdjustmentsHorizontalIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {trainingData.length > 0 ? trainingData[trainingData.length - 1].valAcc.toFixed(1) : '0.0'}%
            </p>
            <p className="text-text-muted text-sm">Validation Acc</p>
          </div>
        </div>

        <div className="progress-bar h-3 mb-2">
          <motion.div
            className="progress-fill"
            initial={{ width: 0 }}
            animate={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="flex justify-between text-sm text-text-muted">
          <span>Progress: {((currentEpoch / totalEpochs) * 100).toFixed(1)}%</span>
          <span>ETA: {trainingState === 'running' ? 
            `${Math.floor((totalEpochs - currentEpoch) * 2)}m` : 'N/A'}</span>
        </div>
      </motion.div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Chart */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3 className="text-lg font-semibold text-text-primary mb-4">Training & Validation Loss</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f3f4f6'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="trainLoss" 
                stroke="#f59e0b" 
                strokeWidth={2}
                name="Training Loss"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="valLoss" 
                stroke="#f37316" 
                strokeWidth={2}
                name="Validation Loss"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Accuracy Chart */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold text-text-primary mb-4">Training & Validation Accuracy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} domain={[0, 100]} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f3f4f6'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="trainAcc" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Training Accuracy"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="valAcc" 
                stroke="#06d6a0" 
                strokeWidth={2}
                name="Validation Accuracy"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Configuration and Logs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration */}
        <motion.div
          className="card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-lg font-semibold text-text-primary mb-4">Training Configuration</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-text-primary mb-2">Classification Model</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Model</p>
                  <p className="text-text-primary font-medium">{config.classification.model}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Epochs</p>
                  <p className="text-text-primary font-medium">{config.classification.epochs}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Batch Size</p>
                  <p className="text-text-primary font-medium">{config.classification.batchSize}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Learning Rate</p>
                  <p className="text-text-primary font-medium">{config.classification.learningRate}</p>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-text-primary mb-2">Segmentation Model</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Model</p>
                  <p className="text-text-primary font-medium">{config.segmentation.model}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Epochs</p>
                  <p className="text-text-primary font-medium">{config.segmentation.epochs}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Batch Size</p>
                  <p className="text-text-primary font-medium">{config.segmentation.batchSize}</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted">Learning Rate</p>
                  <p className="text-text-primary font-medium">{config.segmentation.learningRate}</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Training Logs */}
        <motion.div
          className="card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center space-x-2 mb-4">
            <DocumentTextIcon className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-text-primary">Training Logs</h3>
          </div>
          <div className="bg-background-dark/50 rounded-lg p-4 h-80 overflow-y-auto scrollbar-hide">
            <div className="space-y-2 font-mono text-sm">
              {logs.map((log, index) => (
                <motion.div
                  key={index}
                  className="flex items-start space-x-3"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <span className="text-text-muted text-xs mt-0.5 w-16 flex-shrink-0">
                    {log.time}
                  </span>
                  <span className={`${getLogColor(log.type)} flex-1`}>
                    {log.message}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Training;
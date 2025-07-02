import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  CloudArrowUpIcon,
  PhotoIcon,
  EyeIcon,
  ClockIcon,
  ChartBarIcon,
  CheckCircleIcon,
  XMarkIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const Inference = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      preview: URL.createObjectURL(file),
      status: 'pending',
      result: null
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
    
    // Simulate processing
    newFiles.forEach(fileObj => {
      processFile(fileObj);
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 10
  });

  const processFile = async (fileObj) => {
    setIsProcessing(true);
    
    // Update file status to processing
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileObj.id ? { ...f, status: 'processing' } : f)
    );

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

    // Generate mock results
    const mockResult = {
      classification: {
        predictions: [
          { class: 'Apple', confidence: 0.92, nutrition: { calories: 95, protein: 0.5, carbs: 25, fat: 0.3 } },
          { class: 'Banana', confidence: 0.08, nutrition: { calories: 105, protein: 1.3, carbs: 27, fat: 0.4 } }
        ]
      },
      segmentation: {
        maskUrl: '/api/masks/' + fileObj.id,
        area: 2847, // pixels
        boundingBox: { x: 120, y: 80, width: 200, height: 180 }
      },
      metadata: {
        processingTime: (2000 + Math.random() * 3000).toFixed(0) + 'ms',
        modelVersion: 'v2.0.0',
        timestamp: new Date().toISOString()
      }
    };

    // Update with results
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileObj.id ? { 
        ...f, 
        status: 'completed',
        result: mockResult
      } : f)
    );

    setResults(prev => [...prev, { ...fileObj, result: mockResult }]);
    setIsProcessing(false);
  };

  const removeFile = (id) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
    setResults(prev => prev.filter(r => r.id !== id));
  };

  const ResultCard = ({ fileObj, onClick }) => (
    <motion.div
      className="card cursor-pointer"
      onClick={() => onClick(fileObj)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      layout
    >
      <div className="relative">
        <img
          src={fileObj.preview}
          alt="Food"
          className="w-full h-48 object-cover rounded-lg"
        />
        
        {/* Status Overlay */}
        <div className="absolute top-3 right-3">
          {fileObj.status === 'processing' && (
            <div className="bg-background-dark/80 rounded-full p-2">
              <ArrowPathIcon className="w-5 h-5 text-primary-400 animate-spin" />
            </div>
          )}
          {fileObj.status === 'completed' && (
            <div className="bg-background-dark/80 rounded-full p-2">
              <CheckCircleIcon className="w-5 h-5 text-green-400" />
            </div>
          )}
        </div>

        {/* Remove Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            removeFile(fileObj.id);
          }}
          className="absolute top-3 left-3 bg-background-dark/80 rounded-full p-2 hover:bg-red-500/80 transition-colors"
        >
          <XMarkIcon className="w-4 h-4 text-text-primary" />
        </button>
      </div>

      <div className="mt-4">
        <h3 className="font-semibold text-text-primary truncate">{fileObj.file.name}</h3>
        
        {fileObj.result && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">Top Prediction</span>
              <span className="text-sm font-medium text-primary-400">
                {fileObj.result.classification.predictions[0].class}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">Confidence</span>
              <span className="text-sm font-medium text-text-primary">
                {(fileObj.result.classification.predictions[0].confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}

        {fileObj.status === 'processing' && (
          <div className="mt-3">
            <div className="flex items-center space-x-2">
              <div className="loading-spinner w-4 h-4"></div>
              <span className="text-sm text-text-muted">Processing...</span>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );

  const DetailModal = ({ result, onClose }) => (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      <motion.div
        className="relative glass-effect rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-text-primary">Analysis Results</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-secondary-800 rounded-lg transition-colors"
          >
            <XMarkIcon className="w-6 h-6 text-text-primary" />
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Image */}
          <div>
            <img
              src={result.preview}
              alt="Food Analysis"
              className="w-full rounded-lg"
            />
          </div>

          {/* Results */}
          <div className="space-y-6">
            {/* Classification Results */}
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-3">Classification</h3>
              <div className="space-y-3">
                {result.result.classification.predictions.map((pred, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-background-medium/30 rounded-lg">
                    <span className="font-medium text-text-primary">{pred.class}</span>
                    <div className="flex items-center space-x-3">
                      <div className="w-24 bg-secondary-800 rounded-full h-2">
                        <div 
                          className="h-2 bg-gradient-to-r from-primary-500 to-primary-400 rounded-full"
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-primary-400">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Nutrition Information */}
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-3">Nutrition (Per Serving)</h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(result.result.classification.predictions[0].nutrition).map(([key, value]) => (
                  <div key={key} className="p-3 bg-background-medium/30 rounded-lg">
                    <p className="text-text-muted text-sm capitalize">{key}</p>
                    <p className="text-lg font-bold text-primary-400">
                      {value}{key === 'calories' ? ' kcal' : 'g'}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Segmentation Results */}
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-3">Segmentation</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted text-sm">Detected Area</p>
                  <p className="text-lg font-bold text-primary-400">{result.result.segmentation.area} px²</p>
                </div>
                <div className="p-3 bg-background-medium/30 rounded-lg">
                  <p className="text-text-muted text-sm">Processing Time</p>
                  <p className="text-lg font-bold text-primary-400">{result.result.metadata.processingTime}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-text-primary mb-2">Food Detection</h1>
        <p className="text-text-muted">Upload images to analyze food items and get nutritional information</p>
      </div>

      {/* Upload Zone */}
      <motion.div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'active' : ''}`}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <input {...getInputProps()} />
        <div className="text-center">
          <CloudArrowUpIcon className="w-16 h-16 text-primary-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-text-primary mb-2">
            {isDragActive ? 'Drop images here...' : 'Upload Food Images'}
          </h3>
          <p className="text-text-muted mb-4">
            Drag and drop images or click to browse
          </p>
          <div className="flex items-center justify-center space-x-4 text-sm text-text-muted">
            <span>• Supports JPG, PNG, WebP</span>
            <span>• Max 10 files</span>
            <span>• Up to 10MB each</span>
          </div>
        </div>
      </motion.div>

      {/* Processing Stats */}
      {uploadedFiles.length > 0 && (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="card text-center">
            <PhotoIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">{uploadedFiles.length}</p>
            <p className="text-text-muted text-sm">Images Uploaded</p>
          </div>
          <div className="card text-center">
            <EyeIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {uploadedFiles.filter(f => f.status === 'completed').length}
            </p>
            <p className="text-text-muted text-sm">Analyzed</p>
          </div>
          <div className="card text-center">
            <ClockIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {uploadedFiles.filter(f => f.status === 'processing').length}
            </p>
            <p className="text-text-muted text-sm">Processing</p>
          </div>
        </motion.div>
      )}

      {/* Results Grid */}
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {uploadedFiles.map((fileObj) => (
              <ResultCard
                key={fileObj.id}
                fileObj={fileObj}
                onClick={setSelectedResult}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedResult && (
          <DetailModal
            result={selectedResult}
            onClose={() => setSelectedResult(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Inference;
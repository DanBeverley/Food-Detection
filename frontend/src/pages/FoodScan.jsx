import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  CameraIcon,
  PhotoIcon,
  ClockIcon,
  FireIcon,
  ChartBarIcon,
  CheckCircleIcon,
  XMarkIcon,
  ArrowPathIcon,
  PlusIcon,
  BookmarkIcon
} from '@heroicons/react/24/outline';

const FoodScan = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [savedMeals, setSavedMeals] = useState([]);

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
    maxFiles: 5
  });

  const processFile = async (fileObj) => {
    // Update file status to processing
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileObj.id ? { ...f, status: 'processing' } : f)
    );

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

    // Generate mock results
    const foodItems = [
      { name: 'Grilled Chicken Breast', confidence: 0.94, calories: 231, protein: 43.5, carbs: 0, fat: 5.0, fiber: 0, sugar: 0 },
      { name: 'Caesar Salad', confidence: 0.89, calories: 470, protein: 7.0, carbs: 12, fat: 44, fiber: 3, sugar: 4 },
      { name: 'Brown Rice', confidence: 0.91, calories: 216, protein: 5.0, carbs: 45, fat: 1.8, fiber: 4, sugar: 1 },
      { name: 'Avocado', confidence: 0.87, calories: 234, protein: 2.9, carbs: 12, fat: 21, fiber: 10, sugar: 1 }
    ];

    const selectedFood = foodItems[Math.floor(Math.random() * foodItems.length)];
    
    const mockResult = {
      food: selectedFood,
      servingSize: '1 cup',
      nutrition: {
        calories: selectedFood.calories,
        protein: selectedFood.protein,
        carbs: selectedFood.carbs,
        fat: selectedFood.fat,
        fiber: selectedFood.fiber,
        sugar: selectedFood.sugar
      },
      confidence: selectedFood.confidence,
      alternatives: foodItems.filter(f => f.name !== selectedFood.name).slice(0, 2),
      metadata: {
        processingTime: (2000 + Math.random() * 3000).toFixed(0) + 'ms',
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
  };

  const saveToFoodLog = (fileObj) => {
    const meal = {
      id: Math.random().toString(36).substr(2, 9),
      name: fileObj.result.food.name,
      image: fileObj.preview,
      calories: fileObj.result.nutrition.calories,
      protein: fileObj.result.nutrition.protein,
      carbs: fileObj.result.nutrition.carbs,
      fat: fileObj.result.nutrition.fat,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      date: new Date().toLocaleDateString()
    };
    
    setSavedMeals(prev => [meal, ...prev]);
    
    // Show success message
    alert(`${meal.name} added to your food log!`);
  };

  const removeFile = (id) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
  };

  const FoodResultCard = ({ fileObj, onClick }) => (
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
        <div className="absolute top-3 right-3 flex space-x-2">
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
        {fileObj.result && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-text-primary">{fileObj.result.food.name}</h3>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                {(fileObj.result.confidence * 100).toFixed(0)}% match
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-text-muted">Calories</span>
                <span className="font-medium text-primary-400">{fileObj.result.nutrition.calories}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-muted">Protein</span>
                <span className="font-medium text-green-400">{fileObj.result.nutrition.protein}g</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-muted">Carbs</span>
                <span className="font-medium text-yellow-400">{fileObj.result.nutrition.carbs}g</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-muted">Fat</span>
                <span className="font-medium text-red-400">{fileObj.result.nutrition.fat}g</span>
              </div>
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                saveToFoodLog(fileObj);
              }}
              className="w-full button-primary text-sm py-2"
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              Add to Food Log
            </button>
          </div>
        )}

        {fileObj.status === 'processing' && (
          <div className="mt-3">
            <div className="flex items-center space-x-2">
              <div className="loading-spinner w-4 h-4"></div>
              <span className="text-sm text-text-muted">Analyzing food...</span>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              AI is identifying your food and calculating nutrition
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
        className="relative glass-effect rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-text-primary">Nutrition Analysis</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-secondary-800 rounded-lg transition-colors"
          >
            <XMarkIcon className="w-6 h-6 text-text-primary" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Image */}
          <div>
            <img
              src={result.preview}
              alt="Food Analysis"
              className="w-full rounded-lg"
            />
            <div className="mt-4 p-4 bg-background-medium/30 rounded-lg">
              <h4 className="font-semibold text-text-primary mb-2">Alternatives</h4>
              <div className="space-y-2">
                {result.result.alternatives.map((alt, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span className="text-text-primary">{alt.name}</span>
                    <span className="text-text-muted">{(alt.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Nutrition Details */}
          <div className="space-y-6">
            {/* Main Food */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-text-primary">{result.result.food.name}</h3>
                <span className="text-sm bg-green-500/20 text-green-400 px-3 py-1 rounded-full">
                  {(result.result.confidence * 100).toFixed(0)}% confident
                </span>
              </div>
              <p className="text-text-muted mb-4">Per {result.result.servingSize}</p>
            </div>

            {/* Nutrition Grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-background-medium/30 rounded-lg text-center">
                <FireIcon className="w-6 h-6 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-primary-400">{result.result.nutrition.calories}</p>
                <p className="text-sm text-text-muted">Calories</p>
              </div>
              <div className="p-4 bg-background-medium/30 rounded-lg text-center">
                <ChartBarIcon className="w-6 h-6 text-green-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-green-400">{result.result.nutrition.protein}g</p>
                <p className="text-sm text-text-muted">Protein</p>
              </div>
              <div className="p-4 bg-background-medium/30 rounded-lg text-center">
                <div className="w-6 h-6 bg-yellow-400 rounded mx-auto mb-2"></div>
                <p className="text-2xl font-bold text-yellow-400">{result.result.nutrition.carbs}g</p>
                <p className="text-sm text-text-muted">Carbs</p>
              </div>
              <div className="p-4 bg-background-medium/30 rounded-lg text-center">
                <div className="w-6 h-6 bg-red-400 rounded mx-auto mb-2"></div>
                <p className="text-2xl font-bold text-red-400">{result.result.nutrition.fat}g</p>
                <p className="text-sm text-text-muted">Fat</p>
              </div>
            </div>

            {/* Additional Info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-text-muted">Fiber</span>
                <span className="text-text-primary">{result.result.nutrition.fiber}g</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Sugar</span>
                <span className="text-text-primary">{result.result.nutrition.sugar}g</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              <button
                onClick={() => {
                  saveToFoodLog(result);
                  onClose();
                }}
                className="w-full button-primary"
              >
                <PlusIcon className="w-5 h-5 mr-2" />
                Add to Food Log
              </button>
              <button className="w-full button-secondary">
                <BookmarkIcon className="w-5 h-5 mr-2" />
                Save to Favorites
              </button>
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
        <h1 className="text-4xl font-bold text-text-primary mb-2">Scan Your Food 📸</h1>
        <p className="text-text-muted">Take a photo or upload an image to get instant nutrition analysis</p>
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
          <CameraIcon className="w-16 h-16 text-primary-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-text-primary mb-2">
            {isDragActive ? 'Drop your food photo here!' : 'Snap or Upload Food Photo'}
          </h3>
          <p className="text-text-muted mb-4">
            Our AI will analyze your food and provide detailed nutrition information
          </p>
          <div className="flex items-center justify-center space-x-4 text-sm text-text-muted">
            <span>• Supports JPG, PNG, WebP</span>
            <span>• Max 5 photos</span>
            <span>• Instant analysis</span>
          </div>
        </div>
      </motion.div>

      {/* Quick Stats */}
      {uploadedFiles.length > 0 && (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="card text-center">
            <PhotoIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">{uploadedFiles.length}</p>
            <p className="text-text-muted text-sm">Photos Scanned</p>
          </div>
          <div className="card text-center">
            <CheckCircleIcon className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {uploadedFiles.filter(f => f.status === 'completed').length}
            </p>
            <p className="text-text-muted text-sm">Foods Identified</p>
          </div>
          <div className="card text-center">
            <ClockIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">
              {uploadedFiles.filter(f => f.status === 'processing').length}
            </p>
            <p className="text-text-muted text-sm">Analyzing</p>
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
              <FoodResultCard
                key={fileObj.id}
                fileObj={fileObj}
                onClick={setSelectedResult}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Recent Saves */}
      {savedMeals.length > 0 && (
        <motion.div
          className="card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 className="text-lg font-semibold text-text-primary mb-4">Recently Added to Food Log</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {savedMeals.slice(0, 3).map((meal) => (
              <div key={meal.id} className="flex items-center space-x-3 p-3 bg-background-medium/30 rounded-lg">
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-lg flex items-center justify-center">
                  <CheckCircleIcon className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-text-primary">{meal.name}</h4>
                  <p className="text-sm text-text-muted">{meal.time} • {meal.calories} cal</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

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

export default FoodScan;
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CameraIcon,
  PlusIcon,
  MagnifyingGlassIcon,
  CalendarDaysIcon,
  ClockIcon,
  ChartBarIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { Link } from 'react-router-dom';

const FoodLog = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [searchTerm, setSearchTerm] = useState('');
  const [foodEntries, setFoodEntries] = useState([]); // Empty for new user
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedMealType, setSelectedMealType] = useState('');
  const [newEntry, setNewEntry] = useState({
    name: '',
    calories: '',
    protein: '',
    carbs: '',
    fat: '',
    quantity: '1 serving'
  });

  const handleAddEntry = () => {
    if (!newEntry.name || !newEntry.calories) {
      alert('Please enter at least food name and calories');
      return;
    }

    const entry = {
      id: Date.now(),
      name: newEntry.name,
      calories: parseInt(newEntry.calories),
      protein: parseFloat(newEntry.protein) || 0,
      carbs: parseFloat(newEntry.carbs) || 0,
      fat: parseFloat(newEntry.fat) || 0,
      quantity: newEntry.quantity,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      meal: selectedMealType,
      date: selectedDate
    };

    setFoodEntries(prev => [...prev, entry]);
    setShowAddModal(false);
    setNewEntry({
      name: '',
      calories: '',
      protein: '',
      carbs: '',
      fat: '',
      quantity: '1 serving'
    });
  };

  const openAddModal = (mealType = '') => {
    setSelectedMealType(mealType);
    setShowAddModal(true);
  };

  const AddEntryModal = () => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        className="bg-background-dark border border-secondary-800 rounded-2xl p-6 w-full max-w-md"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-text-primary">
            Add Manual Entry {selectedMealType && `- ${selectedMealType}`}
          </h3>
          <button
            onClick={() => setShowAddModal(false)}
            className="text-text-muted hover:text-text-primary"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Food Name</label>
            <input
              type="text"
              value={newEntry.name}
              onChange={(e) => setNewEntry(prev => ({ ...prev, name: e.target.value }))}
              className="input-field w-full"
              placeholder="e.g. Grilled Chicken Breast"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Calories</label>
              <input
                type="number"
                value={newEntry.calories}
                onChange={(e) => setNewEntry(prev => ({ ...prev, calories: e.target.value }))}
                className="input-field w-full"
                placeholder="250"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Quantity</label>
              <input
                type="text"
                value={newEntry.quantity}
                onChange={(e) => setNewEntry(prev => ({ ...prev, quantity: e.target.value }))}
                className="input-field w-full"
                placeholder="1 serving"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Protein (g)</label>
              <input
                type="number"
                step="0.1"
                value={newEntry.protein}
                onChange={(e) => setNewEntry(prev => ({ ...prev, protein: e.target.value }))}
                className="input-field w-full"
                placeholder="25"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Carbs (g)</label>
              <input
                type="number"
                step="0.1"
                value={newEntry.carbs}
                onChange={(e) => setNewEntry(prev => ({ ...prev, carbs: e.target.value }))}
                className="input-field w-full"
                placeholder="12"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Fat (g)</label>
              <input
                type="number"
                step="0.1"
                value={newEntry.fat}
                onChange={(e) => setNewEntry(prev => ({ ...prev, fat: e.target.value }))}
                className="input-field w-full"
                placeholder="8"
              />
            </div>
          </div>
        </div>

        <div className="flex space-x-4 mt-6">
          <button
            onClick={() => setShowAddModal(false)}
            className="button-secondary flex-1"
          >
            Cancel
          </button>
          <button
            onClick={handleAddEntry}
            className="button-primary flex-1"
          >
            Add Entry
          </button>
        </div>
      </motion.div>
    </div>
  );

  const EmptyState = () => (
    <motion.div
      className="text-center py-16"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <motion.div
        className="w-24 h-24 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto mb-6"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
      >
        <ChartBarIcon className="w-12 h-12 text-white" />
      </motion.div>
      
      <h3 className="text-2xl font-bold text-text-primary mb-4">No meals logged yet</h3>
      <p className="text-text-muted mb-8 max-w-md mx-auto">
        Start tracking your nutrition by scanning a meal or adding a manual entry. Your food log will appear here.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Link to="/scan" className="button-primary">
          <CameraIcon className="w-5 h-5 mr-2" />
          Scan Food
        </Link>
        <button 
          onClick={() => openAddModal()}
          className="button-secondary"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Add Manual Entry
        </button>
      </div>
    </motion.div>
  );

  const DailySummary = () => {
    const todaysEntries = foodEntries.filter(entry => entry.date === selectedDate);
    const totalCalories = todaysEntries.reduce((sum, entry) => sum + entry.calories, 0);
    const totalProtein = todaysEntries.reduce((sum, entry) => sum + entry.protein, 0);
    const totalCarbs = todaysEntries.reduce((sum, entry) => sum + entry.carbs, 0);
    const totalFat = todaysEntries.reduce((sum, entry) => sum + entry.fat, 0);

    return (
      <motion.div
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-400">{totalCalories}</div>
          <div className="text-sm text-text-muted">Calories</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-secondary-400">{totalProtein.toFixed(1)}g</div>
          <div className="text-sm text-text-muted">Protein</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-400">{totalCarbs.toFixed(1)}g</div>
          <div className="text-sm text-text-muted">Carbs</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-secondary-400">{totalFat.toFixed(1)}g</div>
          <div className="text-sm text-text-muted">Fat</div>
        </div>
      </motion.div>
    );
  };

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <motion.div
        className="flex flex-col md:flex-row md:items-center justify-between gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Food Log</h1>
          <p className="text-text-muted">Track your daily nutrition and meals</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <CalendarDaysIcon className="w-5 h-5 text-primary-400" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="input-field text-sm"
            />
          </div>
          <Link to="/scan" className="button-primary">
            <CameraIcon className="w-5 h-5 mr-2" />
            Quick Scan
          </Link>
        </div>
      </motion.div>

      {/* Daily Summary */}
      <DailySummary />

      {/* Search and Filter */}
      <motion.div
        className="flex flex-col md:flex-row gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="relative flex-1">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-text-muted" />
          <input
            type="text"
            placeholder="Search food entries..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field pl-10 w-full"
          />
        </div>
        <button 
          onClick={() => openAddModal()}
          className="button-secondary"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Add Entry
        </button>
      </motion.div>

      {/* Meal Sections */}
      <div className="space-y-6">
        {['Breakfast', 'Lunch', 'Dinner', 'Snacks'].map((mealType, index) => (
          <motion.div
            key={mealType}
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 + index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <ClockIcon className="w-5 h-5 text-primary-400" />
                <h3 className="text-lg font-semibold text-text-primary">{mealType}</h3>
                <span className="text-sm text-text-muted">
                  {foodEntries
                    .filter(entry => entry.meal === mealType && entry.date === selectedDate)
                    .reduce((sum, entry) => sum + entry.calories, 0)} cal
                </span>
              </div>
              <button 
                onClick={() => openAddModal(mealType)}
                className="text-primary-400 hover:text-primary-300 transition-colors"
              >
                <PlusIcon className="w-5 h-5" />
              </button>
            </div>
            
            {(() => {
              const mealEntries = foodEntries.filter(entry => entry.meal === mealType && entry.date === selectedDate);
              if (mealEntries.length === 0) {
                return (
                  <div className="text-center py-8 border-2 border-dashed border-secondary-700 rounded-lg">
                    <p className="text-text-muted text-sm">No {mealType.toLowerCase()} logged</p>
                    <button 
                      onClick={() => openAddModal(mealType)}
                      className="text-primary-400 text-sm mt-2 hover:text-primary-300"
                    >
                      Add food to {mealType.toLowerCase()}
                    </button>
                  </div>
                );
              }
              
              return (
                <div className="space-y-3">
                  {mealEntries.map((entry) => (
                    <div
                      key={entry.id}
                      className="flex items-center justify-between p-4 bg-background-medium/30 rounded-lg border border-secondary-700/50"
                    >
                      <div className="flex-1">
                        <h4 className="font-medium text-text-primary">{entry.name}</h4>
                        <p className="text-sm text-text-muted">
                          {entry.quantity} • {entry.time} • {entry.calories} cal
                        </p>
                        <div className="text-xs text-text-muted mt-1">
                          P: {entry.protein}g • C: {entry.carbs}g • F: {entry.fat}g
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </motion.div>
        ))}
      </div>

      {/* Empty State for entire log */}
      {foodEntries.length === 0 && <EmptyState />}

      {/* Add Entry Modal */}
      {showAddModal && <AddEntryModal />}
    </motion.div>
  );
};

export default FoodLog;
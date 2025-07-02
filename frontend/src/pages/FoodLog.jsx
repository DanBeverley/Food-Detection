import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ClockIcon,
  FireIcon,
  ChartBarIcon,
  PlusIcon,
  MagnifyingGlassIcon,
  CalendarDaysIcon,
  TrashIcon,
  PencilIcon
} from '@heroicons/react/24/outline';

const FoodLog = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [searchTerm, setSearchTerm] = useState('');
  
  const [foodEntries] = useState([
    {
      id: 1,
      name: 'Grilled Chicken Salad',
      brand: 'Homemade',
      calories: 420,
      protein: 35,
      carbs: 12,
      fat: 8,
      fiber: 4,
      time: '12:30 PM',
      meal: 'Lunch',
      quantity: '1 serving',
      image: '/api/placeholder/meal1'
    },
    {
      id: 2,
      name: 'Greek Yogurt with Berries',
      brand: 'Chobani',
      calories: 180,
      protein: 15,
      carbs: 20,
      fat: 5,
      fiber: 3,
      time: '9:15 AM',
      meal: 'Breakfast',
      quantity: '1 cup',
      image: '/api/placeholder/meal2'
    },
    {
      id: 3,
      name: 'Oatmeal with Banana',
      brand: 'Quaker',
      calories: 320,
      protein: 8,
      carbs: 58,
      fat: 6,
      fiber: 8,
      time: '7:45 AM',
      meal: 'Breakfast',
      quantity: '1 bowl',
      image: '/api/placeholder/meal3'
    },
    {
      id: 4,
      name: 'Almonds',
      brand: 'Blue Diamond',
      calories: 164,
      protein: 6,
      carbs: 6,
      fat: 14,
      fiber: 4,
      time: '3:20 PM',
      meal: 'Snack',
      quantity: '1 oz (23 almonds)',
      image: '/api/placeholder/meal4'
    },
    {
      id: 5,
      name: 'Salmon with Quinoa',
      brand: 'Homemade',
      calories: 520,
      protein: 42,
      carbs: 35,
      fat: 18,
      fiber: 5,
      time: '7:00 PM',
      meal: 'Dinner',
      quantity: '1 serving',
      image: '/api/placeholder/meal5'
    }
  ]);

  const mealTimes = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];
  
  const dailyTotals = foodEntries.reduce(
    (acc, entry) => ({
      calories: acc.calories + entry.calories,
      protein: acc.protein + entry.protein,
      carbs: acc.carbs + entry.carbs,
      fat: acc.fat + entry.fat,
      fiber: acc.fiber + entry.fiber
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0 }
  );

  const dailyGoals = {
    calories: 2500,
    protein: 150,
    carbs: 300,
    fat: 85,
    fiber: 30
  };

  const filteredEntries = foodEntries.filter(entry =>
    entry.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entry.brand.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entry.meal.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const groupedEntries = mealTimes.reduce((acc, meal) => {
    acc[meal] = filteredEntries.filter(entry => entry.meal === meal);
    return acc;
  }, {});

  const NutritionSummary = () => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-4">Today's Nutrition</h3>
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {Object.entries(dailyTotals).map(([key, value], index) => {
          const goal = dailyGoals[key];
          const progress = (value / goal) * 100;
          const colors = {
            calories: 'text-primary-400',
            protein: 'text-green-400',
            carbs: 'text-yellow-400',
            fat: 'text-red-400',
            fiber: 'text-blue-400'
          };
          
          return (
            <div key={key} className="text-center">
              <div className="mb-2">
                <p className={`text-2xl font-bold ${colors[key]}`}>{Math.round(value)}</p>
                <p className="text-xs text-text-muted">/ {goal} {key === 'calories' ? 'cal' : 'g'}</p>
              </div>
              <div className="progress-bar h-2 mb-2">
                <motion.div
                  className={`h-2 rounded-full ${
                    key === 'calories' ? 'bg-primary-400' :
                    key === 'protein' ? 'bg-green-400' :
                    key === 'carbs' ? 'bg-yellow-400' :
                    key === 'fat' ? 'bg-red-400' : 'bg-blue-400'
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, progress)}%` }}
                  transition={{ duration: 1, delay: 0.3 + index * 0.1 }}
                />
              </div>
              <p className="text-xs text-text-muted capitalize">{key}</p>
            </div>
          );
        })}
      </div>
    </motion.div>
  );

  const FoodEntryCard = ({ entry }) => (
    <motion.div
      className="flex items-center space-x-4 p-4 rounded-xl bg-background-medium/30 hover:bg-background-medium/50 transition-all group"
      whileHover={{ scale: 1.01 }}
      layout
    >
      <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
        <FireIcon className="w-8 h-8 text-white" />
      </div>
      
      <div className="flex-1">
        <div className="flex items-start justify-between">
          <div>
            <h4 className="font-semibold text-text-primary">{entry.name}</h4>
            <p className="text-sm text-text-muted">{entry.brand} • {entry.quantity}</p>
            <p className="text-xs text-text-muted">{entry.time}</p>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-primary-400">{entry.calories}</p>
            <p className="text-xs text-text-muted">calories</p>
          </div>
        </div>
        
        <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
          <div className="text-center">
            <p className="font-medium text-green-400">{entry.protein}g</p>
            <p className="text-text-muted">Protein</p>
          </div>
          <div className="text-center">
            <p className="font-medium text-yellow-400">{entry.carbs}g</p>
            <p className="text-text-muted">Carbs</p>
          </div>
          <div className="text-center">
            <p className="font-medium text-red-400">{entry.fat}g</p>
            <p className="text-text-muted">Fat</p>
          </div>
          <div className="text-center">
            <p className="font-medium text-blue-400">{entry.fiber}g</p>
            <p className="text-text-muted">Fiber</p>
          </div>
        </div>
      </div>
      
      <div className="opacity-0 group-hover:opacity-100 transition-opacity flex space-x-2">
        <button className="p-2 hover:bg-secondary-800 rounded-lg transition-colors">
          <PencilIcon className="w-4 h-4 text-text-primary" />
        </button>
        <button className="p-2 hover:bg-red-500/20 rounded-lg transition-colors">
          <TrashIcon className="w-4 h-4 text-red-400" />
        </button>
      </div>
    </motion.div>
  );

  const MealSection = ({ mealType, entries }) => {
    const mealCalories = entries.reduce((sum, entry) => sum + entry.calories, 0);
    
    return (
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        layout
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-text-primary">{mealType}</h3>
            <span className="text-sm text-text-muted">({entries.length} items)</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-primary-400">{mealCalories} cal</span>
            <button className="button-secondary text-sm py-1 px-3">
              <PlusIcon className="w-4 h-4 mr-1" />
              Add Food
            </button>
          </div>
        </div>
        
        <div className="space-y-3">
          {entries.length > 0 ? (
            entries.map((entry) => (
              <FoodEntryCard key={entry.id} entry={entry} />
            ))
          ) : (
            <div className="text-center py-8 text-text-muted">
              <FireIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No {mealType.toLowerCase()} logged yet</p>
              <button className="button-primary text-sm mt-3">Add {mealType}</button>
            </div>
          )}
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-text-primary mb-2">Food Log 📝</h1>
          <p className="text-text-muted">Track your daily nutrition and meals</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="relative">
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="input-field pr-10"
            />
            <CalendarDaysIcon className="w-5 h-5 text-text-muted absolute right-3 top-1/2 transform -translate-y-1/2" />
          </div>
          <button className="button-primary">
            <PlusIcon className="w-5 h-5 mr-2" />
            Quick Add
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <motion.div
        className="relative"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <input
          type="text"
          placeholder="Search foods, brands, or meals..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="input-field w-full pl-12"
        />
        <MagnifyingGlassIcon className="w-5 h-5 text-text-muted absolute left-4 top-1/2 transform -translate-y-1/2" />
      </motion.div>

      {/* Daily Nutrition Summary */}
      <NutritionSummary />

      {/* Meal Sections */}
      <div className="space-y-6">
        {mealTimes.map((mealType, index) => (
          <motion.div
            key={mealType}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
          >
            <MealSection 
              mealType={mealType} 
              entries={groupedEntries[mealType] || []} 
            />
          </motion.div>
        ))}
      </div>

      {/* Quick Actions */}
      <motion.div
        className="fixed bottom-8 right-8 flex flex-col space-y-3"
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1 }}
      >
        <button className="floating-action bg-gradient-to-r from-green-500 to-green-600">
          <PlusIcon className="w-6 h-6" />
        </button>
      </motion.div>
    </motion.div>
  );
};

export default FoodLog;
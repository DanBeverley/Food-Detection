import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  FlagIcon,
  FireIcon,
  ChartBarIcon,
  ScaleIcon,
  ClockIcon,
  PlusIcon,
  CheckCircleIcon,
  PencilIcon,
  TrashIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';

const Goals = () => {
  const [activeTab, setActiveTab] = useState('nutrition');
  const [showAddGoal, setShowAddGoal] = useState(false);
  
  const [nutritionGoals, setNutritionGoals] = useState([
    {
      id: 1,
      type: 'calories',
      target: 2500,
      current: 1847,
      unit: 'cal',
      icon: FireIcon,
      color: 'text-primary-400',
      bgColor: 'bg-primary-500/10',
      isDaily: true,
      progress: 74
    },
    {
      id: 2,
      type: 'protein',
      target: 150,
      current: 125,
      unit: 'g',
      icon: ChartBarIcon,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      isDaily: true,
      progress: 83
    },
    {
      id: 3,
      type: 'water',
      target: 8,
      current: 6,
      unit: 'cups',
      icon: BeakerIcon,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
      isDaily: true,
      progress: 75
    }
  ]);

  const [fitnessGoals, setFitnessGoals] = useState([
    {
      id: 4,
      type: 'weight_loss',
      target: 160,
      current: 165.2,
      unit: 'lbs',
      icon: ScaleIcon,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      isDaily: false,
      progress: 68,
      deadline: '2025-09-01'
    },
    {
      id: 5,
      type: 'weekly_deficit',
      target: 3500,
      current: 2800,
      unit: 'cal',
      icon: FlagIcon,
      color: 'text-red-400',
      bgColor: 'bg-red-500/10',
      isDaily: false,
      progress: 80,
      deadline: '2025-07-06'
    }
  ]);

  const [customGoals, setCustomGoals] = useState([
    {
      id: 6,
      name: 'Log meals for 30 days',
      description: 'Build a consistent tracking habit',
      target: 30,
      current: 12,
      unit: 'days',
      icon: ClockIcon,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/10',
      progress: 40,
      deadline: '2025-07-30'
    }
  ]);

  const GoalCard = ({ goal, onEdit, onDelete }) => (
    <motion.div
      className="card group"
      whileHover={{ scale: 1.02 }}
      layout
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-3 rounded-xl ${goal.bgColor}`}>
            <goal.icon className={`w-6 h-6 ${goal.color}`} />
          </div>
          <div>
            <h4 className="font-semibold text-text-primary capitalize">
              {goal.name || goal.type.replace('_', ' ')}
            </h4>
            {goal.description && (
              <p className="text-sm text-text-muted">{goal.description}</p>
            )}
            {goal.deadline && (
              <p className="text-xs text-text-muted">Due: {goal.deadline}</p>
            )}
          </div>
        </div>
        
        <div className="opacity-0 group-hover:opacity-100 transition-opacity flex space-x-2">
          <button 
            onClick={() => onEdit(goal)}
            className="p-2 hover:bg-secondary-800 rounded-lg transition-colors"
          >
            <PencilIcon className="w-4 h-4 text-text-primary" />
          </button>
          <button 
            onClick={() => onDelete(goal.id)}
            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
          >
            <TrashIcon className="w-4 h-4 text-red-400" />
          </button>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-baseline space-x-2">
            <span className={`text-2xl font-bold ${goal.color}`}>
              {goal.type === 'weight_loss' ? goal.current : goal.current}
            </span>
            <span className="text-text-muted">
              / {goal.target} {goal.unit}
            </span>
          </div>
          
          <div className="text-right">
            <p className={`text-lg font-bold ${goal.color}`}>{goal.progress}%</p>
            <p className="text-xs text-text-muted">Complete</p>
          </div>
        </div>

        <div className="progress-bar h-3">
          <motion.div
            className={`h-3 rounded-full ${
              goal.type === 'calories' ? 'bg-gradient-to-r from-primary-500 to-primary-400' :
              goal.type === 'protein' ? 'bg-gradient-to-r from-green-500 to-green-400' :
              goal.type === 'water' ? 'bg-gradient-to-r from-blue-500 to-blue-400' :
              goal.type === 'weight_loss' ? 'bg-gradient-to-r from-purple-500 to-purple-400' :
              goal.type === 'weekly_deficit' ? 'bg-gradient-to-r from-red-500 to-red-400' :
              'bg-gradient-to-r from-yellow-500 to-yellow-400'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${goal.progress}%` }}
            transition={{ duration: 1, delay: 0.2 }}
          />
        </div>

        {goal.isDaily && (
          <div className="text-center">
            <p className="text-xs text-text-muted">
              {goal.target - goal.current > 0 
                ? `${(goal.target - goal.current).toFixed(goal.unit === 'cal' ? 0 : 1)} ${goal.unit} remaining today`
                : `Goal exceeded by ${(goal.current - goal.target).toFixed(goal.unit === 'cal' ? 0 : 1)} ${goal.unit}!`
              }
            </p>
          </div>
        )}

        {!goal.isDaily && goal.deadline && (
          <div className="text-center">
            <p className="text-xs text-text-muted">
              Target date: {new Date(goal.deadline).toLocaleDateString()}
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );

  const AddGoalForm = ({ onClose, onAdd }) => {
    const [goalType, setGoalType] = useState('nutrition');
    const [goalData, setGoalData] = useState({
      name: '',
      target: '',
      unit: '',
      deadline: '',
      isDaily: true
    });

    const handleSubmit = (e) => {
      e.preventDefault();
      const newGoal = {
        id: Date.now(),
        ...goalData,
        target: parseFloat(goalData.target),
        current: 0,
        progress: 0,
        icon: FlagIcon,
        color: 'text-primary-400',
        bgColor: 'bg-primary-500/10'
      };
      onAdd(newGoal);
      onClose();
    };

    return (
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        
        <motion.div
          className="relative glass-effect rounded-2xl p-6 max-w-md w-full"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
        >
          <h3 className="text-xl font-bold text-text-primary mb-6">Add New Goal</h3>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Goal Name</label>
              <input
                type="text"
                value={goalData.name}
                onChange={(e) => setGoalData({...goalData, name: e.target.value})}
                className="input-field w-full"
                placeholder="e.g., Daily steps, Weekly workouts..."
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">Target</label>
                <input
                  type="number"
                  value={goalData.target}
                  onChange={(e) => setGoalData({...goalData, target: e.target.value})}
                  className="input-field w-full"
                  placeholder="10000"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">Unit</label>
                <input
                  type="text"
                  value={goalData.unit}
                  onChange={(e) => setGoalData({...goalData, unit: e.target.value})}
                  className="input-field w-full"
                  placeholder="steps, lbs, cups..."
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Deadline (Optional)</label>
              <input
                type="date"
                value={goalData.deadline}
                onChange={(e) => setGoalData({...goalData, deadline: e.target.value})}
                className="input-field w-full"
              />
            </div>

            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="isDaily"
                checked={goalData.isDaily}
                onChange={(e) => setGoalData({...goalData, isDaily: e.target.checked})}
                className="w-4 h-4 text-primary-600 bg-background-light border-secondary-600 rounded focus:ring-primary-500 focus:ring-2"
              />
              <label htmlFor="isDaily" className="text-sm text-text-primary">
                This is a daily goal
              </label>
            </div>

            <div className="flex space-x-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 button-secondary"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex-1 button-primary"
              >
                Add Goal
              </button>
            </div>
          </form>
        </motion.div>
      </motion.div>
    );
  };

  const tabs = [
    { id: 'nutrition', name: 'Nutrition', count: nutritionGoals.length },
    { id: 'fitness', name: 'Fitness', count: fitnessGoals.length },
    { id: 'custom', name: 'Custom', count: customGoals.length }
  ];

  const getCurrentGoals = () => {
    switch (activeTab) {
      case 'nutrition': return nutritionGoals;
      case 'fitness': return fitnessGoals;
      case 'custom': return customGoals;
      default: return [];
    }
  };

  const handleAddGoal = (newGoal) => {
    switch (activeTab) {
      case 'nutrition':
        setNutritionGoals([...nutritionGoals, newGoal]);
        break;
      case 'fitness':
        setFitnessGoals([...fitnessGoals, newGoal]);
        break;
      case 'custom':
        setCustomGoals([...customGoals, newGoal]);
        break;
    }
  };

  const handleDeleteGoal = (goalId) => {
    switch (activeTab) {
      case 'nutrition':
        setNutritionGoals(nutritionGoals.filter(g => g.id !== goalId));
        break;
      case 'fitness':
        setFitnessGoals(fitnessGoals.filter(g => g.id !== goalId));
        break;
      case 'custom':
        setCustomGoals(customGoals.filter(g => g.id !== goalId));
        break;
    }
  };

  const currentGoals = getCurrentGoals();
  const completedGoals = currentGoals.filter(g => g.progress >= 100).length;
  const avgProgress = currentGoals.reduce((acc, goal) => acc + goal.progress, 0) / currentGoals.length || 0;

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
          <h1 className="text-4xl font-bold text-text-primary mb-2">Goals & Targets 🎯</h1>
          <p className="text-text-muted">Set and track your health and fitness objectives</p>
        </div>
        
        <motion.button
          className="button-primary"
          onClick={() => setShowAddGoal(true)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Add Goal
        </motion.button>
      </div>

      {/* Summary Stats */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="card text-center">
          <FlagIcon className="w-8 h-8 text-primary-400 mx-auto mb-3" />
          <p className="text-3xl font-bold text-text-primary">{currentGoals.length}</p>
          <p className="text-text-muted text-sm">Active Goals</p>
        </div>
        <div className="card text-center">
          <CheckCircleIcon className="w-8 h-8 text-green-400 mx-auto mb-3" />
          <p className="text-3xl font-bold text-text-primary">{completedGoals}</p>
          <p className="text-text-muted text-sm">Completed</p>
        </div>
        <div className="card text-center">
          <ChartBarIcon className="w-8 h-8 text-primary-400 mx-auto mb-3" />
          <p className="text-3xl font-bold text-text-primary">{Math.round(avgProgress)}%</p>
          <p className="text-text-muted text-sm">Avg Progress</p>
        </div>
      </motion.div>

      {/* Tabs */}
      <motion.div
        className="flex space-x-1 bg-background-medium/30 rounded-xl p-1"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-primary-500 text-white shadow-lg'
                : 'text-text-muted hover:text-text-primary hover:bg-secondary-800/50'
            }`}
          >
            {tab.name}
            <span className="ml-2 text-xs opacity-75">({tab.count})</span>
          </button>
        ))}
      </motion.div>

      {/* Goals Grid */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        layout
      >
        {currentGoals.map((goal, index) => (
          <motion.div
            key={goal.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            layout
          >
            <GoalCard 
              goal={goal} 
              onEdit={() => {}} 
              onDelete={handleDeleteGoal}
            />
          </motion.div>
        ))}
      </motion.div>

      {currentGoals.length === 0 && (
        <motion.div
          className="card text-center py-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <FlagIcon className="w-16 h-16 text-secondary-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-text-primary mb-2">No {activeTab} goals yet</h3>
          <p className="text-text-muted mb-6">
            Set your first {activeTab} goal to start tracking your progress
          </p>
          <button 
            onClick={() => setShowAddGoal(true)}
            className="button-primary"
          >
            <PlusIcon className="w-5 h-5 mr-2" />
            Add Your First Goal
          </button>
        </motion.div>
      )}

      {/* Add Goal Modal */}
      {showAddGoal && (
        <AddGoalForm 
          onClose={() => setShowAddGoal(false)}
          onAdd={handleAddGoal}
        />
      )}
    </motion.div>
  );
};

export default Goals;
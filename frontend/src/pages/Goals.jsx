import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  FlagIcon,
  FireIcon,
  ChartBarIcon,
  ScaleIcon,
  ClockIcon,
  PlusIcon,
  CameraIcon,
  TrophyIcon,
  Cog6ToothIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { Link } from 'react-router-dom';

const Goals = () => {
  const [activeTab, setActiveTab] = useState('nutrition');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newGoal, setNewGoal] = useState({
    name: '',
    description: '',
    target: '',
    unit: '',
    deadline: '',
    category: 'custom'
  });
  
  // Empty arrays for new user
  const [nutritionGoals, setNutritionGoals] = useState([]);
  const [fitnessGoals, setFitnessGoals] = useState([]);
  const [customGoals, setCustomGoals] = useState([]);

  const handleCreateGoal = () => {
    if (!newGoal.name || !newGoal.target) {
      alert('Please enter at least goal name and target');
      return;
    }

    const goal = {
      id: Date.now(),
      name: newGoal.name,
      description: newGoal.description,
      target: parseFloat(newGoal.target),
      current: 0,
      unit: newGoal.unit,
      deadline: newGoal.deadline,
      category: newGoal.category,
      icon: TrophyIcon,
      color: 'text-primary-400',
      bgColor: 'bg-primary-500/10',
      progress: 0,
      createdAt: new Date().toISOString()
    };

    if (newGoal.category === 'nutrition') {
      setNutritionGoals(prev => [...prev, goal]);
    } else if (newGoal.category === 'fitness') {
      setFitnessGoals(prev => [...prev, goal]);
    } else {
      setCustomGoals(prev => [...prev, goal]);
    }

    setShowCreateModal(false);
    setNewGoal({
      name: '',
      description: '',
      target: '',
      unit: '',
      deadline: '',
      category: 'custom'
    });
  };

  const CreateGoalModal = () => (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        className="bg-background-dark border border-secondary-800 rounded-2xl p-6 w-full max-w-lg"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-text-primary">Create New Goal</h3>
          <button
            onClick={() => setShowCreateModal(false)}
            className="text-text-muted hover:text-text-primary"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Goal Name</label>
            <input
              type="text"
              value={newGoal.name}
              onChange={(e) => setNewGoal(prev => ({ ...prev, name: e.target.value }))}
              className="input-field w-full"
              placeholder="e.g. Log meals for 30 days"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Description</label>
            <textarea
              value={newGoal.description}
              onChange={(e) => setNewGoal(prev => ({ ...prev, description: e.target.value }))}
              className="input-field w-full h-20 resize-none"
              placeholder="Describe your goal and why it matters to you"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Target</label>
              <input
                type="number"
                value={newGoal.target}
                onChange={(e) => setNewGoal(prev => ({ ...prev, target: e.target.value }))}
                className="input-field w-full"
                placeholder="30"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Unit</label>
              <input
                type="text"
                value={newGoal.unit}
                onChange={(e) => setNewGoal(prev => ({ ...prev, unit: e.target.value }))}
                className="input-field w-full"
                placeholder="days, lbs, meals, etc."
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Category</label>
              <select
                value={newGoal.category}
                onChange={(e) => setNewGoal(prev => ({ ...prev, category: e.target.value }))}
                className="input-field w-full"
              >
                <option value="custom">Custom</option>
                <option value="nutrition">Nutrition</option>
                <option value="fitness">Fitness</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Deadline (optional)</label>
              <input
                type="date"
                value={newGoal.deadline}
                onChange={(e) => setNewGoal(prev => ({ ...prev, deadline: e.target.value }))}
                className="input-field w-full"
              />
            </div>
          </div>
        </div>

        <div className="flex space-x-4 mt-6">
          <button
            onClick={() => setShowCreateModal(false)}
            className="button-secondary flex-1"
          >
            Cancel
          </button>
          <button
            onClick={handleCreateGoal}
            className="button-primary flex-1"
          >
            Create Goal
          </button>
        </div>
      </motion.div>
    </div>
  );

  const GoalCard = ({ goal, onUpdate, onDelete }) => (
    <motion.div
      className="card"
      whileHover={{ scale: 1.02 }}
      layout
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-10 h-10 ${goal.bgColor} rounded-lg flex items-center justify-center`}>
            <goal.icon className={`w-5 h-5 ${goal.color}`} />
          </div>
          <div>
            <h4 className="font-semibold text-text-primary">{goal.name}</h4>
            <p className="text-sm text-text-muted">{goal.description}</p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-text-muted">Progress</span>
          <span className="text-sm font-medium text-text-primary">
            {goal.current} / {goal.target} {goal.unit}
          </span>
        </div>
        
        <div className="progress-bar h-2">
          <motion.div
            className="bg-gradient-to-r from-primary-500 to-primary-400 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, goal.progress)}%` }}
            transition={{ duration: 1 }}
          />
        </div>

        <div className="flex justify-between items-center text-xs text-text-muted">
          <span>{goal.progress.toFixed(1)}% complete</span>
          {goal.deadline && (
            <span>Due: {new Date(goal.deadline).toLocaleDateString()}</span>
          )}
        </div>
      </div>

      <div className="flex space-x-2 mt-4">
        <button 
          onClick={() => onUpdate(goal)}
          className="button-secondary text-sm flex-1"
        >
          Update Progress
        </button>
      </div>
    </motion.div>
  );

  const EmptyState = ({ title, description, icon: Icon, actionText, actionLink, suggestions = [] }) => (
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
        <Icon className="w-12 h-12 text-white" />
      </motion.div>
      
      <h3 className="text-2xl font-bold text-text-primary mb-4">{title}</h3>
      <p className="text-text-muted mb-8 max-w-md mx-auto">{description}</p>
      
      {suggestions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto mb-8">
          {suggestions.map((suggestion, index) => (
            <motion.div
              key={index}
              className="bg-background-medium/30 rounded-lg p-4 border border-secondary-700/50"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className={`w-10 h-10 ${suggestion.bgColor} rounded-lg flex items-center justify-center mx-auto mb-3`}>
                <suggestion.icon className={`w-5 h-5 ${suggestion.color}`} />
              </div>
              <h4 className="text-sm font-medium text-text-primary mb-1">{suggestion.title}</h4>
              <p className="text-xs text-text-muted">{suggestion.description}</p>
            </motion.div>
          ))}
        </div>
      )}
      
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        {actionText && actionLink && (
          <Link to={actionLink} className="button-primary">
            <PlusIcon className="w-5 h-5 mr-2" />
            {actionText}
          </Link>
        )}
        <button 
          onClick={() => setShowCreateModal(true)}
          className="button-secondary"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Create Custom Goal
        </button>
      </div>
    </motion.div>
  );

  const TabButton = ({ id, label, icon: Icon, isActive, onClick, count = 0 }) => (
    <motion.button
      onClick={() => onClick(id)}
      className={`flex items-center space-x-3 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
        isActive
          ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
          : 'text-text-muted hover:text-text-primary hover:bg-background-medium/50'
      }`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Icon className="w-5 h-5" />
      <span>{label}</span>
      <span className={`text-xs px-2 py-1 rounded-full ${
        isActive ? 'bg-primary-500/30' : 'bg-background-medium'
      }`}>
        {count}
      </span>
    </motion.button>
  );

  const nutritionSuggestions = [
    {
      title: "Daily Calories",
      description: "Track your caloric intake",
      icon: FireIcon,
      color: "text-primary-400",
      bgColor: "bg-primary-500/20"
    },
    {
      title: "Protein Target",
      description: "Meet your protein goals",
      icon: ChartBarIcon,
      color: "text-secondary-400",
      bgColor: "bg-secondary-500/20"
    },
    {
      title: "Water Intake",
      description: "Stay hydrated daily",
      icon: ClockIcon,
      color: "text-primary-400",
      bgColor: "bg-primary-500/20"
    }
  ];

  const fitnessSuggestions = [
    {
      title: "Target Weight",
      description: "Set your ideal weight",
      icon: ScaleIcon,
      color: "text-secondary-400",
      bgColor: "bg-secondary-500/20"
    },
    {
      title: "Weekly Goal",
      description: "Track weekly progress",
      icon: FlagIcon,
      color: "text-primary-400",
      bgColor: "bg-primary-500/20"
    },
    {
      title: "Habit Building",
      description: "Build healthy routines",
      icon: TrophyIcon,
      color: "text-secondary-400",
      bgColor: "bg-secondary-500/20"
    }
  ];

  const customSuggestions = [
    {
      title: "Meal Logging",
      description: "Track meals consistently",
      icon: CameraIcon,
      color: "text-primary-400",
      bgColor: "bg-primary-500/20"
    },
    {
      title: "Weekly Streak",
      description: "Build tracking habits",
      icon: ClockIcon,
      color: "text-secondary-400",
      bgColor: "bg-secondary-500/20"
    },
    {
      title: "Achievement",
      description: "Set personal milestones",
      icon: TrophyIcon,
      color: "text-primary-400",
      bgColor: "bg-primary-500/20"
    }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'nutrition':
        if (nutritionGoals.length === 0) {
          return (
            <EmptyState
              title="No nutrition goals set"
              description="Set daily nutrition targets like calories, protein, and water intake to track your dietary progress."
              icon={FireIcon}
              actionText="Set Nutrition Goals"
              actionLink="/profile"
              suggestions={nutritionSuggestions}
            />
          );
        }
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {nutritionGoals.map((goal) => (
              <GoalCard key={goal.id} goal={goal} />
            ))}
          </div>
        );
      case 'fitness':
        if (fitnessGoals.length === 0) {
          return (
            <EmptyState
              title="No fitness goals set"
              description="Define your fitness objectives like target weight, weekly deficits, or exercise milestones."
              icon={ScaleIcon}
              actionText="Set Fitness Goals"
              actionLink="/profile"
              suggestions={fitnessSuggestions}
            />
          );
        }
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {fitnessGoals.map((goal) => (
              <GoalCard key={goal.id} goal={goal} />
            ))}
          </div>
        );
      case 'custom':
        if (customGoals.length === 0) {
          return (
            <EmptyState
              title="No custom goals created"
              description="Create personalized goals that matter to you - habit building, streaks, or personal challenges."
              icon={TrophyIcon}
              suggestions={customSuggestions}
            />
          );
        }
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {customGoals.map((goal) => (
              <GoalCard key={goal.id} goal={goal} />
            ))}
          </div>
        );
      default:
        return null;
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
      <motion.div
        className="flex flex-col md:flex-row md:items-center justify-between gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Goals</h1>
          <p className="text-text-muted">Set and track your health and fitness objectives</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Link to="/profile" className="button-secondary">
            <Cog6ToothIcon className="w-5 h-5 mr-2" />
            Profile Setup
          </Link>
          <Link to="/scan" className="button-primary">
            <CameraIcon className="w-5 h-5 mr-2" />
            Start Tracking
          </Link>
        </div>
      </motion.div>

      {/* Overview Stats */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="card text-center">
          <div className="w-12 h-12 bg-primary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
            <TrophyIcon className="w-6 h-6 text-primary-400" />
          </div>
          <div className="text-2xl font-bold text-text-primary">
            {nutritionGoals.length + fitnessGoals.length + customGoals.length}
          </div>
          <div className="text-sm text-text-muted">Active Goals</div>
          <div className="text-xs text-text-muted mt-1">
            {nutritionGoals.length + fitnessGoals.length + customGoals.length === 0 ? 'Start setting goals' : 'Keep going!'}
          </div>
        </div>
        
        <div className="card text-center">
          <div className="w-12 h-12 bg-secondary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
            <FlagIcon className="w-6 h-6 text-secondary-400" />
          </div>
          <div className="text-2xl font-bold text-text-primary">
            {(() => {
              const allGoals = [...nutritionGoals, ...fitnessGoals, ...customGoals];
              if (allGoals.length === 0) return '0%';
              const avgProgress = allGoals.reduce((sum, goal) => sum + goal.progress, 0) / allGoals.length;
              return `${avgProgress.toFixed(0)}%`;
            })()}
          </div>
          <div className="text-sm text-text-muted">Avg Progress</div>
          <div className="text-xs text-text-muted mt-1">Track your journey</div>
        </div>
        
        <div className="card text-center">
          <div className="w-12 h-12 bg-primary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
            <ClockIcon className="w-6 h-6 text-primary-400" />
          </div>
          <div className="text-2xl font-bold text-text-primary">
            {[...nutritionGoals, ...fitnessGoals, ...customGoals].filter(goal => goal.progress >= 100).length}
          </div>
          <div className="text-sm text-text-muted">Completed</div>
          <div className="text-xs text-text-muted mt-1">Achieve your targets</div>
        </div>
      </motion.div>

      {/* Tab Navigation */}
      <motion.div
        className="flex flex-wrap gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <TabButton
          id="nutrition"
          label="Nutrition"
          icon={FireIcon}
          isActive={activeTab === 'nutrition'}
          onClick={setActiveTab}
          count={nutritionGoals.length}
        />
        <TabButton
          id="fitness"
          label="Fitness"
          icon={ScaleIcon}
          isActive={activeTab === 'fitness'}
          onClick={setActiveTab}
          count={fitnessGoals.length}
        />
        <TabButton
          id="custom"
          label="Custom"
          icon={TrophyIcon}
          isActive={activeTab === 'custom'}
          onClick={setActiveTab}
          count={customGoals.length}
        />
      </motion.div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        {renderTabContent()}
      </motion.div>

      {/* Create Goal Modal */}
      {showCreateModal && <CreateGoalModal />}
    </motion.div>
  );
};

export default Goals;
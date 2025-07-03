import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  TrophyIcon,
  ChartBarIcon,
  CalendarDaysIcon,
  ScaleIcon,
  CameraIcon,
  PlusIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';
import { Link } from 'react-router-dom';

const Progress = () => {
  const [timeRange, setTimeRange] = useState('7d');
  
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
      
      <h3 className="text-2xl font-bold text-text-primary mb-4">Start tracking to see progress</h3>
      <p className="text-text-muted mb-8 max-w-md mx-auto">
        Log your meals and set up your profile to start visualizing your health journey with detailed charts and insights.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Link to="/scan" className="button-primary">
          <CameraIcon className="w-5 h-5 mr-2" />
          Start Logging Meals
        </Link>
        <Link to="/profile" className="button-secondary">
          <Cog6ToothIcon className="w-5 h-5 mr-2" />
          Set Up Profile
        </Link>
      </div>
    </motion.div>
  );

  const StatsGrid = () => (
    <motion.div
      className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <div className="card text-center">
        <div className="w-12 h-12 bg-primary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
          <ScaleIcon className="w-6 h-6 text-primary-400" />
        </div>
        <div className="text-2xl font-bold text-text-primary">--</div>
        <div className="text-sm text-text-muted">Current Weight</div>
        <div className="text-xs text-text-muted mt-1">Set up in profile</div>
      </div>
      
      <div className="card text-center">
        <div className="w-12 h-12 bg-secondary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
          <TrophyIcon className="w-6 h-6 text-secondary-400" />
        </div>
        <div className="text-2xl font-bold text-text-primary">0</div>
        <div className="text-sm text-text-muted">Days Logged</div>
        <div className="text-xs text-text-muted mt-1">Start scanning meals</div>
      </div>
      
      <div className="card text-center">
        <div className="w-12 h-12 bg-primary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
          <ChartBarIcon className="w-6 h-6 text-primary-400" />
        </div>
        <div className="text-2xl font-bold text-text-primary">0</div>
        <div className="text-sm text-text-muted">Avg Calories</div>
        <div className="text-xs text-text-muted mt-1">Track your intake</div>
      </div>
      
      <div className="card text-center">
        <div className="w-12 h-12 bg-secondary-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
          <CalendarDaysIcon className="w-6 h-6 text-secondary-400" />
        </div>
        <div className="text-2xl font-bold text-text-primary">0</div>
        <div className="text-sm text-text-muted">Streak Days</div>
        <div className="text-xs text-text-muted mt-1">Build your habit</div>
      </div>
    </motion.div>
  );

  const ChartsPlaceholder = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {/* Weight Progress Chart */}
      <motion.div
        className="card"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">Weight Progress</h3>
          <ScaleIcon className="w-5 h-5 text-primary-400" />
        </div>
        
        <div className="flex items-center justify-center h-64 border-2 border-dashed border-secondary-700 rounded-lg">
          <div className="text-center">
            <ScaleIcon className="w-12 h-12 text-text-muted mx-auto mb-2" />
            <p className="text-text-muted text-sm">Weight data will appear here</p>
            <Link to="/profile" className="text-primary-400 text-sm hover:text-primary-300">
              Add your current weight
            </Link>
          </div>
        </div>
      </motion.div>

      {/* Calorie Trends Chart */}
      <motion.div
        className="card"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">Calorie Trends</h3>
          <ChartBarIcon className="w-5 h-5 text-primary-400" />
        </div>
        
        <div className="flex items-center justify-center h-64 border-2 border-dashed border-secondary-700 rounded-lg">
          <div className="text-center">
            <ChartBarIcon className="w-12 h-12 text-text-muted mx-auto mb-2" />
            <p className="text-text-muted text-sm">Calorie trends will appear here</p>
            <Link to="/scan" className="text-primary-400 text-sm hover:text-primary-300">
              Start logging meals
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const AchievementsPlaceholder = () => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-text-primary">Achievements</h3>
        <TrophyIcon className="w-5 h-5 text-primary-400" />
      </div>
      
      <div className="text-center py-12">
        <TrophyIcon className="w-16 h-16 text-text-muted mx-auto mb-4 opacity-50" />
        <h4 className="text-lg font-medium text-text-primary mb-2">Unlock your first achievement</h4>
        <p className="text-text-muted text-sm mb-6">
          Start logging meals and tracking your progress to earn badges and celebrate milestones.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-lg mx-auto">
          <div className="bg-background-medium/30 rounded-lg p-4 border border-secondary-700/50">
            <div className="w-8 h-8 bg-primary-500/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <CameraIcon className="w-4 h-4 text-primary-400" />
            </div>
            <p className="text-xs text-text-muted">First Scan</p>
          </div>
          
          <div className="bg-background-medium/30 rounded-lg p-4 border border-secondary-700/50">
            <div className="w-8 h-8 bg-secondary-500/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <CalendarDaysIcon className="w-4 h-4 text-secondary-400" />
            </div>
            <p className="text-xs text-text-muted">7-Day Streak</p>
          </div>
          
          <div className="bg-background-medium/30 rounded-lg p-4 border border-secondary-700/50">
            <div className="w-8 h-8 bg-primary-500/20 rounded-full flex items-center justify-center mx-auto mb-2">
              <TrophyIcon className="w-4 h-4 text-primary-400" />
            </div>
            <p className="text-xs text-text-muted">Goal Reached</p>
          </div>
        </div>
      </div>
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
      <motion.div
        className="flex flex-col md:flex-row md:items-center justify-between gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Progress</h1>
          <p className="text-text-muted">Track your health journey and achievements</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="input-field text-sm"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 3 months</option>
            <option value="1y">Last year</option>
          </select>
          <Link to="/goals" className="button-primary">
            <TrophyIcon className="w-5 h-5 mr-2" />
            Set Goals
          </Link>
        </div>
      </motion.div>

      {/* Stats Overview */}
      <StatsGrid />

      {/* Charts Section */}
      <ChartsPlaceholder />

      {/* Achievements Section */}
      <AchievementsPlaceholder />

      {/* Main Empty State */}
      <EmptyState />
    </motion.div>
  );
};

export default Progress;
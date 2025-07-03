import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CameraIcon,
  FireIcon,
  ChartBarIcon,
  TrophyIcon,
  HeartIcon,
  CalendarDaysIcon,
  BeakerIcon,
  PlusIcon,
  ArrowRightIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  const [userName] = useState(''); // Empty for new user

  const EmptyStateCard = ({ title, description, icon: Icon, actionText, actionLink, gradient = "from-primary-500 to-primary-600" }) => (
    <motion.div
      className="card text-center"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className={`w-16 h-16 bg-gradient-to-br ${gradient} rounded-full flex items-center justify-center mx-auto mb-4 opacity-80`}>
        <Icon className="w-8 h-8 text-white" />
      </div>
      <h3 className="text-lg font-semibold text-text-primary mb-2">{title}</h3>
      <p className="text-sm text-text-muted mb-4">{description}</p>
      <Link 
        to={actionLink}
        className="inline-flex items-center justify-center space-x-2 button-primary text-sm py-2 px-4"
      >
        <span>{actionText}</span>
        <ArrowRightIcon className="w-4 h-4" />
      </Link>
    </motion.div>
  );

  const WelcomeHero = () => (
    <motion.div
      className="card bg-gradient-to-br from-primary-500/10 to-secondary-500/10 border-primary-500/20 text-center py-12"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <motion.div
        className="w-24 h-24 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto mb-6"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
      >
        <CameraIcon className="w-12 h-12 text-white" />
      </motion.div>
      
      <motion.h1 
        className="text-4xl font-bold text-text-primary mb-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        Welcome to NutriScan! 🌟
      </motion.h1>
      
      <motion.p 
        className="text-lg text-text-muted mb-8 max-w-2xl mx-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        Your intelligent nutrition tracking companion powered by AI. Start your health journey by scanning your first meal or setting up your profile.
      </motion.p>
      
      <motion.div 
        className="flex flex-col sm:flex-row gap-4 justify-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Link to="/scan" className="button-primary">
          <CameraIcon className="w-5 h-5 mr-2" />
          Scan Your First Meal
        </Link>
        <Link to="/profile" className="button-secondary">
          <Cog6ToothIcon className="w-5 h-5 mr-2" />
          Set Up Profile
        </Link>
      </motion.div>
    </motion.div>
  );

  const QuickStartSection = () => (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7 }}
    >
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-text-primary">Quick Start Guide</h2>
        <div className="flex items-center space-x-2 text-primary-400">
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
          <span className="text-sm">Get started in 3 steps</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <EmptyStateCard
          title="Scan Food"
          description="Use AI to instantly analyze and log your meals with detailed nutritional information"
          icon={CameraIcon}
          actionText="Start Scanning"
          actionLink="/scan"
          gradient="from-primary-500 to-primary-600"
        />
        
        <EmptyStateCard
          title="Set Goals"
          description="Define your nutrition and fitness goals to get personalized recommendations"
          icon={TrophyIcon}
          actionText="Set Goals"
          actionLink="/goals"
          gradient="from-secondary-500 to-secondary-600"
        />
        
        <EmptyStateCard
          title="Track Progress"
          description="Monitor your daily intake, weight changes, and achievement milestones"
          icon={ChartBarIcon}
          actionText="View Progress"
          actionLink="/progress"
          gradient="from-primary-600 to-secondary-500"
        />
      </div>
    </motion.div>
  );

  const FeaturesPreview = () => (
    <motion.div
      className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.8 }}
    >
      {/* AI Features */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
            <BeakerIcon className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-lg font-semibold text-text-primary">AI-Powered Analysis</h3>
        </div>
        <div className="space-y-3 text-sm text-text-muted">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
            <span>Instant food recognition from photos</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
            <span>Automatic calorie and macro calculation</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
            <span>Smart portion size estimation</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
            <span>Nutritional insights and alternatives</span>
          </div>
        </div>
      </div>

      {/* Health Tracking */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-gradient-to-br from-secondary-500 to-secondary-600 rounded-lg flex items-center justify-center">
            <HeartIcon className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-lg font-semibold text-text-primary">Comprehensive Tracking</h3>
        </div>
        <div className="space-y-3 text-sm text-text-muted">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-secondary-500 rounded-full"></div>
            <span>Daily calorie and macro monitoring</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-secondary-500 rounded-full"></div>
            <span>Weight loss progress visualization</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-secondary-500 rounded-full"></div>
            <span>Achievement badges and streaks</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-secondary-500 rounded-full"></div>
            <span>Personalized health insights</span>
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
      {/* Welcome Hero Section */}
      <WelcomeHero />

      {/* Quick Start Guide */}
      <QuickStartSection />

      {/* Features Preview */}
      <FeaturesPreview />

      {/* Call to Action */}
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
      >
        <p className="text-text-muted mb-4">
          Ready to transform your nutrition habits with AI?
        </p>
        <Link 
          to="/scan"
          className="inline-flex items-center space-x-2 button-primary text-lg px-8 py-4"
        >
          <CameraIcon className="w-6 h-6" />
          <span>Start Your Journey</span>
        </Link>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;
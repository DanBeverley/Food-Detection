import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  UserIcon,
  Cog6ToothIcon,
  CameraIcon,
  ScaleIcon,
  HeartIcon,
  CalculatorIcon,
  FireIcon,
  TrophyIcon
} from '@heroicons/react/24/outline';

const Profile = () => {
  const [activeSection, setActiveSection] = useState('profile');
  
  // Empty state for new user
  const [userProfile, setUserProfile] = useState({
    name: '',
    email: '',
    age: '',
    height: '', // inches
    weight: '',
    activityLevel: 'moderately_active',
    goal: 'lose_weight',
    weeklyGoal: 1, // pounds per week
    profilePicture: null
  });

  const [notifications, setNotifications] = useState({
    mealReminders: true,
    goalAchievements: true,
    weeklyReports: false
  });

  const [privacy, setPrivacy] = useState({
    profileVisibility: 'private',
    shareProgress: false,
    dataCollection: false
  });

  // Debounced update function to prevent excessive re-renders
  const updateProfile = useCallback((field, value) => {
    setUserProfile(prev => ({ ...prev, [field]: value }));
  }, []);

  // Calculate BMI and daily calorie needs only if data exists
  const calculateMetrics = () => {
    if (!userProfile.height || !userProfile.weight || !userProfile.age) {
      return { bmi: null, calories: null };
    }

    const heightInMeters = (parseFloat(userProfile.height) * 2.54) / 100;
    const weightInKg = parseFloat(userProfile.weight) * 0.453592;
    const bmi = weightInKg / (heightInMeters * heightInMeters);
    
    // Harris-Benedict Equation (simplified for demo)
    let bmr = 88.362 + (13.397 * weightInKg) + (4.799 * (parseFloat(userProfile.height) * 2.54)) - (5.677 * parseFloat(userProfile.age));
    
    const activityMultipliers = {
      sedentary: 1.2,
      lightly_active: 1.375,
      moderately_active: 1.55,
      very_active: 1.725,
      extremely_active: 1.9
    };
    
    const tdee = bmr * activityMultipliers[userProfile.activityLevel];
    
    // Adjust for goal
    let calories = tdee;
    if (userProfile.goal === 'lose_weight') {
      calories = tdee - (userProfile.weeklyGoal * 500);
    } else if (userProfile.goal === 'gain_weight') {
      calories = tdee + 500;
    }
    
    return { bmi: bmi.toFixed(1), calories: Math.round(calories) };
  };

  const { bmi, calories } = calculateMetrics();

  const EmptyProfileState = () => (
    <motion.div
      className="text-center py-16"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="w-24 h-24 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto mb-6">
        <UserIcon className="w-12 h-12 text-white" />
      </div>
      <h3 className="text-2xl font-bold text-text-primary mb-4">Complete your profile</h3>
      <p className="text-text-muted mb-8 max-w-md mx-auto">
        Add your basic information to get personalized calorie recommendations and track your health journey.
      </p>
    </motion.div>
  );

  const ProfileSection = () => {
    const hasBasicInfo = userProfile.name || userProfile.email || userProfile.age || userProfile.height || userProfile.weight;

    if (!hasBasicInfo) {
      return <EmptyProfileState />;
    }

    return (
      <div className="space-y-6">
        {/* Profile Picture and Basic Info */}
        <motion.div
          className="card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center space-x-6">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center">
                {userProfile.profilePicture ? (
                  <img 
                    src={userProfile.profilePicture} 
                    alt="Profile" 
                    className="w-24 h-24 rounded-full object-cover"
                  />
                ) : (
                  <UserIcon className="w-12 h-12 text-white" />
                )}
              </div>
              <button className="absolute -bottom-2 -right-2 p-2 bg-primary-500 rounded-full hover:bg-primary-600 transition-colors">
                <CameraIcon className="w-4 h-4 text-white" />
              </button>
            </div>
            
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-text-primary">
                {userProfile.name || 'Your Name'}
              </h2>
              <p className="text-text-muted">{userProfile.email || 'your.email@example.com'}</p>
              <div className="mt-2 flex items-center space-x-4 text-sm text-text-muted">
                {userProfile.age && <span>Age: {userProfile.age}</span>}
                {userProfile.height && (
                  <span>Height: {Math.floor(userProfile.height / 12)}'{userProfile.height % 12}"</span>
                )}
                {userProfile.weight && <span>Weight: {userProfile.weight} lbs</span>}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Health Metrics */}
        {(bmi || calories) && (
          <motion.div
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-lg font-semibold text-text-primary mb-4">Health Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-background-medium/30 rounded-lg">
                <CalculatorIcon className="w-8 h-8 text-secondary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-text-primary">{bmi || '--'}</p>
                <p className="text-sm text-text-muted">BMI</p>
                {bmi && (
                  <p className="text-xs text-text-muted mt-1">
                    {parseFloat(bmi) < 18.5 ? 'Underweight' : 
                     parseFloat(bmi) < 25 ? 'Normal' : 
                     parseFloat(bmi) < 30 ? 'Overweight' : 'Obese'}
                  </p>
                )}
              </div>
              
              <div className="text-center p-4 bg-background-medium/30 rounded-lg">
                <FireIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-text-primary">{calories || '--'}</p>
                <p className="text-sm text-text-muted">Daily Calories</p>
                <p className="text-xs text-text-muted mt-1">Recommended intake</p>
              </div>
              
              <div className="text-center p-4 bg-background-medium/30 rounded-lg">
                <TrophyIcon className="w-8 h-8 text-secondary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-text-primary">{userProfile.weeklyGoal}</p>
                <p className="text-sm text-text-muted">lbs/week</p>
                <p className="text-xs text-text-muted mt-1">Weight goal</p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    );
  };

  const PersonalInfoForm = () => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-6">Personal Information</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Full Name</label>
          <input
            type="text"
            value={userProfile.name}
            onChange={(e) => updateProfile('name', e.target.value)}
            className="input-field w-full"
            placeholder="Enter your full name"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Email</label>
          <input
            type="email"
            value={userProfile.email}
            onChange={(e) => updateProfile('email', e.target.value)}
            className="input-field w-full"
            placeholder="Enter your email"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Age</label>
          <input
            type="number"
            value={userProfile.age}
            onChange={(e) => updateProfile('age', e.target.value)}
            className="input-field w-full"
            placeholder="Enter your age"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Height (inches)</label>
          <input
            type="number"
            value={userProfile.height}
            onChange={(e) => updateProfile('height', e.target.value)}
            className="input-field w-full"
            placeholder="Enter height in inches"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Weight (lbs)</label>
          <input
            type="number"
            step="0.1"
            value={userProfile.weight}
            onChange={(e) => updateProfile('weight', e.target.value)}
            className="input-field w-full"
            placeholder="Enter current weight"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Activity Level</label>
          <select
            value={userProfile.activityLevel}
            onChange={(e) => updateProfile('activityLevel', e.target.value)}
            className="input-field w-full"
          >
            <option value="sedentary">Sedentary (desk job)</option>
            <option value="lightly_active">Lightly Active (light exercise)</option>
            <option value="moderately_active">Moderately Active (moderate exercise)</option>
            <option value="very_active">Very Active (hard exercise)</option>
            <option value="extremely_active">Extremely Active (very hard exercise)</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Goal</label>
          <select
            value={userProfile.goal}
            onChange={(e) => updateProfile('goal', e.target.value)}
            className="input-field w-full"
          >
            <option value="lose_weight">Lose Weight</option>
            <option value="maintain_weight">Maintain Weight</option>
            <option value="gain_weight">Gain Weight</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">Weekly Goal (lbs)</label>
          <select
            value={userProfile.weeklyGoal}
            onChange={(e) => updateProfile('weeklyGoal', parseFloat(e.target.value))}
            className="input-field w-full"
          >
            <option value={0.5}>0.5 lbs/week (slow)</option>
            <option value={1}>1 lb/week (moderate)</option>
            <option value={1.5}>1.5 lbs/week (fast)</option>
            <option value={2}>2 lbs/week (aggressive)</option>
          </select>
        </div>
      </div>
    </motion.div>
  );

  const NotificationSettings = () => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-6">Notification Preferences</h3>
      <div className="space-y-4">
        {Object.entries(notifications).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between">
            <div>
              <p className="text-text-primary font-medium">
                {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </p>
              <p className="text-sm text-text-muted">
                {key === 'mealReminders' && 'Get reminders to log your meals'}
                {key === 'goalAchievements' && 'Celebrate when you reach your goals'}
                {key === 'weeklyReports' && 'Weekly progress summaries'}
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => setNotifications(prev => ({ ...prev, [key]: e.target.checked }))}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-secondary-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>
        ))}
      </div>
    </motion.div>
  );

  const PrivacySettings = () => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-6">Privacy Settings</h3>
      <div className="space-y-4">
        {Object.entries(privacy).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between">
            <div>
              <p className="text-text-primary font-medium">
                {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </p>
              <p className="text-sm text-text-muted">
                {key === 'profileVisibility' && 'Control who can see your profile'}
                {key === 'shareProgress' && 'Allow sharing progress with friends'}
                {key === 'dataCollection' && 'Help improve our AI with anonymous data'}
              </p>
            </div>
            {key === 'profileVisibility' ? (
              <select
                value={value}
                onChange={(e) => setPrivacy(prev => ({ ...prev, [key]: e.target.value }))}
                className="input-field text-sm"
              >
                <option value="private">Private</option>
                <option value="friends">Friends Only</option>
                <option value="public">Public</option>
              </select>
            ) : (
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={value}
                  onChange={(e) => setPrivacy(prev => ({ ...prev, [key]: e.target.checked }))}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-secondary-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
              </label>
            )}
          </div>
        ))}
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
          <h1 className="text-3xl font-bold text-text-primary">Profile & Settings</h1>
          <p className="text-text-muted">Manage your account and preferences</p>
        </div>
      </motion.div>

      {/* Tab Navigation */}
      <motion.div
        className="flex flex-wrap gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {[
          { id: 'profile', label: 'Profile', icon: UserIcon },
          { id: 'notifications', label: 'Notifications', icon: Cog6ToothIcon },
          { id: 'privacy', label: 'Privacy', icon: Cog6ToothIcon }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveSection(tab.id)}
            className={`flex items-center space-x-3 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
              activeSection === tab.id
                ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                : 'text-text-muted hover:text-text-primary hover:bg-background-medium/50'
            }`}
          >
            <tab.icon className="w-5 h-5" />
            <span>{tab.label}</span>
          </button>
        ))}
      </motion.div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeSection === 'profile' && (
          <>
            <ProfileSection />
            <PersonalInfoForm />
          </>
        )}
        {activeSection === 'notifications' && <NotificationSettings />}
        {activeSection === 'privacy' && <PrivacySettings />}
      </div>
    </motion.div>
  );
};

export default Profile;
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  UserIcon,
  Cog6ToothIcon,
  BellIcon,
  ShieldCheckIcon,
  CameraIcon,
  PencilIcon,
  ChartBarIcon,
  ScaleIcon,
  HeartIcon,
  CalculatorIcon
} from '@heroicons/react/24/outline';

const Profile = () => {
  const [activeSection, setActiveSection] = useState('profile');
  const [userProfile, setUserProfile] = useState({
    name: 'Alex Johnson',
    email: 'alex.johnson@email.com',
    age: 28,
    height: 70, // inches
    weight: 165.2,
    activityLevel: 'moderately_active',
    goal: 'lose_weight',
    weeklyGoal: 1, // pounds per week
    profilePicture: null
  });

  const [notifications, setNotifications] = useState({
    mealReminders: true,
    goalAchievements: true,
    weeklyReports: true,
    marketingEmails: false
  });

  const [privacy, setPrivacy] = useState({
    profileVisibility: 'private',
    shareProgress: false,
    dataCollection: true
  });

  // Calculate BMI and daily calorie needs
  const heightInMeters = (userProfile.height * 2.54) / 100;
  const weightInKg = userProfile.weight * 0.453592;
  const bmi = weightInKg / (heightInMeters * heightInMeters);
  
  const calculateCalories = () => {
    // Harris-Benedict Equation (simplified for demo)
    let bmr;
    // Assuming male for demo - in real app, would have gender field
    bmr = 88.362 + (13.397 * weightInKg) + (4.799 * (userProfile.height * 2.54)) - (5.677 * userProfile.age);
    
    const activityMultipliers = {
      sedentary: 1.2,
      lightly_active: 1.375,
      moderately_active: 1.55,
      very_active: 1.725,
      extremely_active: 1.9
    };
    
    const tdee = bmr * activityMultipliers[userProfile.activityLevel];
    
    // Adjust for goal
    if (userProfile.goal === 'lose_weight') {
      return Math.round(tdee - (userProfile.weeklyGoal * 500)); // 500 cal deficit per lb
    } else if (userProfile.goal === 'gain_weight') {
      return Math.round(tdee + 500);
    }
    return Math.round(tdee);
  };

  const ProfileSection = () => (
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
            <h2 className="text-2xl font-bold text-text-primary">{userProfile.name}</h2>
            <p className="text-text-muted">{userProfile.email}</p>
            <div className="mt-2 flex items-center space-x-4 text-sm text-text-muted">
              <span>Age: {userProfile.age}</span>
              <span>Height: {Math.floor(userProfile.height / 12)}'{userProfile.height % 12}"</span>
              <span>Weight: {userProfile.weight} lbs</span>
            </div>
          </div>
          
          <button className="button-secondary">
            <PencilIcon className="w-4 h-4 mr-2" />
            Edit Profile
          </button>
        </div>
      </motion.div>

      {/* Health Metrics */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">Health Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-background-medium/30 rounded-lg">
            <CalculatorIcon className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">{bmi.toFixed(1)}</p>
            <p className="text-sm text-text-muted">BMI</p>
            <p className="text-xs text-text-muted mt-1">
              {bmi < 18.5 ? 'Underweight' : 
               bmi < 25 ? 'Normal' : 
               bmi < 30 ? 'Overweight' : 'Obese'}
            </p>
          </div>
          
          <div className="text-center p-4 bg-background-medium/30 rounded-lg">
            <HeartIcon className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">{calculateCalories()}</p>
            <p className="text-sm text-text-muted">Daily Calories</p>
            <p className="text-xs text-text-muted mt-1">Recommended intake</p>
          </div>
          
          <div className="text-center p-4 bg-background-medium/30 rounded-lg">
            <ScaleIcon className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <p className="text-2xl font-bold text-text-primary">{userProfile.weeklyGoal}</p>
            <p className="text-sm text-text-muted">lbs/week</p>
            <p className="text-xs text-text-muted mt-1">Weight goal</p>
          </div>
        </div>
      </motion.div>

      {/* Personal Information Form */}
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
              onChange={(e) => setUserProfile({...userProfile, name: e.target.value})}
              className="input-field w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Email</label>
            <input
              type="email"
              value={userProfile.email}
              onChange={(e) => setUserProfile({...userProfile, email: e.target.value})}
              className="input-field w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Age</label>
            <input
              type="number"
              value={userProfile.age}
              onChange={(e) => setUserProfile({...userProfile, age: parseInt(e.target.value)})}
              className="input-field w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Height (inches)</label>
            <input
              type="number"
              value={userProfile.height}
              onChange={(e) => setUserProfile({...userProfile, height: parseInt(e.target.value)})}
              className="input-field w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Current Weight (lbs)</label>
            <input
              type="number"
              step="0.1"
              value={userProfile.weight}
              onChange={(e) => setUserProfile({...userProfile, weight: parseFloat(e.target.value)})}
              className="input-field w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">Weekly Goal (lbs)</label>
            <select
              value={userProfile.weeklyGoal}
              onChange={(e) => setUserProfile({...userProfile, weeklyGoal: parseFloat(e.target.value)})}
              className="input-field w-full"
            >
              <option value={0.5}>Lose 0.5 lbs/week</option>
              <option value={1}>Lose 1 lb/week</option>
              <option value={1.5}>Lose 1.5 lbs/week</option>
              <option value={2}>Lose 2 lbs/week</option>
              <option value={0}>Maintain weight</option>
              <option value={-0.5}>Gain 0.5 lbs/week</option>
              <option value={-1}>Gain 1 lb/week</option>
            </select>
          </div>
        </div>
      </motion.div>

      {/* Activity Level */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-6">Activity Level</h3>
        <div className="space-y-3">
          {[
            { value: 'sedentary', label: 'Sedentary', desc: 'Little or no exercise' },
            { value: 'lightly_active', label: 'Lightly Active', desc: 'Light exercise 1-3 days/week' },
            { value: 'moderately_active', label: 'Moderately Active', desc: 'Moderate exercise 3-5 days/week' },
            { value: 'very_active', label: 'Very Active', desc: 'Hard exercise 6-7 days/week' },
            { value: 'extremely_active', label: 'Extremely Active', desc: 'Very hard exercise, physical job' }
          ].map((activity) => (
            <label key={activity.value} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-background-medium/30 transition-colors cursor-pointer">
              <input
                type="radio"
                name="activityLevel"
                value={activity.value}
                checked={userProfile.activityLevel === activity.value}
                onChange={(e) => setUserProfile({...userProfile, activityLevel: e.target.value})}
                className="w-4 h-4 text-primary-600 bg-background-light border-secondary-600 focus:ring-primary-500 focus:ring-2"
              />
              <div>
                <p className="font-medium text-text-primary">{activity.label}</p>
                <p className="text-sm text-text-muted">{activity.desc}</p>
              </div>
            </label>
          ))}
        </div>
      </motion.div>
    </div>
  );

  const SettingsSection = () => (
    <div className="space-y-6">
      {/* Notifications */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-6">Notifications</h3>
        <div className="space-y-4">
          {Object.entries(notifications).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between">
              <div>
                <p className="font-medium text-text-primary">
                  {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </p>
                <p className="text-sm text-text-muted">
                  {key === 'mealReminders' && 'Get reminded to log your meals'}
                  {key === 'goalAchievements' && 'Celebrate when you reach your goals'}
                  {key === 'weeklyReports' && 'Receive weekly progress summaries'}
                  {key === 'marketingEmails' && 'Updates about new features and tips'}
                </p>
              </div>
              <motion.button
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  value ? 'bg-primary-500' : 'bg-secondary-700'
                }`}
                onClick={() => setNotifications({...notifications, [key]: !value})}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="absolute top-1 w-4 h-4 bg-white rounded-full shadow-md"
                  animate={{ x: value ? 24 : 4 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              </motion.button>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Privacy */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-6">Privacy & Data</h3>
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-text-primary mb-3">Profile Visibility</label>
            <div className="space-y-2">
              {[
                { value: 'public', label: 'Public', desc: 'Anyone can see your profile' },
                { value: 'friends', label: 'Friends Only', desc: 'Only your friends can see your profile' },
                { value: 'private', label: 'Private', desc: 'Only you can see your profile' }
              ].map((option) => (
                <label key={option.value} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-background-medium/30 transition-colors cursor-pointer">
                  <input
                    type="radio"
                    name="profileVisibility"
                    value={option.value}
                    checked={privacy.profileVisibility === option.value}
                    onChange={(e) => setPrivacy({...privacy, profileVisibility: e.target.value})}
                    className="w-4 h-4 text-primary-600 bg-background-light border-secondary-600 focus:ring-primary-500 focus:ring-2"
                  />
                  <div>
                    <p className="font-medium text-text-primary">{option.label}</p>
                    <p className="text-sm text-text-muted">{option.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-text-primary">Share Progress</p>
                <p className="text-sm text-text-muted">Allow others to see your fitness progress</p>
              </div>
              <motion.button
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  privacy.shareProgress ? 'bg-primary-500' : 'bg-secondary-700'
                }`}
                onClick={() => setPrivacy({...privacy, shareProgress: !privacy.shareProgress})}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="absolute top-1 w-4 h-4 bg-white rounded-full shadow-md"
                  animate={{ x: privacy.shareProgress ? 24 : 4 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              </motion.button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-text-primary">Data Collection</p>
                <p className="text-sm text-text-muted">Help improve our service with anonymous usage data</p>
              </div>
              <motion.button
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  privacy.dataCollection ? 'bg-primary-500' : 'bg-secondary-700'
                }`}
                onClick={() => setPrivacy({...privacy, dataCollection: !privacy.dataCollection})}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="absolute top-1 w-4 h-4 bg-white rounded-full shadow-md"
                  animate={{ x: privacy.dataCollection ? 24 : 4 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              </motion.button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Advanced Features */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-6">Advanced Features</h3>
        <div className="space-y-4">
          <div className="p-4 border border-secondary-800/30 rounded-lg">
            <h4 className="font-medium text-text-primary mb-2">Custom Model Training</h4>
            <p className="text-sm text-text-muted mb-4">
              Train your own food recognition model with your specific dietary needs
            </p>
            <button className="button-secondary text-sm">
              Enable Advanced Features
            </button>
          </div>
          
          <div className="p-4 border border-secondary-800/30 rounded-lg">
            <h4 className="font-medium text-text-primary mb-2">API Access</h4>
            <p className="text-sm text-text-muted mb-4">
              Integrate NutriScan with your existing fitness apps and devices
            </p>
            <button className="button-secondary text-sm">
              Generate API Key
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const sections = [
    { id: 'profile', name: 'Profile', icon: UserIcon },
    { id: 'settings', name: 'Settings', icon: Cog6ToothIcon }
  ];

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-text-primary mb-2">Profile & Settings ⚙️</h1>
        <p className="text-text-muted">Manage your account and personalize your experience</p>
      </div>

      {/* Section Tabs */}
      <motion.div
        className="flex space-x-1 bg-background-medium/30 rounded-xl p-1"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {sections.map((section) => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className={`flex items-center space-x-2 flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeSection === section.id
                ? 'bg-primary-500 text-white shadow-lg'
                : 'text-text-muted hover:text-text-primary hover:bg-secondary-800/50'
            }`}
          >
            <section.icon className="w-4 h-4" />
            <span>{section.name}</span>
          </button>
        ))}
      </motion.div>

      {/* Content */}
      <motion.div
        key={activeSection}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
      >
        {activeSection === 'profile' && <ProfileSection />}
        {activeSection === 'settings' && <SettingsSection />}
      </motion.div>

      {/* Save Button */}
      <motion.div
        className="flex justify-end pt-6 border-t border-secondary-800/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <button className="button-primary">
          Save Changes
        </button>
      </motion.div>
    </motion.div>
  );
};

export default Profile;
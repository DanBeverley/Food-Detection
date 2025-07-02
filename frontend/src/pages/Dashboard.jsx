import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  CameraIcon,
  FireIcon,
  ChartBarIcon,
  ClockIcon,
  TrophyIcon,
  HeartIcon,
  ArrowTrendingUpIcon,
  CalendarDaysIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Dashboard = () => {
  const [userStats] = useState({
    dailyCalories: 1847,
    dailyGoal: 2500,
    protein: 125,
    proteinGoal: 150,
    carbs: 230,
    carbsGoal: 300,
    fat: 68,
    fatGoal: 85,
    water: 6,
    waterGoal: 8,
    streak: 12,
    weight: 165.2,
    weightGoal: 160
  });

  const [weeklyData] = useState([
    { day: 'Mon', calories: 2234, goal: 2500, weight: 166.1 },
    { day: 'Tue', calories: 2156, goal: 2500, weight: 165.8 },
    { day: 'Wed', calories: 2089, goal: 2500, weight: 165.6 },
    { day: 'Thu', calories: 2312, goal: 2500, weight: 165.4 },
    { day: 'Fri', calories: 1998, goal: 2500, weight: 165.2 },
    { day: 'Sat', calories: 2445, goal: 2500, weight: 165.2 },
    { day: 'Sun', calories: 1847, goal: 2500, weight: 165.2 }
  ]);

  const [recentMeals] = useState([
    {
      id: 1,
      name: 'Grilled Chicken Salad',
      time: '12:30 PM',
      calories: 420,
      image: '/api/placeholder/meal1',
      macros: { protein: 35, carbs: 12, fat: 8 }
    },
    {
      id: 2,
      name: 'Greek Yogurt with Berries',
      time: '9:15 AM',
      calories: 180,
      image: '/api/placeholder/meal2',
      macros: { protein: 15, carbs: 20, fat: 5 }
    },
    {
      id: 3,
      name: 'Oatmeal with Banana',
      time: '7:45 AM',
      calories: 320,
      image: '/api/placeholder/meal3',
      macros: { protein: 8, carbs: 58, fat: 6 }
    }
  ]);

  const macroData = [
    { name: 'Protein', value: userStats.protein, goal: userStats.proteinGoal, color: '#10b981' },
    { name: 'Carbs', value: userStats.carbs, goal: userStats.carbsGoal, color: '#f59e0b' },
    { name: 'Fat', value: userStats.fat, goal: userStats.fatGoal, color: '#ef4444' }
  ];

  const calorieProgress = (userStats.dailyCalories / userStats.dailyGoal) * 100;
  const proteinProgress = (userStats.protein / userStats.proteinGoal) * 100;
  const carbsProgress = (userStats.carbs / userStats.carbsGoal) * 100;
  const fatProgress = (userStats.fat / userStats.fatGoal) * 100;

  const StatsCard = ({ title, value, unit, goal, progress, icon: Icon, color = "primary" }) => (
    <motion.div
      className="card"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${color === 'primary' ? 'bg-primary-500/10' : 'bg-green-500/10'}`}>
            <Icon className={`w-5 h-5 ${color === 'primary' ? 'text-primary-400' : 'text-green-400'}`} />
          </div>
          <div>
            <p className="text-text-muted text-sm">{title}</p>
            <div className="flex items-baseline space-x-1">
              <span className="text-2xl font-bold text-text-primary">{value}</span>
              <span className="text-sm text-text-muted">/{goal} {unit}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-text-muted">Progress</span>
          <span className={`font-medium ${color === 'primary' ? 'text-primary-400' : 'text-green-400'}`}>
            {progress.toFixed(1)}%
          </span>
        </div>
        <div className="progress-bar h-2">
          <motion.div
            className={`h-2 rounded-full ${color === 'primary' ? 'bg-gradient-to-r from-primary-500 to-primary-400' : 'bg-gradient-to-r from-green-500 to-green-400'}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, progress)}%` }}
            transition={{ duration: 1, delay: 0.2 }}
          />
        </div>
      </div>
    </motion.div>
  );

  const MealCard = ({ meal }) => (
    <motion.div
      className="flex items-center space-x-4 p-4 rounded-xl bg-background-medium/30 hover:bg-background-medium/50 transition-all cursor-pointer"
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
        <BeakerIcon className="w-6 h-6 text-white" />
      </div>
      <div className="flex-1">
        <h4 className="font-medium text-text-primary">{meal.name}</h4>
        <p className="text-sm text-text-muted">{meal.time} • {meal.calories} cal</p>
      </div>
      <div className="text-right">
        <div className="text-xs text-text-muted space-y-1">
          <div>P: {meal.macros.protein}g</div>
          <div>C: {meal.macros.carbs}g</div>
          <div>F: {meal.macros.fat}g</div>
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
      {/* Header with greeting */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h1 className="text-4xl font-bold text-text-primary mb-2">Good morning, Alex! 👋</h1>
          <p className="text-text-muted">Let's make today a healthy one. You're doing great!</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-center">
            <div className="flex items-center space-x-1">
              <TrophyIcon className="w-5 h-5 text-yellow-400" />
              <span className="text-xl font-bold text-text-primary">{userStats.streak}</span>
            </div>
            <p className="text-xs text-text-muted">Day Streak</p>
          </div>
          <motion.button
            className="button-primary"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <CameraIcon className="w-5 h-5 mr-2" />
            Scan Food
          </motion.button>
        </div>
      </motion.div>

      {/* Main Stats Cards */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <StatsCard
          title="Calories"
          value={userStats.dailyCalories}
          unit="cal"
          goal={userStats.dailyGoal}
          progress={calorieProgress}
          icon={FireIcon}
        />
        <StatsCard
          title="Protein"
          value={userStats.protein}
          unit="g"
          goal={userStats.proteinGoal}
          progress={proteinProgress}
          icon={ChartBarIcon}
          color="success"
        />
        <StatsCard
          title="Weight"
          value={userStats.weight}
          unit="lbs"
          goal={userStats.weightGoal}
          progress={((userStats.weightGoal/userStats.weight) * 100)}
          icon={HeartIcon}
          color="success"
        />
        <StatsCard
          title="Water"
          value={userStats.water}
          unit="cups"
          goal={userStats.waterGoal}
          progress={(userStats.water / userStats.waterGoal) * 100}
          icon={BeakerIcon}
        />
      </motion.div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Calories Chart */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Weekly Calories</h3>
            <CalendarDaysIcon className="w-5 h-5 text-primary-400" />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="day" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f3f4f6'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="calories" 
                stroke="#f37316" 
                strokeWidth={3}
                name="Calories"
                dot={{ fill: '#f37316', strokeWidth: 2, r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="goal" 
                stroke="#64748b" 
                strokeWidth={1}
                strokeDasharray="5 5"
                name="Goal"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Macro Distribution */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-lg font-semibold text-text-primary mb-4">Today's Macros</h3>
          <div className="space-y-4">
            {macroData.map((macro, index) => (
              <div key={macro.name} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-text-primary font-medium">{macro.name}</span>
                  <span className="text-sm text-text-muted">
                    {macro.value}g / {macro.goal}g
                  </span>
                </div>
                <div className="progress-bar h-2">
                  <motion.div
                    className="h-2 rounded-full"
                    style={{ backgroundColor: macro.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, (macro.value / macro.goal) * 100)}%` }}
                    transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                  />
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold" style={{ color: '#10b981' }}>
                {proteinProgress.toFixed(0)}%
              </p>
              <p className="text-xs text-text-muted">Protein</p>
            </div>
            <div>
              <p className="text-2xl font-bold" style={{ color: '#f59e0b' }}>
                {carbsProgress.toFixed(0)}%
              </p>
              <p className="text-xs text-text-muted">Carbs</p>
            </div>
            <div>
              <p className="text-2xl font-bold" style={{ color: '#ef4444' }}>
                {fatProgress.toFixed(0)}%
              </p>
              <p className="text-xs text-text-muted">Fat</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Meals & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Meals */}
        <motion.div
          className="lg:col-span-2 card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-text-primary">Recent Meals</h3>
            <button className="text-primary-400 text-sm hover:text-primary-300 transition-colors">
              View All
            </button>
          </div>
          <div className="space-y-3">
            {recentMeals.map((meal) => (
              <MealCard key={meal.id} meal={meal} />
            ))}
          </div>
        </motion.div>

        {/* Quick Stats & Actions */}
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          {/* Achievement */}
          <div className="card text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <TrophyIcon className="w-8 h-8 text-white" />
            </div>
            <h4 className="font-semibold text-text-primary mb-2">Great Progress!</h4>
            <p className="text-sm text-text-muted mb-4">
              You've logged meals for 12 days straight. Keep it up!
            </p>
            <div className="text-xs text-primary-400">+50 XP earned</div>
          </div>

          {/* Today's Goal */}
          <div className="card">
            <h4 className="font-semibold text-text-primary mb-4">Today's Focus</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">Drink more water</span>
                <span className="text-xs bg-primary-500/20 text-primary-400 px-2 py-1 rounded-full">
                  2 more cups
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">Hit protein goal</span>
                <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                  25g left
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">10k steps</span>
                <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full">
                  3,200 left
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Dashboard;
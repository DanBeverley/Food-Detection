import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  TrophyIcon,
  FireIcon,
  HeartIcon,
  ChartBarIcon,
  CalendarDaysIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ScaleIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';

const Progress = () => {
  const [timeRange, setTimeRange] = useState('7d'); // 7d, 30d, 90d, 1y
  
  const [weightData] = useState([
    { date: '2025-06-24', weight: 168.2, goal: 160 },
    { date: '2025-06-25', weight: 167.8, goal: 160 },
    { date: '2025-06-26', weight: 167.5, goal: 160 },
    { date: '2025-06-27', weight: 167.1, goal: 160 },
    { date: '2025-06-28', weight: 166.8, goal: 160 },
    { date: '2025-06-29', weight: 166.4, goal: 160 },
    { date: '2025-06-30', weight: 165.2, goal: 160 }
  ]);

  const [calorieData] = useState([
    { date: '06/24', calories: 2234, goal: 2500, burned: 342 },
    { date: '06/25', calories: 2156, goal: 2500, burned: 398 },
    { date: '06/26', calories: 2089, goal: 2500, burned: 425 },
    { date: '06/27', calories: 2312, goal: 2500, burned: 380 },
    { date: '06/28', calories: 1998, goal: 2500, burned: 455 },
    { date: '06/29', calories: 2445, goal: 2500, burned: 320 },
    { date: '06/30', calories: 1847, goal: 2500, burned: 390 }
  ]);

  const [macroTrends] = useState([
    { date: '06/24', protein: 125, carbs: 280, fat: 78 },
    { date: '06/25', protein: 142, carbs: 245, fat: 72 },
    { date: '06/26', protein: 138, carbs: 220, fat: 85 },
    { date: '06/27', protein: 155, carbs: 290, fat: 88 },
    { date: '06/28', protein: 148, carbs: 195, fat: 65 },
    { date: '06/29', protein: 162, carbs: 310, fat: 92 },
    { date: '06/30', protein: 125, carbs: 230, fat: 68 }
  ]);

  const [achievements] = useState([
    {
      id: 1,
      title: '7-Day Streak',
      description: 'Logged meals for 7 consecutive days',
      icon: TrophyIcon,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10',
      earned: true,
      date: '2025-06-30'
    },
    {
      id: 2,
      title: 'Protein Goal Master',
      description: 'Hit protein goal 5 days in a row',
      icon: ChartBarIcon,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10',
      earned: true,
      date: '2025-06-29'
    },
    {
      id: 3,
      title: 'Weight Loss Warrior',
      description: 'Lost 3 pounds this month',
      icon: ScaleIcon,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10',
      earned: true,
      date: '2025-06-28'
    },
    {
      id: 4,
      title: 'Calorie Deficit Champion',
      description: 'Maintained calorie deficit for 14 days',
      icon: FireIcon,
      color: 'text-orange-400',
      bgColor: 'bg-orange-400/10',
      earned: false,
      progress: 78
    }
  ]);

  const currentWeight = weightData[weightData.length - 1].weight;
  const startWeight = weightData[0].weight;
  const weightLoss = startWeight - currentWeight;
  const weightGoal = 160;
  const weightProgress = ((startWeight - currentWeight) / (startWeight - weightGoal)) * 100;

  const avgCalories = calorieData.reduce((sum, day) => sum + day.calories, 0) / calorieData.length;
  const avgDeficit = calorieData.reduce((sum, day) => sum + (day.goal - day.calories), 0) / calorieData.length;

  const StatCard = ({ title, value, unit, subtitle, trend, icon: Icon, color = "primary" }) => (
    <motion.div
      className="card"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl ${color === 'primary' ? 'bg-primary-500/10' : color === 'green' ? 'bg-green-500/10' : color === 'blue' ? 'bg-blue-500/10' : 'bg-red-500/10'}`}>
          <Icon className={`w-6 h-6 ${color === 'primary' ? 'text-primary-400' : color === 'green' ? 'text-green-400' : color === 'blue' ? 'text-blue-400' : 'text-red-400'}`} />
        </div>
        {trend && (
          <div className="flex items-center space-x-1">
            {trend > 0 ? (
              <ArrowTrendingUpIcon className="w-4 h-4 text-green-400" />
            ) : (
              <ArrowTrendingDownIcon className="w-4 h-4 text-red-400" />
            )}
            <span className={`text-sm font-medium ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {Math.abs(trend).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
      
      <div>
        <p className="text-text-muted text-sm mb-1">{title}</p>
        <div className="flex items-baseline space-x-2">
          <span className="text-3xl font-bold text-text-primary">{value}</span>
          {unit && <span className="text-text-muted">{unit}</span>}
        </div>
        {subtitle && <p className="text-text-muted text-sm mt-1">{subtitle}</p>}
      </div>
    </motion.div>
  );

  const AchievementCard = ({ achievement }) => (
    <motion.div
      className={`card ${achievement.earned ? 'border-2 border-green-500/30' : ''}`}
      whileHover={{ scale: 1.02 }}
      layout
    >
      <div className="flex items-start space-x-4">
        <div className={`p-3 rounded-xl ${achievement.bgColor}`}>
          <achievement.icon className={`w-6 h-6 ${achievement.color}`} />
        </div>
        
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-text-primary">{achievement.title}</h4>
            {achievement.earned && (
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                Earned
              </span>
            )}
          </div>
          
          <p className="text-sm text-text-muted mb-3">{achievement.description}</p>
          
          {achievement.earned ? (
            <p className="text-xs text-text-muted">Earned on {achievement.date}</p>
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Progress</span>
                <span className="text-primary-400 font-medium">{achievement.progress}%</span>
              </div>
              <div className="progress-bar h-2">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${achievement.progress}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </div>
            </div>
          )}
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-text-primary mb-2">Your Progress 📈</h1>
          <p className="text-text-muted">Track your fitness journey and celebrate achievements</p>
        </div>
        
        <div className="flex items-center space-x-2">
          {['7d', '30d', '90d', '1y'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeRange === range
                  ? 'bg-primary-500 text-white'
                  : 'bg-secondary-800 text-text-muted hover:bg-secondary-700'
              }`}
            >
              {range === '7d' ? '7 Days' : range === '30d' ? '30 Days' : range === '90d' ? '90 Days' : '1 Year'}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <StatCard
          title="Weight Loss"
          value={weightLoss.toFixed(1)}
          unit="lbs"
          subtitle={`${currentWeight} lbs current`}
          trend={-2.3}
          icon={ScaleIcon}
          color="blue"
        />
        <StatCard
          title="Avg Daily Calories"
          value={Math.round(avgCalories)}
          unit="cal"
          subtitle={`${Math.round(avgDeficit)} cal deficit`}
          trend={-5.2}
          icon={FireIcon}
          color="primary"
        />
        <StatCard
          title="Weight Goal Progress"
          value={Math.round(weightProgress)}
          unit="%"
          subtitle={`${(weightGoal - currentWeight).toFixed(1)} lbs to go`}
          trend={12.4}
          icon={TrophyIcon}
          color="green"
        />
        <StatCard
          title="Logging Streak"
          value={12}
          unit="days"
          subtitle="Personal best!"
          trend={8.7}
          icon={CalendarDaysIcon}
          color="green"
        />
      </motion.div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weight Progress */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Weight Progress</h3>
            <ScaleIcon className="w-5 h-5 text-blue-400" />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weightData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} domain={['dataMin - 1', 'dataMax + 1']} />
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
                dataKey="weight" 
                stroke="#3b82f6" 
                strokeWidth={3}
                name="Weight"
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
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

        {/* Calorie Trends */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">Calorie Balance</h3>
            <FireIcon className="w-5 h-5 text-primary-400" />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={calorieData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f3f4f6'
                }}
              />
              <Bar dataKey="calories" fill="#f37316" name="Calories Consumed" />
              <Bar dataKey="burned" fill="#10b981" name="Calories Burned" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Macro Trends */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">Macro Nutrition Trends</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={macroTrends}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
            <YAxis stroke="#9CA3AF" fontSize={12} />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6'
              }}
            />
            <Area 
              type="monotone" 
              dataKey="protein" 
              stackId="1"
              stroke="#10b981" 
              fill="#10b981"
              fillOpacity={0.6}
              name="Protein (g)"
            />
            <Area 
              type="monotone" 
              dataKey="carbs" 
              stackId="1"
              stroke="#f59e0b" 
              fill="#f59e0b"
              fillOpacity={0.6}
              name="Carbs (g)"
            />
            <Area 
              type="monotone" 
              dataKey="fat" 
              stackId="1"
              stroke="#ef4444" 
              fill="#ef4444"
              fillOpacity={0.6}
              name="Fat (g)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Achievements */}
      <motion.div
        className="space-y-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-text-primary">Achievements</h3>
          <span className="text-sm text-text-muted">
            {achievements.filter(a => a.earned).length} of {achievements.length} earned
          </span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {achievements.map((achievement) => (
            <AchievementCard key={achievement.id} achievement={achievement} />
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Progress;
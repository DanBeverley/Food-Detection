import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  HomeIcon,
  CameraIcon,
  CpuChipIcon,
  CubeIcon,
  Cog6ToothIcon,
  ChartBarIcon,
  PlayIcon,
  StopIcon
} from '@heroicons/react/24/outline';

const Sidebar = () => {
  const location = useLocation();
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, completed
  
  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: HomeIcon,
      description: 'Your daily overview'
    },
    {
      name: 'Scan Food',
      href: '/scan',
      icon: CameraIcon,
      description: 'AI-powered food analysis'
    },
    {
      name: 'Food Log',
      href: '/food-log',
      icon: ChartBarIcon,
      description: 'Track your meals'
    },
    {
      name: 'Progress',
      href: '/progress',
      icon: ChartBarIcon,
      description: 'Your fitness journey'
    },
    {
      name: 'Goals',
      href: '/goals',
      icon: CubeIcon,
      description: 'Set and track goals'
    },
    {
      name: 'Profile',
      href: '/profile',
      icon: Cog6ToothIcon,
      description: 'Your account settings'
    }
  ];

  // Simulate training status updates
  useEffect(() => {
    const interval = setInterval(() => {
      // This would be replaced with real WebSocket connection
      setTrainingStatus(prev => {
        const statuses = ['idle', 'training', 'completed'];
        const currentIndex = statuses.indexOf(prev);
        return statuses[(currentIndex + 1) % statuses.length];
      });
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    switch (trainingStatus) {
      case 'training':
        return <PlayIcon className="w-4 h-4 text-primary-500 animate-pulse" />;
      case 'completed':
        return <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse" />;
      default:
        return <StopIcon className="w-4 h-4 text-secondary-500" />;
    }
  };

  const getStatusText = () => {
    switch (trainingStatus) {
      case 'training':
        return 'Training in progress...';
      case 'completed':
        return 'Training completed';
      default:
        return 'Ready';
    }
  };

  return (
    <motion.aside
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="fixed left-0 top-0 h-screen w-64 glass-effect border-r border-secondary-800/30 z-50"
    >
      <div className="flex flex-col h-full p-6">
        {/* Logo and Brand */}
        <motion.div 
          className="flex items-center space-x-3 mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center shadow-glow">
            <CameraIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">NutriScan</h1>
            <p className="text-xs text-text-muted">Smart Nutrition Tracking</p>
          </div>
        </motion.div>

        {/* Getting Started Status */}
        <motion.div
          className="mb-6 p-4 rounded-xl bg-background-medium/50 border border-secondary-800/50"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="text-center">
            <p className="text-2xl font-bold text-primary-400">0</p>
            <p className="text-xs text-text-muted">Meals Logged</p>
            <div className="mt-2 w-full bg-secondary-800 rounded-full h-2">
              <div className="bg-gradient-to-r from-primary-500 to-primary-400 h-2 rounded-full" style={{width: '0%'}}></div>
            </div>
            <p className="text-xs text-text-muted mt-1">Start your journey</p>
          </div>
        </motion.div>

        {/* Navigation Items */}
        <nav className="flex-1 space-y-2">
          {navigationItems.map((item, index) => {
            const isActive = location.pathname === item.href || 
                           (location.pathname === '/' && item.href === '/dashboard');
            
            return (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * index + 0.4 }}
              >
                <Link
                  to={item.href}
                  className={`nav-item group ${isActive ? 'active' : ''}`}
                >
                  <div className="relative">
                    <item.icon className="w-5 h-5 transition-all duration-300 group-hover:scale-110" />
                    {isActive && (
                      <motion.div
                        className="absolute -inset-1 bg-primary-500/20 rounded-lg -z-10"
                        layoutId="activeNavItem"
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                      />
                    )}
                  </div>
                  <div className="flex-1">
                    <span className="font-medium">{item.name}</span>
                    <p className="text-xs text-text-muted opacity-0 group-hover:opacity-100 transition-opacity">
                      {item.description}
                    </p>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </nav>

        {/* Quick Actions */}
        <motion.div
          className="mt-6 space-y-3"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <button className="w-full button-primary text-sm py-2">
            <CameraIcon className="w-4 h-4 mr-2" />
            Quick Scan
          </button>
          <button className="w-full button-secondary text-sm py-2">
            Add Manual Entry
          </button>
        </motion.div>

        {/* Version Info */}
        <motion.div
          className="mt-6 pt-6 border-t border-secondary-800/30"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <p className="text-xs text-text-muted text-center">
            Version 2.0.0 • Production
          </p>
        </motion.div>
      </div>
    </motion.aside>
  );
};

export default Sidebar;
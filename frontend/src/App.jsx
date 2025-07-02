import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// Import components
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import FoodScan from './pages/FoodScan';
import FoodLog from './pages/FoodLog';
import Progress from './pages/Progress';
import Goals from './pages/Goals';
import Profile from './pages/Profile';

// Optional advanced features (can be enabled in settings)
import Training from './pages/Training';
import ModelManager from './pages/ModelManager';

// Import styles
import './styles/globals.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background-dark flex">
        {/* Sidebar Navigation */}
        <Sidebar />
        
        {/* Main Content Area */}
        <main className="flex-1 ml-64 p-8">
          <AnimatePresence mode="wait">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/scan" element={<FoodScan />} />
                <Route path="/food-log" element={<FoodLog />} />
                <Route path="/progress" element={<Progress />} />
                <Route path="/goals" element={<Goals />} />
                <Route path="/profile" element={<Profile />} />
                
                {/* Advanced features - can be hidden/optional */}
                <Route path="/training" element={<Training />} />
                <Route path="/models" element={<ModelManager />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </main>
        
        {/* Background Elements */}
        <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
          {/* Animated Background Gradient */}
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-primary-600/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
          
          {/* Grid Pattern */}
          <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        </div>
      </div>
    </Router>
  );
}

export default App;
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CubeIcon,
  CloudArrowDownIcon,
  TrashIcon,
  EyeIcon,
  Cog6ToothIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const ModelManager = () => {
  const [models] = useState([
    {
      id: 1,
      name: 'MobileNetV3Small Classification',
      type: 'classification',
      version: 'v2.0.0',
      status: 'active',
      accuracy: 92.1,
      size: '12.3 MB',
      lastTrained: '2025-06-30',
      deploymentStatus: 'deployed',
      metrics: {
        precision: 0.923,
        recall: 0.918,
        f1Score: 0.920
      }
    },
    {
      id: 2,
      name: 'U-Net EfficientNetB0 Segmentation',
      type: 'segmentation',
      version: 'v2.0.0',
      status: 'active',
      accuracy: 89.5,
      size: '28.7 MB',
      lastTrained: '2025-06-30',
      deploymentStatus: 'deployed',
      metrics: {
        iou: 0.895,
        diceCoeff: 0.942,
        pixelAcc: 0.967
      }
    },
    {
      id: 3,
      name: 'MobileNetV3Small Classification',
      type: 'classification',
      version: 'v1.8.2',
      status: 'archived',
      accuracy: 90.8,
      size: '11.9 MB',
      lastTrained: '2025-06-25',
      deploymentStatus: 'inactive',
      metrics: {
        precision: 0.908,
        recall: 0.903,
        f1Score: 0.905
      }
    }
  ]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <CheckCircleIcon className="w-5 h-5 text-green-400" />;
      case 'training':
        return <ArrowPathIcon className="w-5 h-5 text-primary-400 animate-spin" />;
      case 'archived':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />;
      default:
        return <Cog6ToothIcon className="w-5 h-5 text-secondary-400" />;
    }
  };

  const getTypeColor = (type) => {
    return type === 'classification' ? 'bg-blue-500/10 text-blue-400' : 'bg-purple-500/10 text-purple-400';
  };

  const ModelCard = ({ model }) => (
    <motion.div
      className="card group"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      layout
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h3 className="text-lg font-semibold text-text-primary group-hover:text-primary-400 transition-colors">
              {model.name}
            </h3>
            <span className={`px-2 py-1 rounded-lg text-xs font-medium ${getTypeColor(model.type)}`}>
              {model.type}
            </span>
          </div>
          <p className="text-text-muted text-sm">Version {model.version}</p>
        </div>
        <div className="flex items-center space-x-2">
          {getStatusIcon(model.status)}
          <span className="text-sm text-text-muted capitalize">{model.status}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-background-medium/30 rounded-lg">
          <p className="text-text-muted text-xs mb-1">Accuracy</p>
          <p className="text-lg font-bold text-primary-400">{model.accuracy}%</p>
        </div>
        <div className="p-3 bg-background-medium/30 rounded-lg">
          <p className="text-text-muted text-xs mb-1">Model Size</p>
          <p className="text-lg font-bold text-text-primary">{model.size}</p>
        </div>
      </div>

      <div className="space-y-2 mb-4">
        <h4 className="text-sm font-medium text-text-primary">Performance Metrics</h4>
        {Object.entries(model.metrics).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between text-sm">
            <span className="text-text-muted capitalize">
              {key.replace(/([A-Z])/g, ' $1').trim()}
            </span>
            <span className="text-text-primary font-medium">
              {(value * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between pt-4 border-t border-secondary-800/30">
        <div>
          <p className="text-xs text-text-muted">Last trained</p>
          <p className="text-sm font-medium text-text-primary">{model.lastTrained}</p>
        </div>
        <div className="flex items-center space-x-2">
          <motion.button
            className="p-2 hover:bg-secondary-800 rounded-lg transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <EyeIcon className="w-4 h-4 text-text-primary" />
          </motion.button>
          <motion.button
            className="p-2 hover:bg-secondary-800 rounded-lg transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <CloudArrowDownIcon className="w-4 h-4 text-text-primary" />
          </motion.button>
          <motion.button
            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <TrashIcon className="w-4 h-4 text-red-400" />
          </motion.button>
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
          <h1 className="text-4xl font-bold text-text-primary mb-2">Model Manager</h1>
          <p className="text-text-muted">Manage your trained models and their deployments</p>
        </div>
        <button className="button-primary">
          <CloudArrowDownIcon className="w-5 h-5 mr-2" />
          Import Model
        </button>
      </div>

      {/* Stats Overview */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="card text-center">
          <CubeIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-text-primary">{models.length}</p>
          <p className="text-text-muted text-sm">Total Models</p>
        </div>
        <div className="card text-center">
          <CheckCircleIcon className="w-8 h-8 text-green-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-text-primary">
            {models.filter(m => m.status === 'active').length}
          </p>
          <p className="text-text-muted text-sm">Active Models</p>
        </div>
        <div className="card text-center">
          <CloudArrowDownIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-text-primary">
            {models.filter(m => m.deploymentStatus === 'deployed').length}
          </p>
          <p className="text-text-muted text-sm">Deployed</p>
        </div>
        <div className="card text-center">
          <Cog6ToothIcon className="w-8 h-8 text-primary-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-text-primary">
            {Math.round(models.reduce((acc, m) => acc + parseFloat(m.size), 0))}
          </p>
          <p className="text-text-muted text-sm">Total Size (MB)</p>
        </div>
      </motion.div>

      {/* Filter Tabs */}
      <motion.div
        className="flex items-center space-x-4"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
      >
        <button className="nav-item active">
          All Models
        </button>
        <button className="nav-item">
          Classification
        </button>
        <button className="nav-item">
          Segmentation
        </button>
        <button className="nav-item">
          Active Only
        </button>
      </motion.div>

      {/* Models Grid */}
      <motion.div
        className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, staggerChildren: 0.1 }}
      >
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
      </motion.div>

      {/* Model Comparison */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">Model Performance Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-secondary-800/30">
                <th className="text-left text-text-muted text-sm font-medium pb-3">Model</th>
                <th className="text-left text-text-muted text-sm font-medium pb-3">Type</th>
                <th className="text-left text-text-muted text-sm font-medium pb-3">Accuracy</th>
                <th className="text-left text-text-muted text-sm font-medium pb-3">Size</th>
                <th className="text-left text-text-muted text-sm font-medium pb-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <motion.tr
                  key={model.id}
                  className="border-b border-secondary-800/10 hover:bg-background-medium/20"
                  whileHover={{ backgroundColor: "rgba(31, 41, 55, 0.3)" }}
                >
                  <td className="py-4">
                    <div>
                      <p className="text-text-primary font-medium">{model.name}</p>
                      <p className="text-text-muted text-sm">v{model.version}</p>
                    </div>
                  </td>
                  <td className="py-4">
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${getTypeColor(model.type)}`}>
                      {model.type}
                    </span>
                  </td>
                  <td className="py-4">
                    <span className="text-primary-400 font-medium">{model.accuracy}%</span>
                  </td>
                  <td className="py-4">
                    <span className="text-text-primary">{model.size}</span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(model.status)}
                      <span className="text-text-muted text-sm capitalize">{model.status}</span>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ModelManager;
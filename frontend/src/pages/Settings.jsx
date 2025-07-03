import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cog6ToothIcon,
  ComputerDesktopIcon,
  CloudIcon,
  ShieldCheckIcon,
  BellIcon,
  ChartBarIcon,
  DocumentTextIcon,
  UserIcon
} from '@heroicons/react/24/outline';

const Settings = () => {
  const [settings, setSettings] = useState({
    general: {
      theme: 'dark',
      language: 'en',
      autoSave: true,
      notifications: true
    },
    training: {
      autoBatchSize: true,
      mixedPrecision: true,
      earlyStoppingPatience: 8,
      maxEpochs: 50
    },
    inference: {
      batchProcessing: true,
      confidenceThreshold: 0.7,
      maxFileSize: 10,
      outputFormat: 'json'
    },
    system: {
      enableLogging: true,
      logLevel: 'info',
      cacheSize: 1024,
      autoCleanup: true
    }
  });

  const updateSetting = (category, key, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }));
  };

  const SettingSection = ({ title, icon: Icon, children }) => (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2 bg-primary-500/10 rounded-lg">
          <Icon className="w-5 h-5 text-primary-400" />
        </div>
        <h3 className="text-lg font-semibold text-text-primary">{title}</h3>
      </div>
      <div className="space-y-6">
        {children}
      </div>
    </motion.div>
  );

  const ToggleSetting = ({ label, description, value, onChange }) => (
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <p className="text-text-primary font-medium">{label}</p>
        {description && <p className="text-text-muted text-sm mt-1">{description}</p>}
      </div>
      <motion.button
        className={`relative w-12 h-6 rounded-full transition-colors ${
          value ? 'bg-primary-500' : 'bg-secondary-700'
        }`}
        onClick={() => onChange(!value)}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          className="absolute top-1 w-4 h-4 bg-white rounded-full shadow-md"
          animate={{ x: value ? 24 : 4 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      </motion.button>
    </div>
  );

  const SelectSetting = ({ label, description, value, options, onChange }) => (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex-1">
          <p className="text-text-primary font-medium">{label}</p>
          {description && <p className="text-text-muted text-sm mt-1">{description}</p>}
        </div>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input-field w-full"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );

  const NumberSetting = ({ label, description, value, min, max, onChange }) => (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex-1">
          <p className="text-text-primary font-medium">{label}</p>
          {description && <p className="text-text-muted text-sm mt-1">{description}</p>}
        </div>
        <span className="text-primary-400 font-medium">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-secondary-700 rounded-lg appearance-none cursor-pointer slider"
      />
    </div>
  );

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-text-primary mb-2">Settings</h1>
        <p className="text-text-muted">Configure your food detection system preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* General Settings */}
        <SettingSection title="General" icon={Cog6ToothIcon}>
          <SelectSetting
            label="Theme"
            description="Choose your preferred interface theme"
            value={settings.general.theme}
            options={[
              { value: 'dark', label: 'Dark Theme' },
              { value: 'light', label: 'Light Theme' },
              { value: 'auto', label: 'Auto (System)' }
            ]}
            onChange={(value) => updateSetting('general', 'theme', value)}
          />

          <SelectSetting
            label="Language"
            description="Select your preferred language"
            value={settings.general.language}
            options={[
              { value: 'en', label: 'English' },
              { value: 'es', label: 'Spanish' },
              { value: 'fr', label: 'French' },
              { value: 'de', label: 'German' }
            ]}
            onChange={(value) => updateSetting('general', 'language', value)}
          />

          <ToggleSetting
            label="Auto Save"
            description="Automatically save your work and settings"
            value={settings.general.autoSave}
            onChange={(value) => updateSetting('general', 'autoSave', value)}
          />

          <ToggleSetting
            label="Notifications"
            description="Receive notifications about training progress and system events"
            value={settings.general.notifications}
            onChange={(value) => updateSetting('general', 'notifications', value)}
          />
        </SettingSection>

        {/* Training Settings */}
        <SettingSection title="Training" icon={ComputerDesktopIcon}>
          <ToggleSetting
            label="Auto Batch Size"
            description="Automatically determine optimal batch size based on available memory"
            value={settings.training.autoBatchSize}
            onChange={(value) => updateSetting('training', 'autoBatchSize', value)}
          />

          <ToggleSetting
            label="Mixed Precision"
            description="Use mixed precision training for faster performance"
            value={settings.training.mixedPrecision}
            onChange={(value) => updateSetting('training', 'mixedPrecision', value)}
          />

          <NumberSetting
            label="Early Stopping Patience"
            description="Number of epochs to wait before early stopping"
            value={settings.training.earlyStoppingPatience}
            min={3}
            max={20}
            onChange={(value) => updateSetting('training', 'earlyStoppingPatience', value)}
          />

          <NumberSetting
            label="Max Epochs"
            description="Maximum number of training epochs"
            value={settings.training.maxEpochs}
            min={10}
            max={200}
            onChange={(value) => updateSetting('training', 'maxEpochs', value)}
          />
        </SettingSection>

        {/* Inference Settings */}
        <SettingSection title="Inference" icon={ChartBarIcon}>
          <ToggleSetting
            label="Batch Processing"
            description="Process multiple images simultaneously for better performance"
            value={settings.inference.batchProcessing}
            onChange={(value) => updateSetting('inference', 'batchProcessing', value)}
          />

          <NumberSetting
            label="Confidence Threshold"
            description="Minimum confidence level for predictions"
            value={settings.inference.confidenceThreshold * 100}
            min={50}
            max={95}
            onChange={(value) => updateSetting('inference', 'confidenceThreshold', value / 100)}
          />

          <NumberSetting
            label="Max File Size (MB)"
            description="Maximum allowed file size for image uploads"
            value={settings.inference.maxFileSize}
            min={1}
            max={50}
            onChange={(value) => updateSetting('inference', 'maxFileSize', value)}
          />

          <SelectSetting
            label="Output Format"
            description="Default format for inference results"
            value={settings.inference.outputFormat}
            options={[
              { value: 'json', label: 'JSON' },
              { value: 'csv', label: 'CSV' },
              { value: 'xml', label: 'XML' }
            ]}
            onChange={(value) => updateSetting('inference', 'outputFormat', value)}
          />
        </SettingSection>

        {/* System Settings */}
        <SettingSection title="System" icon={ComputerDesktopIcon}>
          <ToggleSetting
            label="Enable Logging"
            description="Log system events and errors for debugging"
            value={settings.system.enableLogging}
            onChange={(value) => updateSetting('system', 'enableLogging', value)}
          />

          <SelectSetting
            label="Log Level"
            description="Verbosity level for system logs"
            value={settings.system.logLevel}
            options={[
              { value: 'error', label: 'Error Only' },
              { value: 'warning', label: 'Warning' },
              { value: 'info', label: 'Info' },
              { value: 'debug', label: 'Debug' }
            ]}
            onChange={(value) => updateSetting('system', 'logLevel', value)}
          />

          <NumberSetting
            label="Cache Size (MB)"
            description="Amount of memory to use for caching"
            value={settings.system.cacheSize}
            min={256}
            max={4096}
            onChange={(value) => updateSetting('system', 'cacheSize', value)}
          />

          <ToggleSetting
            label="Auto Cleanup"
            description="Automatically clean up temporary files and logs"
            value={settings.system.autoCleanup}
            onChange={(value) => updateSetting('system', 'autoCleanup', value)}
          />
        </SettingSection>
      </div>

      {/* Action Buttons */}
      <motion.div
        className="flex items-center justify-end space-x-4 pt-6 border-t border-secondary-800/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <button className="button-secondary">
          Reset to Defaults
        </button>
        <button className="button-secondary">
          Export Settings
        </button>
        <button className="button-primary">
          Save Changes
        </button>
      </motion.div>

      {/* System Information */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="p-3 bg-background-medium/30 rounded-lg">
            <p className="text-text-muted mb-1">Version</p>
            <p className="text-text-primary font-medium">v2.0.0</p>
          </div>
          <div className="p-3 bg-background-medium/30 rounded-lg">
            <p className="text-text-muted mb-1">Last Updated</p>
            <p className="text-text-primary font-medium">June 30, 2025</p>
          </div>
          <div className="p-3 bg-background-medium/30 rounded-lg">
            <p className="text-text-muted mb-1">Environment</p>
            <p className="text-text-primary font-medium">Production</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Settings;
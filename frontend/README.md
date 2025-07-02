# FoodDetect Frontend

A modern, interactive frontend for the Food Detection AI system built with React, Tailwind CSS, and Framer Motion.

## Features

- **Modern UI Design**: Clean, minimal interface with smooth animations and transitions
- **Real-time Dashboard**: Live training progress monitoring with interactive charts
- **Food Inference**: Drag-and-drop image upload with instant AI analysis
- **Model Management**: Comprehensive model versioning and deployment control
- **Training Interface**: Visual training progress with real-time metrics
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices

## Color Palette

- **Primary**: Orange-ish yellow (less saturated) - `#f37316`
- **Background**: Deep black gradients - `#0a0a0a`, `#1a1a1a`, `#2a2a2a`
- **Text**: White and grey variations - `#ffffff`, `#e2e8f0`, `#94a3b8`
- **Accents**: Secondary greys - `#64748b`, `#475569`, `#334155`

## Getting Started

### Prerequisites

- Node.js 16.0 or higher
- npm or yarn package manager

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
src/
├── components/          # Reusable UI components
│   └── Sidebar.jsx     # Navigation sidebar
├── pages/              # Main application pages
│   ├── Dashboard.jsx   # Training overview and metrics
│   ├── Inference.jsx   # Image upload and analysis
│   ├── Training.jsx    # Training progress monitoring
│   ├── ModelManager.jsx # Model management interface
│   └── Settings.jsx    # System configuration
├── styles/             # CSS and styling
│   └── globals.css     # Global styles and Tailwind config
├── hooks/              # Custom React hooks
├── utils/              # Utility functions
└── assets/             # Static assets

```

## Key Technologies

- **React 18** - Modern React with hooks and concurrent features
- **Tailwind CSS** - Utility-first CSS framework with custom design system
- **Framer Motion** - Smooth animations and page transitions
- **Recharts** - Interactive charts for training metrics
- **React Router** - Client-side routing
- **React Dropzone** - File upload with drag-and-drop
- **Heroicons** - Beautiful SVG icons

## Features Overview

### Dashboard
- Real-time training metrics visualization
- Live loss and accuracy charts
- Model status indicators
- Recent activity feed

### Inference
- Drag-and-drop image upload
- Real-time food detection and classification
- Nutritional information display
- Segmentation mask visualization
- Batch processing support

### Training
- Interactive training control (start/pause/stop)
- Live progress monitoring
- Configuration management
- Training logs display

### Model Manager
- Model version control
- Performance comparison
- Deployment status tracking
- Model import/export

### Settings
- System configuration
- Training parameters
- Inference settings
- Theme and language preferences

## Customization

### Colors
Modify the color palette in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    500: '#f37316', // Main orange-yellow
    // ... other shades
  },
  // ... other color definitions
}
```

### Animations
Custom animations are defined in `globals.css` and can be extended:

```css
@keyframes customAnimation {
  /* Animation keyframes */
}

.animate-custom {
  animation: customAnimation 1s ease-in-out;
}
```

## Integration with Backend

The frontend is designed to integrate with the Python backend through:

- **REST APIs** for model management and configuration
- **WebSocket connections** for real-time training updates
- **File upload endpoints** for image inference
- **SSE (Server-Sent Events)** for live progress monitoring

## Performance Optimizations

- **Code splitting** with React.lazy()
- **Image optimization** with proper loading states
- **Virtualized lists** for large datasets
- **Memoized components** to prevent unnecessary re-renders
- **Efficient state management** with React hooks

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the existing code style and structure
2. Use semantic commit messages
3. Add proper TypeScript types for new components
4. Include unit tests for new functionality
5. Update documentation for new features

## License

This project is part of the Food Detection AI system and follows the same licensing terms.
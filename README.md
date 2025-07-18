# Food Detection, Volume, and Calorie Estimation System

A multimodal deep learning pipeline for food analysis using RGB-D cameras. Combines computer vision and nutritional analysis for volume estimation and calorie calculation.

## Current Status: Development Version

**Latest Update (July 2025)**: Multimodal segmentation model with extensive debugging and optimization work completed.

### Implemented Features
- **Food Segmentation**: U-Net architecture with multimodal inputs (RGB + depth + point cloud)
- **Food Classification**: MobileNetV3Small for 108 food classes
- **Volume Estimation**: Point cloud-based volume calculation
- **Nutritional Analysis**: USDA API integration
- **Training Infrastructure**: TPU/GPU support with distribution strategies

### Known Working Components
- Data pipeline with TFRecord format
- Multimodal model architecture
- TPU/GPU training compatibility
- Point cloud processing and sanitization

## Core Components

### Deep Learning Pipeline
- **Segmentation Model**: U-Net with parallel backbones (EfficientNetB0, MobileNetV3Small, PointNet-style)
- **Classification Model**: MobileNetV3Small with standard augmentation
- **Training Features**: TPU/GPU support, mixed precision, staged training
- **Data Pipeline**: TFRecord format with NaN/inf sanitization

### Volume Estimation System
- **Point Cloud Processing**: RGB-D to 3D point cloud conversion
- **Volume Calculation**: Point cloud-based volume estimation
- **Camera Calibration**: Configurable intrinsics for different devices

### Nutritional Analysis
- **USDA API Integration**: FoodData Central API support
- **Density Database**: Food-specific density values for mass calculation
- **Pipeline**: Volume → Density → Mass → Calories

## Dataset Integration

**MetaFood3D Support**:
- 108 food classes with multimodal data
- RGB images + depth maps + 3D point clouds
- TFRecord preprocessing for efficient training

## Project Structure

```
Food-Detection/
├── main.py                    # Central pipeline orchestrator
├── food_analyzer.py           # Core analysis engine
├── config_pipeline.yaml       # Production configuration
├── requirements.txt           # Dependencies
│
├── models/                    # Deep Learning Models
│   ├── classification/        # Food classification (MobileNetV3)
│   └── segmentation/          # Food segmentation (U-Net)
│
├── volume_helpers/            # Volume Estimation
│   ├── volume_estimator.py    # Point cloud processing
│   ├── volume_helpers.py      # Mesh processing
│   └── density_lookup.py      # Nutritional database
│
├── trained_models/            # Exported Models
│   ├── classification/exported/  # TFLite classification model
│   └── segmentation/exported/    # TFLite segmentation model
│
├── data/                      # Datasets and Metadata
├── scripts/                   # Utilities and preprocessing
└── tests/                     # Test suite
```

## Quick Start

### Setup Environment
```bash
git clone <repository-url>
cd Food-Detection
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Training

#### Segmentation Model
```bash
python models/segmentation/train.py
```

#### Classification Model
```bash
python models/classification/train.py
```

#### Data Preparation
```bash
python scripts/prepare_segmentation_metadata.py
```

### Food Analysis
```bash
python main.py --run-inference \
  --image_path "path/to/food_image.jpg" \
  --depth_map_path "path/to/depth_map.jpg" \
  --volume_estimation_method depth \
  --usda_api_key "your_api_key"
```

## Configuration

The system uses YAML configuration files:

- **Pipeline Config**: `config_pipeline.yaml` - Camera intrinsics, paths, volume parameters
- **Classification Config**: `models/classification/config.yaml` - Training parameters
- **Segmentation Config**: `models/segmentation/config.yaml` - Model architecture and training

## Technical Implementation

### Training Optimizations
- **Mixed Precision**: Enabled for performance on compatible hardware
- **Distribution Strategy**: TPU/GPU support with automatic hardware detection
- **Data Pipeline**: TFRecord format with optimized loading
- **Numerical Stability**: NaN/inf sanitization and gradient clipping

### Model Architecture
- **Segmentation**: U-Net with skip connections and multimodal fusion
- **Classification**: MobileNetV3Small for efficiency
- **Point Cloud Processing**: PointNet-style encoder for 3D data

## API Reference

### Core Analysis Function
```python
from food_analyzer import analyze_food_item

results = analyze_food_item(
    image_path="food.jpg",
    depth_map_path="depth.jpg",
    config=pipeline_config,
    volume_estimation_method="depth",
    usda_api_key="your_key"
)

# Results include:
# - food_label, confidence
# - volume_cm3, estimated_mass_g
# - estimated_total_calories
# - timing metrics
```

## Recent Work (July 2025)

### Multimodal Segmentation Implementation
- **U-Net Architecture**: Parallel backbones for RGB, depth, and point cloud inputs
- **TPU/GPU Training**: Optimized data pipeline with automatic hardware detection
- **Staged Training**: Two-phase training strategy (freeze backbones, then fine-tune)
- **Numerical Stability**: Extensive debugging and optimization for gradient stability

### Training Infrastructure
- **TFRecord Pipeline**: Efficient data loading for distributed training
- **Mixed Precision**: bfloat16 support for TPU and GPU acceleration
- **Distribution Strategy**: OneDeviceStrategy to avoid gradient aggregation issues
- **Data Sanitization**: NaN/inf handling for point cloud data

### Known Issues
- **Training Stability**: Ongoing work on gradient overflow in multimodal fusion
- **Hardware Compatibility**: MirroredStrategy conflicts with mixed precision resolved
- **XLA Compilation**: Debug tools incompatibility with XLA compiler addressed

## Development Status

This is an active development project with working components and ongoing optimization efforts. The codebase demonstrates multimodal deep learning techniques and distributed training infrastructure.

---

**Status**: Development version with functional components and active debugging/optimization work.

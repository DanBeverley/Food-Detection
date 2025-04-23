# Food Detection & Calorie Estimator

A cross-platform mobile app (iOS & Android) that performs food segmentation, classification, depth-based volume estimation, and calorie calculation. MVP focuses on common foods, using device depth sensors (LiDAR/ToF) or manual volume entry fallback. Models are trained in TensorFlow/PyTorch and exported to TensorFlow Lite & Core ML.

## Folder Structure

```text
.
├── .gitignore
├── README.md
├── data/
├── notebooks/
├── models/
├── src/
├── scripts/
└── tests/
```

## Getting Started

1. Clone the repo

```bash
git clone <repo-url>
cd Food-Detection
```

2. Create Python env and install dependencies

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

3. Populate data folders as described in docs.

## Next Steps

- Load and preprocess data (scripts)
- Train and export models (models/*)
- Integrate with mobile apps (src/ios, src/android)
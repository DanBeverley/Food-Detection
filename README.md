# Food Detection, Volume, and Calorie Estimation Project

This project provides a comprehensive Python-based pipeline for detecting food items in images, estimating their volume and calories. It leverages deep learning models for segmentation and classification, and supports various methods for volume estimation. The ultimate goal is to create a production-ready system deployable on iOS devices, utilizing datasets like MetaFood3D for robust real-world performance.

## Key Features

*   **End-to-End Pipeline Orchestration**: `main.py` serves as the central script to run various stages including data preparation, model training, TFLite model exportation, and inference for both classification and segmentation models.
*   **Food Segmentation**: Identifies and masks food items using a U-Net-like model (architecture configurable, e.g., with an EfficientNet backbone). TFLite format supported for deployment.
*   **Food Classification**: Classifies segmented food items (architecture configurable, e.g., MobileNetV3Small, EfficientNetV2). TFLite format supported for deployment.
*   **Configurable Training Pipelines**: Includes scripts and YAML configurations for training both segmentation and classification models, supporting features like:
    *   Custom datasets (e.g., prepared from MetaFood3D).
    *   Data augmentation.
    *   Learning rate schedules (e.g., cosine decay).
    *   L2 regularization.
    *   Early stopping.
    *   TensorBoard logging.
*   **TFLite Export**: Scripts to convert trained Keras models to TensorFlow Lite format, with options for quantization (currently defaults to none).
*   **Google Colab Integration**: A unified script (`train_all_colab.py`) to facilitate training both models sequentially on Google Colab.
*   **Volume Estimation**: Calculates 3D volume using:
    *   Depth maps (from RGB-D sensors) and camera intrinsics.
    *   Direct 3D mesh files (e.g., `.obj`).
    *   Fallback to a dummy depth map if real data is unavailable (for pipeline flow, not accuracy).
*   **Nutritional Information Lookup**: Retrieves food density (g/cm³) and calories (kcal/100g) from:
    *   A local custom JSON database (`data/databases/custom_density_db.json`).
    *   USDA FoodData Central API (requires API key, with caching).
*   **Mass & Calorie Calculation**: Estimates mass and total calories based on volume and nutritional data.
*   **Modular and Configurable**: Key parameters, model paths, and settings are managed through YAML configuration files. Script execution from `main.py` is robust, ensuring correct working directories for sub-scripts.
*   **Automated Testing**: Basic end-to-end pipeline tests using `unittest`.

## Current Project Status (As of May 2025)

*   The `main.py` script has been enhanced to orchestrate the full pre-inference pipeline.
*   Individual data preparation scripts (`prepare_classification_dataset.py`, `prepare_segmentation_metadata.py`) are functional.
*   Individual training scripts (`models/classification/train.py`, `models/segmentation/train.py`) have been successfully debugged for issues related to logging and execution when called from `main.py`.
*   Segmentation model training shows logs and completes successfully when initiated via `main.py`.
*   The next immediate step is to test the combined pipeline for: Classification Data Prep -> Classification Training -> Segmentation Data Prep -> Segmentation Training -> TFLite Export for both models, all orchestrated through `main.py`.

## Project Structure

```
Food-Detection/
├── .git/                     # Git version control
├── .gitignore
├── README.md                 # This file
├── config_pipeline.yaml      # Configuration for the end-to-end inference pipeline
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point for running the inference pipeline and other stages (data prep, training, export)
├── food_analyzer.py          # Core logic for the inference pipeline
├── train_all_colab.py        # Unified training script for Google Colab
|
├── data/
│   ├── classification/       # Processed data for classification model
│   │   ├── metadata.json     # Image paths and labels
│   │   └── label_map.json    # Maps class names to integer labels
│   ├── segmentation/         # Processed data for segmentation model
│   │   └── metadata.json     # Image and mask paths
│   ├── databases/
│   │   └── custom_density_db.json # Local nutritional database
│   └── cache/                # Cached API responses (if implemented)
│   └── sample_data/          # Sample images, depth maps for testing
|
├── models/                   # Model-specific scripts, configs, and modules
│   ├── __init__.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── config.yaml       # Training & export config for classification
│   │   ├── data.py           # Data loading and preprocessing
│   │   ├── train.py          # Training script
│   │   ├── export_tflite.py  # TFLite export script
│   │   ├── evaluate.py       # Evaluation script for Keras model
│   │   ├── evaluate_tflite.py # Evaluation script for TFLite model
│   │   └── predict_classification.py # Prediction script
│   └── segmentation/
│       ├── __init__.py
│       ├── config.yaml       # Training & export config for segmentation
│       ├── data.py           # Data loading and preprocessing
│       ├── train.py          # Training script
│       ├── export_tflite.py  # TFLite export script
│       ├── evaluate_segmentation.py # Evaluation script
│       └── predict_segmentation.py # Prediction script
|
├── trained_models/           # Where trained models (.h5, .tflite) are saved by default
│   ├── classification/
│   │   ├── exported/         # TFLite models and labels
│   │   │   └── classification_model_full_default.tflite
│   │   │   └── label_map.json (copied here by export script)
│   │   └── checkpoints/      # Keras model checkpoints
│   └── segmentation/
│       ├── exported/
│       │   └── segmentation_model_default.tflite
│       └── checkpoints/
|
├── volume_helpers/
│   ├── __init__.py
│   ├── volume_helpers.py
│   └── density_lookup.py
|
├── scripts/                  # Utility scripts for data preparation, etc.
│   └── prepare_classification_dataset.py # Example dataset preparation script
|
├── logs/                     # Default directory for TensorBoard logs and other logs
│   ├── classification/
│   └── segmentation/
|
└── tests/
    └── test_pipeline.py      # Automated tests for the inference pipeline
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Food-Detection
    ```

2.  **Python Environment:**
    It's highly recommended to use a Python virtual environment. Python 3.9+ is recommended.
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install TensorFlow, OpenCV, PyYAML, and other necessary packages.

## Data Preparation

Raw datasets like MetaFood3D need to be processed into a format suitable for the training scripts. This can be done using standalone scripts or orchestrated via `main.py`.

*   **Using `main.py`:**
    ```bash
    # Prepare classification data
    python main.py --prepare-classification-data --classification_input_dir path/to/classification/images --classification_output_meta_dir data/classification

    # Prepare segmentation data
    python main.py --prepare-segmentation-data --segmentation_rgbd_input_dir path/to/segmentation/rgbd --segmentation_pointcloud_input_dir path/to/segmentation/pc --segmentation_output_meta_dir data/segmentation
    ```

*   **Classification Data (`data/classification/`):**
    *   Images should be organized into subdirectories by class, or listed in `metadata.json` with their corresponding labels.
    *   `metadata.json`: A JSON file, typically a list of dictionaries, where each dictionary contains at least `image_path` (relative to a base directory defined in `models/classification/config.yaml`) and `label` (class name).
    *   `label_map.json`: Maps string class labels to integer indices. This can be generated by `scripts/prepare_classification_dataset.py` or manually.
    *   The `scripts/prepare_classification_dataset.py` script provides an example of how to generate `metadata.json` and `label_map.json` from a directory of images.

*   **Segmentation Data (`data/segmentation/`):**
    *   `metadata.json`: A JSON file, typically a list of dictionaries, where each dictionary contains `image_path` and `mask_path` (relative to base directories defined in `models/segmentation/config.yaml`).
    *   Masks should be binary images where food pixels are distinct from background pixels.

## Configuration Files

Configuration is managed through YAML files:

1.  **`models/classification/config.yaml`**: For classification model training and export.
    *   `paths`: Directories for data, model saving, logs.
    *   `data`: Image size, batch size, class mode, validation split, label map filename.
    *   `model`: Architecture (e.g., `EfficientNetV2B0`), number of classes, input shape, if to use pretrained weights.
    *   `training`: Epochs, optimizer settings (name, learning rate), loss function, metrics, regularization (L2), learning rate schedule (e.g., `cosine_decay`), early stopping parameters.
    *   `augmentation`: Parameters for data augmentation (rotation, zoom, shift, shear, flip, brightness).
    *   `export`: TFLite filename, quantization settings.

2.  **`models/segmentation/config.yaml`**: For segmentation model training and export.
    *   Similar structure to classification config, but tailored for segmentation (e.g., U-Net architecture, specific backbone like `EfficientNetB0`, binary crossentropy or dice loss).

3.  **`config_pipeline.yaml`**: For the end-to-end inference pipeline (`main.py`).
    *   `models`: Paths to the `.tflite` segmentation and classification models, and classification labels file.
    *   `model_params`: Input sizes for models, number of segmentation classes.
    *   `camera_intrinsics`: `fx`, `fy`, `cx`, `cy` for depth-to-point-cloud conversion.
    *   `volume_params`: Parameters for volume calculation (e.g., depth cutoffs).
    *   `classification_confidence_threshold`: Minimum confidence for a classification to be considered valid.

## Training Models

Ensure data is prepared and configuration files (`models/*/config.yaml`) are set up correctly. Training can be initiated directly via model-specific scripts or through `main.py`.

*   **Using `main.py` (Recommended for pipeline consistency):**
    ```bash
    # Train classification model
    python main.py --train-classification

    # Train segmentation model
    python main.py --train-segmentation

    # Train both
    python main.py --train-classification --train-segmentation
    ```

*   **Locally (Direct Scripts):**
    1.  **Train Classification Model:**
    ```bash
    python models/classification/train.py --config models/classification/config.yaml
    ```
    Logs and checkpoints will be saved to directories specified in the config (e.g., `logs/classification/` and `trained_models/classification/checkpoints/`). Monitor training with TensorBoard:
    ```bash
    tensorboard --logdir logs/classification
    ```

2.  **Train Segmentation Model:**
    ```bash
    python models/segmentation/train.py --config models/segmentation/config.yaml
    ```
    Monitor training with TensorBoard:
    ```bash
    tensorboard --logdir logs/segmentation
    ```

## Exporting Models to TFLite

After training, export the Keras models to TensorFlow Lite format. This can be done using standalone scripts or orchestrated via `main.py`.

*   **Using `main.py` (Recommended for pipeline consistency):**
    ```bash
    # Export classification model to TFLite
    python main.py --export-classification-tflite

    # Export segmentation model to TFLite
    python main.py --export-segmentation-tflite

    # Export both
    python main.py --export-classification-tflite --export-segmentation-tflite
    ```

*   **Locally (Direct Scripts):**
    1.  **Export Classification Model:**
    ```bash
    python models/classification/export_tflite.py --config models/classification/config.yaml
    ```
    The `.tflite` model will be saved (e.g., to `trained_models/classification/exported/classification_model_full_default.tflite`).

2.  **Export Segmentation Model:**
    ```bash
    python models/segmentation/export_tflite.py --config models/segmentation/config.yaml
    ```
    The `.tflite` model will be saved (e.g., to `trained_models/segmentation/exported/segmentation_model_default.tflite`).

## Training on Google Colab

Use the `train_all_colab.py` script for a streamlined training experience on Colab.

1.  **Prepare Project for Upload:**
    *   Ensure your local `Food-Detection` project directory contains all necessary code, data preparation scripts, and updated configuration files.
    *   Zip your `Food-Detection` project directory.
    *   If your processed datasets (`data/classification/`, `data/segmentation/`) are very large, consider zipping them separately.

2.  **Upload to Google Drive:**
    *   Upload the project zip file (e.g., `Food-Detection.zip`) to your Google Drive.
    *   Upload dataset zips if separate.

3.  **Colab Notebook Setup:**
    Open a new Colab notebook and run the following cells:

    ```python
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Unzip your project (adjust path to your zip file)
    !unzip "/content/drive/My Drive/Food-Detection.zip" -d "/content/"

    # Navigate to the project directory
    import os
    os.chdir('/content/Food-Detection')
    !ls # Verify contents (should show train_all_colab.py, models/, data/, etc.)

    # If datasets were zipped separately, unzip them into the correct locations:
    # !unzip "/content/drive/My Drive/my_classification_data.zip" -d "/content/Food-Detection/data/"
    # !unzip "/content/drive/My Drive/my_segmentation_data.zip" -d "/content/Food-Detection/data/"
    # Verify data structure, e.g., /content/Food-Detection/data/classification/metadata.json should exist.

    # Install dependencies (Colab often has TensorFlow pre-installed, adjust if needed)
    !pip install -r requirements.txt
    ```
    *Ensure your Colab runtime has a GPU enabled (Runtime -> Change runtime type).* 

4.  **Run Unified Training Script:**
    ```python
    !python train_all_colab.py
    ```
    This script will train and export both classification and segmentation models sequentially. Logs will be printed in the notebook output.

5.  **Save Trained Models:**
    After training, the `.h5` checkpoints and `.tflite` models will be in `/content/Food-Detection/trained_models/`. Copy them back to your Google Drive for persistence:
    ```python
    # Example: Create a directory in Drive and copy models
    !mkdir -p "/content/drive/My Drive/Food-Detection-Output/trained_models"
    !cp -r /content/Food-Detection/trained_models/* "/content/drive/My Drive/Food-Detection-Output/trained_models/"
    !mkdir -p "/content/drive/My Drive/Food-Detection-Output/logs"
    !cp -r /content/Food-Detection/logs/* "/content/drive/My Drive/Food-Detection-Output/logs/"
    ```

## Running the Inference Pipeline

Once models are trained and exported (and paths in `config_pipeline.yaml` are correct), run the end-to-end analysis:

1.  **Set USDA API Key (Optional):**
    If you want to use the USDA API for nutritional lookups for items not in `custom_density_db.json`:
    *   Windows (PowerShell): `$env:USDA_API_KEY="YOUR_KEY_HERE"`
    *   macOS/Linux (Bash): `export USDA_API_KEY="YOUR_KEY_HERE"`

2.  **Execute `main.py`:**
    ```bash
    python main.py --image "path/to/image.jpg" \
                   --depth "path/to/depth_map.npy_or_png" \
                   --mesh_file_path "path/to/mesh.obj" \
                   --config "config_pipeline.yaml"
    ```
    *   `--depth`: Optional path to a depth map.
    *   `--mesh_file_path`: Optional path to a 3D mesh file. Takes precedence over depth for volume estimation.
    *   If neither depth nor mesh is provided, volume/mass/calorie estimates will be unreliable.

## Testing

The project includes an initial automated testing suite for the inference pipeline:
```bash
python -m unittest tests.test_pipeline
```
Refer to `tests/test_pipeline.py` for example test cases.

## Upcoming Goals / Production Plan

*   **iOS Deployment**: Package the inference pipeline and TFLite models into an iOS application for on-device food analysis.
*   **Model Accuracy & Robustness**: Continuously improve model performance on diverse, real-world food items using datasets like MetaFood3D. This includes fine-tuning, exploring different architectures/quantization, and enhancing data augmentation.
*   **User Testing**: Conduct user testing on the iOS application to gather feedback and identify areas for improvement in usability and accuracy.
*   **Optimize for Mobile**: Ensure efficient performance (latency, memory usage) of models on mobile devices.
*   **Expand Nutritional Database**: Enhance the local `custom_density_db.json` and improve USDA API integration.

## Key Dependencies

*   `tensorflow` (>=2.10 recommended)
*   `opencv-python-headless`
*   `PyYAML`
*   `scipy`
*   `numpy`
*   `requests`
*   `trimesh`
*   `scikit-learn`
*   `tensorflow-addons`
*   (Full list in `requirements.txt`)
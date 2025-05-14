# Food Detection & Calorie Estimator

This project provides a Python-based pipeline for analyzing food images. It performs food segmentation, classification, volume estimation (using a depth map, a 3D mesh file, or a fallback dummy depth map), density lookup, calorie estimation, and mass calculation. The system is designed to be configurable and extensible.

## Key Features of the Python Pipeline

*   **Food Segmentation**: Identifies and masks the food item in an image using a TFLite segmentation model.
*   **Food Classification**: Classifies the segmented food item using a TFLite classification model and a label map.
*   **Volume Estimation**:
    *   Calculates the 3D volume of the food item using either a depth map (and camera intrinsics) or a direct 3D mesh file (e.g., `.obj`).
    *   Supports fallback to a dummy depth map if a real one or a mesh file is not provided.
    *   Utilizes `scipy.spatial.ConvexHull` (for depth maps) or `trimesh` (for mesh files).
*   **Nutritional Information Lookup (Density & Calories)**:
    *   Retrieves food density (g/cm³) and calories (kcal/100g) from a local custom JSON database (`data/databases/custom_density_db.json`). The database supports entries with both density and calories, or density alone.
    *   Optionally queries the USDA FoodData Central API for this information if not found locally (requires an API key).
    *   Caches API results to minimize redundant lookups.
*   **Mass & Calorie Calculation**:
    *   Estimates the mass of the food item based on its calculated volume and looked-up density.
    *   Estimates the total calories of the food item based on its mass and calories per 100g.
*   **Configurable**: Pipeline parameters (model paths, camera intrinsics, API keys) are managed through a YAML configuration file (`config_pipeline.yaml`).

## Recent Pipeline Enhancements

*   **Masked Classification Input**: To potentially improve classification accuracy, the image provided to the classification model is now masked using the output from the segmentation stage. This ensures the classifier focuses primarily on the food item, minimizing background interference.
*   **Flexible Segmentation Mask Handling**: The pipeline now supports the use of pre-computed segmentation masks. Users can provide a path to an existing mask image via the `--mask_path` command-line argument. If a valid pre-computed mask is supplied, it will be used; otherwise, the system defaults to generating a mask using the integrated segmentation model. The source of the segmentation mask (e.g., "precomputed_mask_file: <filename>" or "model_generated") is included in the analysis results.
*   **Automated Testing Suite**: An initial automated testing suite has been introduced in the `tests/` directory (`tests/test_pipeline.py`). This suite, built using Python's `unittest` framework, facilitates end-to-end testing of the pipeline. It allows for defining specific test cases with known inputs and expected outputs to verify pipeline integrity and track the impact of future modifications.

## How the Python Pipeline Works

The analysis is orchestrated by `main.py` and `food_analyzer.py`:

1.  **Initialization (`main.py`)**:
    *   Parses command-line arguments: paths to the input image, depth map (optional), mesh file path (optional), configuration file, and USDA API key (optional).
    *   Calls the main analysis function in `food_analyzer.py`.

2.  **Core Analysis (`food_analyzer.py` - `analyze_food_item` function)**:
    *   **Load Configuration**: Reads `config_pipeline.yaml`.
    *   **Load Inputs**: Loads the RGB image. Attempts to load the provided depth map or mesh file; if unavailable or invalid, creates a dummy depth map based on image dimensions.
    *   **Load Models**: Loads TFLite segmentation and classification models, along with the classification `label_map.json`.
    *   **Segmentation**: Runs inference with the segmentation model to produce a food mask.
    *   **Classification**: Runs inference with the classification model on the (potentially masked) image to get a food label and confidence.
    *   **Volume Estimation** (orchestrated by `food_analyzer.py`, implemented in `volume_helpers.py`):
        *   If a mesh file path is provided, its volume is calculated using `trimesh` and used directly.
        *   Otherwise, if a depth map is provided, it's converted into a 3D point cloud using camera intrinsics. The volume of this point cloud is calculated using `ConvexHull`.
        *   If neither is available, a dummy volume might be assumed or an error/warning logged.
    *   **Nutritional Lookup (`density_lookup.py`)**: Queries local `custom_density_db.json` and/or USDA API for density (g/cm³) and calories (kcal/100g) of the classified food label.
    *   **Mass & Calorie Calculation**: Computes mass (volume × density) and then total estimated calories (mass × calories_per_100g / 100).
    *   **Output**: Returns a dictionary containing all results (mask shape, food label, confidence, volume, density, mass, calories_kcal_per_100g, estimated_total_calories).

## Project Structure

The repository is organized as follows:

```text
.
├── .git/                   # Git version control files
├── .gitignore             
├── README.md               # This file
├── main.py                 # Main script to run the food analysis pipeline
├── food_analyzer.py        # Core logic for the food analysis pipeline
├── config_pipeline.yaml    # Configuration file for the pipeline
├── requirements.txt        # Python package dependencies
├── create_dummy_classification_data.py # Script (enerating placeholder data/models)
├── data/                   # Data files used by the pipeline
│   ├── databases/          # Custom databases (e.g., for density and calories)
│   │   └── custom_density_db.json  # Stores food items with their density (g/cm³) and calories (kcal/100g)
│   ├── cache/              # Cached data (e.g., from API lookups) - (Verify if used and path)
│   └── sample_data/        # Sample images, depth maps for testing
├── models/                 # Python scripts for model definition, loading, and inference logic
│   ├── segmentation/
│   │   └── predict_segmentation.py
│   └── classification/
│       └── predict_classification.py
├── trained_models/         # Pre-trained model files (e.g., .tflite) and label maps
│   ├── segmentation/
│   │   └── segmentation_model.tflite
│   └── classification/
│       ├── classification_model.tflite
│       └── label_map.json
├── volume_helpers/         # Utility modules for volume estimation and density lookup
│   ├── volume_helpers.py
│   └── density_lookup.py
├── scripts/                # Additional helper scripts 
├── logs/                   # Log files generated by the application
├── checkpoints/            # Model training checkpoints
├── tests/                  # Automated test suite
│   └── test_pipeline.py    # End-to-end pipeline tests
└── __pycache__/            # Python bytecode cache
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Food-Detection
    ```

2.  **Create a Python environment and install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows
    venv\\Scripts\\activate
    # On macOS/Linux
    # source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Prepare Data and Models:**
    *   Ensure your TFLite models (`segmentation_model.tflite`, `classification_model.tflite`) and `label_map.json` are placed in the `trained_models/` subdirectories as specified in `config_pipeline.yaml`.
    *   Update `config_pipeline.yaml` with correct paths and camera intrinsic values if necessary.
    *   Populate `data/databases/custom_density_db.json` with any custom food densities and calories.

## Configuration

The main configuration for the pipeline is done via `config_pipeline.yaml`. This file includes:
*   Paths to segmentation and classification models.
*   Path to the classification label map.
*   Camera intrinsic parameters (`fx`, `fy`, `cx`, `cy`) for accurate 3D conversions.
*   Depth processing parameters (e.g., min/max depth cutoffs).

## How to Run

Execute the `main.py` script with the required arguments. Example:

```bash
python main.py --image "path/to/your/image.jpg" --depth "path/to/your/depth_map.npy_or_png" --mesh_file_path "path/to/your/mesh.obj" --config "config_pipeline.yaml" --api_key "YOUR_USDA_API_KEY_IF_NEEDED"
```
*   `--depth`: Optional. Path to the depth map. If a `--mesh_file_path` is also provided, the mesh file will take precedence for volume estimation.
*   `--mesh_file_path`: Optional. Path to a 3D mesh file (e.g., `.obj`). Takes precedence over `--depth` for volume estimation.
*   If neither `--depth` nor `--mesh_file_path` is valid, a dummy depth map might be used for basic pipeline flow, but volume/mass/calorie estimates will be unreliable.
*   `--api_key`: Optional. Needed if you want to use the USDA API for density and calorie lookups if the food item isn't in your custom DB.

## Future Enhancements (from original vision)

*   Train models on comprehensive datasets like FoodSeg-103 for real food recognition.
*   Develop more sophisticated dummy depth map generation for better volume estimation when real depth is unavailable.
*   Integrate the pipeline into a cross-platform mobile application (iOS & Android).
*   Explore using device-specific depth sensors (LiDAR/ToF on mobile devices).
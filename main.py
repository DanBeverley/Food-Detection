import argparse
import logging
import os
import subprocess
import yaml
import glob
from food_analyzer import analyze_food_item # Assuming analyze_food_item is the entry point for inference
from scripts.utils import _get_project_root # Assuming this utility exists

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Paths ---
# These might be better managed or passed as arguments if they vary often
CLASSIFICATION_CONFIG_PATH = "models/classification/config.yaml"
SEGMENTATION_CONFIG_PATH = "models/segmentation/config.yaml"
PIPELINE_CONFIG_PATH = "config_pipeline.yaml"

# --- Script Paths (relative to project root) ---
PREPARE_CLASSIFICATION_SCRIPT = "scripts/prepare_classification_dataset.py"
PREPARE_SEGMENTATION_SCRIPT = "scripts/prepare_segmentation_metadata.py"
TRAIN_CLASSIFICATION_SCRIPT = "models/classification/train.py"
TRAIN_SEGMENTATION_SCRIPT = "models/segmentation/train.py"
EXPORT_CLASSIFICATION_TFLITE_SCRIPT = "models/classification/export_tflite.py"
EXPORT_SEGMENTATION_TFLITE_SCRIPT = "models/segmentation/export_tflite.py"

# --- Default Output Dirs (relative to project root) ---
CLASSIFICATION_MODEL_DIR = "trained_models/classification"
SEGMENTATION_MODEL_DIR = "trained_models/segmentation"
CLASSIFICATION_TFLITE_EXPORT_DIR = os.path.join(CLASSIFICATION_MODEL_DIR, "exported") # Default, can be overridden by config
SEGMENTATION_TFLITE_EXPORT_DIR = os.path.join(SEGMENTATION_MODEL_DIR, "exported") # Default, can be overridden by config
DEFAULT_CLASSIFICATION_META_OUTPUT_DIR = "data/classification"
DEFAULT_SEGMENTATION_META_OUTPUT_DIR = "data/segmentation"

def run_script(script_path, config_path=None, project_root='.', extra_args=None):
    """Helper function to run a Python script using subprocess."""
    cmd = ['python', os.path.join(project_root, script_path)]
    if config_path:
        cmd.extend(['--config', os.path.join(project_root, config_path)])
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        logger.info(f"Script {script_path} output:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"Script {script_path} stderr:\n{process.stderr}")
        logger.info(f"Successfully executed: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}:")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"Error: Script not found at {os.path.join(project_root, script_path)}. Ensure paths are correct.")
        return False

def prepare_data(args, project_root):
    logger.info("Starting data preparation stage...")
    success = True
    if args.prepare_classification_data or args.run_all:
        logger.info("Preparing classification data...")
        if not args.classification_input_dir:
            logger.error("Error: --classification_input_dir is required for preparing classification data.")
            return False
        class_prep_args = [
            '--source_dir', args.classification_input_dir,
            '--output_metadata_dir', args.classification_output_meta_dir
        ]
        if not run_script(PREPARE_CLASSIFICATION_SCRIPT, project_root=project_root, extra_args=class_prep_args):
            success = False
            logger.error("Classification data preparation failed.")

    if args.prepare_segmentation_data or args.run_all:
        logger.info("Preparing segmentation data...")
        if not args.segmentation_rgbd_input_dir or not args.segmentation_pointcloud_input_dir:
            logger.error("Error: --segmentation_rgbd_input_dir and --segmentation_pointcloud_input_dir are required for preparing segmentation data.")
            return False
        seg_prep_args = [
            '--source_dir', args.segmentation_rgbd_input_dir,
            '--output_metadata_dir', args.segmentation_output_meta_dir,
            '--source_point_cloud_dir', args.segmentation_pointcloud_input_dir
        ]
        if not run_script(PREPARE_SEGMENTATION_SCRIPT, project_root=project_root, extra_args=seg_prep_args):
            success = False
            logger.error("Segmentation data preparation failed.")
    
    if success:
        logger.info("Data preparation stage completed.")
    else:
        logger.error("Data preparation stage encountered errors.")
    return success

def train_models(args, project_root):
    logger.info("Starting model training stage...")
    success = True
    if args.train_classification or args.run_all:
        logger.info("Training classification model...")
        if not run_script(TRAIN_CLASSIFICATION_SCRIPT, CLASSIFICATION_CONFIG_PATH, project_root):
            success = False
            logger.error("Classification model training failed.")
        else:
            logger.info("Classification model training finished.")

    if args.train_segmentation or args.run_all:
        logger.info("Training segmentation model...")
        if not run_script(TRAIN_SEGMENTATION_SCRIPT, SEGMENTATION_CONFIG_PATH, project_root):
            success = False
            logger.error("Segmentation model training failed.")
        else:
            logger.info("Segmentation model training finished.")

    if success:
        logger.info("Model training stage completed.")
    else:
        logger.error("Model training stage encountered errors.")
    return success

def export_tflite_models(args, project_root):
    logger.info("Starting TFLite model export stage...")
    success = True
    if args.export_classification_tflite or args.run_all:
        logger.info("Exporting classification model to TFLite...")
        if not run_script(EXPORT_CLASSIFICATION_TFLITE_SCRIPT, CLASSIFICATION_CONFIG_PATH, project_root):
            success = False
            logger.error("Classification model TFLite export failed.")
        else:
            logger.info("Classification model TFLite export finished.")

    if args.export_segmentation_tflite or args.run_all:
        logger.info("Exporting segmentation model to TFLite...")
        if not run_script(EXPORT_SEGMENTATION_TFLITE_SCRIPT, SEGMENTATION_CONFIG_PATH, project_root):
            success = False
            logger.error("Segmentation model TFLite export failed.")
        else:
            logger.info("Segmentation model TFLite export finished.")

    if success:
        logger.info("TFLite model export stage completed.")
    else:
        logger.error("TFLite model export stage encountered errors.")
    return success

def find_final_tflite_model(project_root, model_config_rel_path: str, model_type: str):
    """Finds the TFLite model file containing 'final' in its name, 
       reading the export directory from the model's specific config.yaml.
    Args:
        project_root (str): Absolute path to the project root.
        model_config_rel_path (str): Relative path from project_root to the model's config.yaml.
        model_type (str): 'classification' or 'segmentation', for logging.
    """
    model_config_abs_path = os.path.join(project_root, model_config_rel_path)
    search_dir = None

    if not os.path.exists(model_config_abs_path):
        logger.error(f"Model configuration file not found for {model_type}: {model_config_abs_path}")
        # Fallback to default constants if config is missing, though this is not ideal
        if model_type == "classification":
            search_dir = os.path.join(project_root, CLASSIFICATION_TFLITE_EXPORT_DIR)
            logger.warning(f"Falling back to default classification TFLite export dir: {search_dir}")
        elif model_type == "segmentation":
            search_dir = os.path.join(project_root, SEGMENTATION_TFLITE_EXPORT_DIR)
            logger.warning(f"Falling back to default segmentation TFLite export dir: {search_dir}")
        else:
            logger.error(f"Unknown model type '{model_type}' for fallback TFLite search.")
            return None
    else:
        try:
            with open(model_config_abs_path, 'r') as f_config:
                model_config = yaml.safe_load(f_config)
            
            # Assuming the export dir is 'paths.tflite_export_dir' and is relative to project_root
            tflite_export_dir_from_config = model_config.get('paths', {}).get('tflite_export_dir')
            
            if tflite_export_dir_from_config:
                search_dir = os.path.join(project_root, tflite_export_dir_from_config)
                logger.info(f"Using TFLite export directory from {model_config_rel_path}: {tflite_export_dir_from_config}")
            else:
                logger.warning(f"'paths.tflite_export_dir' not found in {model_config_rel_path}. Attempting fallback for {model_type}.")
                # Fallback logic if key is missing in config
                if model_type == "classification":
                    search_dir = os.path.join(project_root, CLASSIFICATION_TFLITE_EXPORT_DIR)
                elif model_type == "segmentation":
                    search_dir = os.path.join(project_root, SEGMENTATION_TFLITE_EXPORT_DIR)
                logger.warning(f"Falling back to default TFLite export dir for {model_type}: {search_dir}")

        except Exception as e:
            logger.error(f"Error reading or parsing model config {model_config_abs_path}: {e}. Attempting fallback for {model_type}.")
            if model_type == "classification":
                search_dir = os.path.join(project_root, CLASSIFICATION_TFLITE_EXPORT_DIR)
            elif model_type == "segmentation":
                search_dir = os.path.join(project_root, SEGMENTATION_TFLITE_EXPORT_DIR)
            logger.warning(f"Falling back to default TFLite export dir for {model_type}: {search_dir}")

    if not search_dir:
        logger.error(f"Could not determine TFLite search directory for {model_type}. Cannot find model.")
        return None

    pattern = "*final*.tflite"
    logger.info(f"Searching for {model_type} TFLite model in: {search_dir} with pattern: {pattern}")
    
    if not os.path.isdir(search_dir):
        logger.warning(f"TFLite export directory not found: {search_dir}")
        return None

    final_models = glob.glob(os.path.join(search_dir, pattern))
    if not final_models:
        logger.warning(f"No 'final' TFLite model found for {model_type} in {search_dir}.")
        return None
    
    # Prefer models with 'final' and sort by modification time (newest first) if multiple exist
    final_models.sort(key=os.path.getmtime, reverse=True)
    logger.info(f"Found final {model_type} TFLite model: {final_models[0]}")
    return final_models[0]

def run_inference(args, project_root):
    logger.info("Starting inference stage...")
    if not args.image_path:
        logger.error("Image path (--image_path) is required for inference.")
        return False

    # Load base pipeline config
    pipeline_config_full_path = os.path.join(project_root, PIPELINE_CONFIG_PATH)
    try:
        with open(pipeline_config_full_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded base pipeline config from {pipeline_config_full_path}")
    except FileNotFoundError:
        logger.error(f"Pipeline configuration file not found: {pipeline_config_full_path}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"Error parsing pipeline configuration file {pipeline_config_full_path}: {e}")
        return False

    # Dynamically find the latest 'final' TFLite models by reading their respective config files
    classification_model_path = find_final_tflite_model(project_root, CLASSIFICATION_CONFIG_PATH, "classification")
    segmentation_model_path = find_final_tflite_model(project_root, SEGMENTATION_CONFIG_PATH, "segmentation")

    if classification_model_path:
        logger.info(f"Using classification TFLite model: {classification_model_path}")
        # Store relative path in config
        config['models']['classification_tflite'] = os.path.relpath(classification_model_path, project_root)
        logger.info(f"Updated classification TFLite model path in config to: {config['models']['classification_tflite']}")
    else:
        logger.warning("Could not find a 'final' classification TFLite model. Using path from config or default.")
        if 'models' not in config or 'classification_tflite' not in config['models']:
            logger.error("Original classification TFLite path missing in config['models'] and no final model found to replace it.")
            # food_analyzer will likely fail if this path is also invalid/missing

    if segmentation_model_path:
        logger.info(f"Using segmentation TFLite model: {segmentation_model_path}")
        # Store relative path in config
        config['models']['segmentation_tflite'] = os.path.relpath(segmentation_model_path, project_root)
        logger.info(f"Updated segmentation TFLite model path in config to: {config['models']['segmentation_tflite']}")
    else:
        logger.warning("Could not find a 'final' segmentation TFLite model. Using path from config or default.")
        if 'models' not in config or 'segmentation_tflite' not in config['models']:
            logger.error("Original segmentation TFLite path missing in config['models'] and no final model found to replace it.")

    # Ensure all necessary paths in config are absolute for food_analyzer, or food_analyzer handles relative paths from project_root
    # For now, assume food_analyzer.py can handle paths relative to project_root if they are stored that way in config_pipeline.yaml
    # or that it resolves them correctly.

    # Call the original inference logic
    try:
        logger.info(f"Analyzing food item: {args.image_path}")
        analysis_results = analyze_food_item(
            image_path=args.image_path,
            depth_map_path=args.depth_map_path,
            point_cloud_path=args.point_cloud_path,
            mesh_file_path=args.mesh_file_path, # Added mesh file path
            config=config, # Pass the potentially modified config
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            display_results=not args.no_display
        )
        logger.info("Inference completed.")
        logger.info("Analysis Results:")
        for key, value in analysis_results.items():
            if isinstance(value, dict): # For timing dict
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
        return True
    except FileNotFoundError as e:
        logger.error(f"File not found during inference: {e}. Check input paths and paths in config_pipeline.yaml.")
        return False
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="End-to-end Food Detection Pipeline: Data Prep, Training, TFLite Export, Inference.")
    project_root = _get_project_root() # Get project root once

    # --- Stage Control Arguments ---
    parser.add_argument('--run-all', action='store_true', help="Run all stages: data prep, train all, export all, then inference if image_path provided.")
    parser.add_argument('--prepare-all-data', action='store_true', help="Run all data preparation steps.")
    parser.add_argument('--prepare-classification-data', action='store_true', help="Prepare classification dataset metadata.")
    parser.add_argument('--prepare-segmentation-data', action='store_true', help="Prepare segmentation dataset metadata.")
    
    parser.add_argument('--train-all', action='store_true', help="Train both classification and segmentation models.")
    parser.add_argument('--train-classification', action='store_true', help="Train the classification model.")
    parser.add_argument('--train-segmentation', action='store_true', help="Train the segmentation model.")

    parser.add_argument('--export-all-tflite', action='store_true', help="Export both trained models to TFLite.")
    parser.add_argument('--export-classification-tflite', action='store_true', help="Export classification model to TFLite.")
    parser.add_argument('--export-segmentation-tflite', action='store_true', help="Export segmentation model to TFLite.")

    parser.add_argument('--run-inference', action='store_true', help="Run the inference pipeline on a food item.")

    # --- Inference Specific Arguments (copied from original main.py, ensure they match food_analyzer call) ---
    parser.add_argument('--image_path', type=str, help="Path to the input RGB image file for inference.")
    parser.add_argument('--depth_map_path', type=str, default=None, help="Path to the input depth map file (optional).")
    parser.add_argument('--point_cloud_path', type=str, default=None, help="Path to the input point cloud file (optional).")
    parser.add_argument('--mesh_file_path', type=str, default=None, help="Path to the .obj mesh file for volume calculation (optional).")
    parser.add_argument('--output_dir', type=str, default='output_analysis', help="Directory to save analysis results and intermediate steps.")
    parser.add_argument('--save_steps', action='store_true', help="Save intermediate images/outputs from the analysis pipeline.")
    parser.add_argument('--no_display', action='store_true', help="Do not display results directly (e.g., images).")

    # --- Dataset Path Arguments (for data preparation scripts if they need overrides) ---
    parser.add_argument('--classification_input_dir', type=str, help="Root directory for raw classification images (required if preparing classification data).")
    parser.add_argument('--classification_output_meta_dir', type=str, default=DEFAULT_CLASSIFICATION_META_OUTPUT_DIR, help=f"Directory to save classification metadata.json and label_map.json (default: {DEFAULT_CLASSIFICATION_META_OUTPUT_DIR}).")
    parser.add_argument('--segmentation_rgbd_input_dir', type=str, help="Root directory for segmentation RGBD data (images, masks, depth maps) (required if preparing segmentation data).")
    parser.add_argument('--segmentation_pointcloud_input_dir', type=str, help="Root directory for segmentation point cloud data (required if preparing segmentation data).")
    parser.add_argument('--segmentation_output_meta_dir', type=str, default=DEFAULT_SEGMENTATION_META_OUTPUT_DIR, help=f"Directory to save segmentation metadata.json (default: {DEFAULT_SEGMENTATION_META_OUTPUT_DIR}).")

    args = parser.parse_args()

    # Determine which stages to run
    run_any_prep = args.prepare_all_data or args.prepare_classification_data or args.prepare_segmentation_data
    run_any_train = args.train_all or args.train_classification or args.train_segmentation
    run_any_export = args.export_all_tflite or args.export_classification_tflite or args.export_segmentation_tflite

    if args.run_all:
        run_any_prep = run_any_train = run_any_export = args.run_inference = True # Inference only if image_path given
        if not args.image_path:
            logger.info("Running all pre-inference stages. Inference will be skipped as --image_path is not provided.")
            args.run_inference = False # Explicitly turn off inference if no image for run_all

    # Link flags like --prepare-all-data to individual flags
    if args.prepare_all_data:
        args.prepare_classification_data = True
        args.prepare_segmentation_data = True
    if args.train_all:
        args.train_classification = True
        args.train_segmentation = True
    if args.export_all_tflite:
        args.export_classification_tflite = True
        args.export_segmentation_tflite = True

    # Execute stages
    if run_any_prep:
        if not prepare_data(args, project_root):
            logger.error("Data preparation failed. Halting pipeline.")
            return

    if run_any_train:
        if not train_models(args, project_root):
            logger.error("Model training failed. Halting pipeline.")
            return

    if run_any_export:
        if not export_tflite_models(args, project_root):
            logger.error("TFLite export failed. Halting pipeline.")
            return

    if args.run_inference:
        if not args.image_path:
            logger.error("Cannot run inference without --image_path.")
        elif not run_inference(args, project_root):
            logger.error("Inference pipeline failed.")
        else:
            logger.info("Inference pipeline completed successfully.")
    
    # If no specific stage is requested, print help (or a default action if desired)
    if not (run_any_prep or run_any_train or run_any_export or args.run_inference):
        logger.info("No pipeline stage specified. Use --help to see available options.")
        parser.print_help()

if __name__ == '__main__':
    main()
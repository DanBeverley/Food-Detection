import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
import json
import yaml
import numpy as np
import shutil 

from data import load_classification_data, _get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def representative_dataset_gen(dataset: tf.data.Dataset, num_samples: int):
    """Generator function for TFLite int8 quantization representative dataset."""
    count = 0
    # Take a limited number of samples from the dataset
    for image, _ in dataset.take(num_samples):
        if count >= num_samples:
             break
        # Get input signature from the model
        yield [tf.cast(image, tf.float32)]
        count += 1
    logger.info(f"Generated {count} samples for representative dataset.")

def export_model_to_tflite(config: dict):
    """
    Export a trained Keras model to TensorFlow Lite format based on config.

    Args:
        config: Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If model or config file is missing.
        ValueError: If configuration is invalid.
        RuntimeError: If export fails.
    """
    paths_config = config.get('paths', {})
    export_config = config.get('export', {})
    quant_config = export_config.get('quantization', {})

    project_root = _get_project_root()

    # Determine Input Model Path
    model_dir = os.path.join(project_root, paths_config.get('model_save_dir', 'trained_models/classification'))
    
    keras_model_path = None
    specified_model_filename = export_config.get('keras_model_filename')

    if specified_model_filename:
        potential_path = os.path.join(model_dir, specified_model_filename)
        if os.path.exists(potential_path) and os.path.isfile(potential_path):
            keras_model_path = potential_path
            logger.info(f"Using Keras model specified in config: {specified_model_filename}")
        else:
            logger.warning(
                f"Keras model specified in config ('{specified_model_filename}') not found at '{potential_path}'. \
                Attempting to find latest model in '{model_dir}'."
            )

    if not keras_model_path:  # If not specified or specified file not found
        if not os.path.isdir(model_dir):
            logger.error(f"Model save directory not found: {model_dir}")
            raise FileNotFoundError(f"Model save directory missing: {model_dir}")

        h5_files = [
            f for f in os.listdir(model_dir) 
            if f.endswith('.h5') and os.path.isfile(os.path.join(model_dir, f))
        ]

        if not h5_files:
            logger.error(f"No .h5 model files found in directory: {model_dir}")
            raise FileNotFoundError(f"No .h5 Keras model files found in {model_dir}")

        # Prefer files with "final" in their name (case-insensitive)
        final_files = [f for f in h5_files if "final" in f.lower()]
        
        selected_file_list = final_files if final_files else h5_files
        
        # Get the most recently modified file from the selected list
        try:
            latest_model_file = max(
                selected_file_list, 
                key=lambda f: os.path.getmtime(os.path.join(model_dir, f))
            )
            keras_model_path = os.path.join(model_dir, latest_model_file)
            if final_files and latest_model_file in final_files:
                logger.info(f"Found 'final' model. Using latest: {latest_model_file} from {model_dir}")
            elif h5_files: # Ensure we have some h5 files before logging this path
                logger.info(f"No 'final' model found or specified one missing. Using latest .h5 model: {latest_model_file} from {model_dir}")
        except ValueError: # Should not happen if h5_files is not empty
             logger.error(f"Could not determine latest model file in {model_dir} from list: {selected_file_list}")
             raise FileNotFoundError(f"Could not determine latest model file in {model_dir}")

    # Check if a model path was actually determined
    if not keras_model_path or not os.path.exists(keras_model_path):
        # This case should ideally be caught by earlier checks, but as a safeguard:
        error_msg = f"Keras model not found. Searched in '{model_dir}'" 
        if specified_model_filename:
            error_msg += f" (specified: '{specified_model_filename}')"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Loading Keras model from: {keras_model_path}")

    # Determine Output TFLite Path 
    tflite_dir = os.path.join(project_root, paths_config.get('tflite_export_dir', os.path.join(model_dir, 'exported')))
    tflite_filename = export_config.get('tflite_filename', 'model.tflite')
    output_tflite_path = os.path.join(tflite_dir, tflite_filename)
    os.makedirs(tflite_dir, exist_ok=True)
    logger.info(f"TFLite model will be saved to: {output_tflite_path}")

    # Load Model
    try:
        
        model = keras.models.load_model(keras_model_path)
        logger.info("Keras model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Keras model: {e}")
        raise RuntimeError("Failed to load Keras model.") from e

    # Setup TFLite Converter 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply Quantization 
    quant_type = quant_config.get('type', 'none').lower()
    converter.optimizations = [] # Default: No optimization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS # Always include built-in ops
    ]

    if quant_type == 'default': # TF default optimization (includes some quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Applying default TFLite optimizations (may include quantization).")

    elif quant_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS) # Only if needed
        logger.info("Applying float16 quantization.")

    elif quant_type == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Requires a representative dataset
        rep_dataset_config = quant_config.get('representative_dataset', {})
        use_rep_dataset = rep_dataset_config.get('enabled', False)
        num_samples = rep_dataset_config.get('num_samples', 100)

        if use_rep_dataset:
            try:
                logger.info(f"Loading validation dataset for int8 representative data ({num_samples} samples)...")
                # We only need the validation dataset structure and paths
                # Pass config, but only use the val_dataset part
                _, val_dataset, _ = load_classification_data(config)
                converter.representative_dataset = lambda: representative_dataset_gen(val_dataset, num_samples)
                
                # Specify integer-only quantization
                converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8 # or tf.uint8 depending on model
                converter.inference_output_type = tf.int8 # or tf.uint8
                logger.info("Applying int8 quantization with representative dataset.")
            except Exception as e:
                 logger.error(f"Failed to prepare representative dataset: {e}. Falling back to default optimizations.")
                 converter.optimizations = [tf.lite.Optimize.DEFAULT]
                 quant_type = 'default_fallback' # Indicate fallback
        else:
             logger.warning("int8 quantization selected, but 'representative_dataset.enabled' is false in config.")
             logger.warning("Applying default TFLite optimizations instead.")
             converter.optimizations = [tf.lite.Optimize.DEFAULT]
             quant_type = 'default_fallback' # Indicate fallback
             
    elif quant_type != 'none':
        logger.warning(f"Unsupported quantization type '{quant_type}' in config. Applying no specific quantization.")
        quant_type = 'none'
    else:
         logger.info("No specific quantization requested (type='none').")

    # Convert Model 
    try:
        tflite_model = converter.convert()
        logger.info("Model converted to TFLite format successfully.")
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        # Provide more context if possible
        if 'representative_dataset' in str(e):
             logger.error("Error likely related to the representative dataset generator or int8 settings.")
        raise RuntimeError("TFLite conversion failed.") from e

    # Save TFLite Model
    try:
        with open(output_tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"TFLite model saved to: '{output_tflite_path}' (Quantization: {quant_type})")
    except IOError as e:
        logger.error(f"Failed to write TFLite model file: {e}")
        raise RuntimeError("Failed to save TFLite model.") from e

    # Copy Label Map 
    label_map_filename = paths_config.get('label_map_filename', 'label_map.json')
    source_label_map_path = os.path.join(model_dir, label_map_filename)
    dest_label_map_path = os.path.join(tflite_dir, label_map_filename)

    if os.path.exists(source_label_map_path):
        try:
            shutil.copy2(source_label_map_path, dest_label_map_path)
            logger.info(f"Label map copied to: {dest_label_map_path}")
        except Exception as e:
            logger.warning(f"Failed to copy label map from {source_label_map_path} to {dest_label_map_path}: {e}")
    else:
        logger.warning(f"Label map not found at expected location: {source_label_map_path}. Skipping copy.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export Keras classification model to TensorFlow Lite using a config file.")
    parser.add_argument('--config', type=str, default='models/classification/config.yaml', help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load Configuration  
    config_path = args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        exit(1)

    # Run Export
    try:
        export_model_to_tflite(config)
        logger.info("TFLite export process completed successfully.")
    except Exception as e:
        logger.error(f"Export script failed: {e}")
        exit(1)

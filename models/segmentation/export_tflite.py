import os
import argparse # Keep for __main__ guard
import logging
import yaml
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np

from data import load_segmentation_data, _get_project_root # CORRECTED IMPORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def representative_dataset_gen(dataset: tf.data.Dataset, num_samples: int):
    """Generator for representative dataset for INT8 quantization."""
    # Take a sample from the dataset
    for image, _ in dataset.take(num_samples):
        # Model expects batch dimension
        # Ensure image is float32 as expected by the original model input
        yield [tf.cast(image, tf.float32)] 

def export_model(config_path: str):
    """
    Loads a trained Keras model and exports it to TensorFlow Lite format 
    based on settings in the configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Raises:
        FileNotFoundError: If config or model file is missing.
        ValueError: If configuration settings are invalid.
        RuntimeError: If export fails.
    """
    # Load Config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise ValueError(f"Invalid YAML format in {config_path}")

    # Get Paths
    project_root = _get_project_root()
    paths_config = config.get('paths', {})
    model_save_dir_rel = paths_config.get('model_save_dir', 'trained_models/segmentation/')
    model_save_dir = os.path.join(project_root, model_save_dir_rel)

    export_config = config.get('export', {})
    tflite_filename = export_config.get('tflite_filename', 'segmentation_model.tflite')
    tflite_export_dir_rel = paths_config.get('tflite_export_dir', os.path.join(model_save_dir_rel, 'exported'))
    tflite_export_dir = os.path.join(project_root, tflite_export_dir_rel)
    output_tflite_path = os.path.join(tflite_export_dir, tflite_filename)

    # --- Dynamically find the Keras model file ---
    keras_model_path = None
    specified_model_filename = export_config.get('keras_model_filename')

    if specified_model_filename:
        potential_path = os.path.join(model_save_dir, specified_model_filename)
        if os.path.exists(potential_path) and os.path.isfile(potential_path):
            keras_model_path = potential_path
            logger.info(f"Using Keras model specified in config: {specified_model_filename}")
        else:
            logger.warning(
                f"Keras model specified in config ('{specified_model_filename}') not found at '{potential_path}'. \
                Attempting to find latest model in '{model_save_dir}'."
            )

    if not keras_model_path:  # If not specified or specified file not found
        if not os.path.isdir(model_save_dir):
            logger.error(f"Model save directory not found: {model_save_dir}")
            raise FileNotFoundError(f"Model save directory missing: {model_save_dir}")

        h5_files = [
            f for f in os.listdir(model_save_dir) 
            if f.endswith('.h5') and os.path.isfile(os.path.join(model_save_dir, f))
        ]

        if not h5_files:
            logger.error(f"No .h5 model files found in directory: {model_save_dir}")
            raise FileNotFoundError(f"No .h5 Keras model files found in {model_save_dir}")

        # Prefer files with "final" in their name (case-insensitive)
        final_files = [f for f in h5_files if "final" in f.lower()]
        
        selected_file_list = final_files if final_files else h5_files
        
        # Get the most recently modified file from the selected list
        try:
            latest_model_file = max(
                selected_file_list, 
                key=lambda f: os.path.getmtime(os.path.join(model_save_dir, f))
            )
            keras_model_path = os.path.join(model_save_dir, latest_model_file)
            if final_files and latest_model_file in final_files:
                logger.info(f"Found 'final' model. Using latest: {latest_model_file} from {model_save_dir}")
            else:
                logger.info(f"No 'final' model found or specified one missing. Using latest .h5 model: {latest_model_file} from {model_save_dir}")
        except ValueError: # Should not happen if h5_files is not empty
             logger.error(f"Could not determine latest model file in {model_save_dir} from list: {selected_file_list}")
             raise FileNotFoundError(f"Could not determine latest model file in {model_save_dir}")

    # Load Keras Model
    # The check os.path.exists(keras_model_path) is still good as a final validation
    if not os.path.exists(keras_model_path):
        logger.error(f"Keras model file not found at: {keras_model_path}")
        raise FileNotFoundError(f"Model file missing: {keras_model_path}")
        
    logger.info(f"Loading Keras model from: {keras_model_path}")
    try:
        # Try to load with custom objects first
        try:
            from train import dice_loss, focal_loss, combined_loss, BinaryIoU, DiceCoefficient
            custom_objects = {
                'dice_loss': dice_loss,
                'focal_loss': focal_loss,
                'combined_loss': combined_loss,
                'BinaryIoU': BinaryIoU,
                'DiceCoefficient': DiceCoefficient,
            }
            model = keras.models.load_model(keras_model_path, custom_objects=custom_objects)
            logger.info("Model loaded with custom objects")
        except ValueError as ve:
            if "lambda" in str(ve):
                logger.warning("Model contains lambda functions that can't be loaded. Creating inference model...")
                # Load model architecture and weights separately
                from train import unet_model
                
                # Get model config to rebuild architecture
                model_config = config.get('model', {})
                data_config = config.get('data', {})
                
                # Rebuild the model architecture
                output_channels = data_config.get('num_classes', 1)
                image_size = tuple(data_config.get('image_size', [128, 128])[:2])
                
                model = unet_model(
                    output_channels=output_channels,
                    image_size=image_size,
                    model_config=model_config,
                    data_config=data_config
                )
                
                # Load weights from the saved model
                model.load_weights(keras_model_path)
                logger.info("Model architecture rebuilt and weights loaded successfully")
            else:
                raise ve
        
        logger.info(f"Keras model loaded successfully from: {keras_model_path}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Error loading Keras model: {e}", exc_info=True)
        raise RuntimeError("Failed to load Keras model.")

    # Configure TFLite Converter 
    logger.info("Configuring TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Handle Quantization
    quant_config = export_config.get('quantization', {})
    quant_type = quant_config.get('type', 'none').lower()
    use_repr_dataset = quant_config.get('use_representative_dataset', False)
    num_repr_samples = quant_config.get('num_representative_samples', 100)

    if quant_type == 'float16':
        logger.info("Applying Float16 quantization.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
             tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
             tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops (if needed).
        ]

    elif quant_type == 'int8':
        logger.info("Applying INT8 quantization.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if use_repr_dataset:
            logger.info(f"Using representative dataset with {num_repr_samples} samples.")
            try:
                # Load a dataset for representative data (e.g., using train_ds)
                # Consider using a smaller batch size or unbatching for the generator
                repr_conf = config.copy() 
                repr_conf['data']['batch_size'] = 1 # Use batch size 1 for generator
                repr_conf['data']['shuffle'] = False # No need to shuffle for representative data
                repr_conf['data']['augment'] = False # Don't augment representative data
                
                # Temporarily adjust config to load only what's needed
                if 'validation_split' in repr_conf['data']:
                     del repr_conf['data']['validation_split']
                if 'test_split' in repr_conf['data']:
                     del repr_conf['data']['test_split']
                repr_conf['data']['split_ratios'] = [1.0, 0.0, 0.0] # Load all as 'train'

                # CORRECTED USAGE:
                # train_ds, _, _ = load_segmentation_datasets(repr_conf) 
                # datasets_tuple returns: (train_dataset, val_dataset, test_dataset, steps_per_epoch, validation_steps, test_steps, num_classes)
                datasets_tuple = load_segmentation_data(repr_conf) 
                train_ds = datasets_tuple[0] # train_dataset is the first element

                if train_ds is None:
                    raise ValueError("Failed to load dataset for representative data.")
                
                # Create the generator
                converter.representative_dataset = lambda: representative_dataset_gen(train_ds, num_repr_samples)
                
                # Specify INT8 specifics
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                # Input/output types often remain float32 for INT8 quantization with float fallback
                converter.inference_input_type = tf.float32 # Or tf.int8/uint8 if input layer is quantized
                converter.inference_output_type = tf.float32 # Or tf.int8/uint8 if output layer is quantized
                logger.info("Representative dataset configured for INT8 quantization.")
            except Exception as e:
                logger.error(f"Error preparing representative dataset: {e}", exc_info=True)
                logger.warning("Falling back to INT8 quantization WITHOUT representative dataset.")
                # Fallback: Remove representative dataset if loading failed
                converter.representative_dataset = None 
                # Ensure ops support standard float fallback
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        else:
            logger.warning("Performing INT8 quantization WITHOUT a representative dataset. "
                           "Accuracy may be impacted. Consider enabling 'use_representative_dataset'.")
            # Ensure ops support standard float fallback if no representative data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    elif quant_type == 'none' or quant_type == 'default':
        logger.info("No quantization applied (default conversion).")
        # ensure TF ops are allowed if needed by the model
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    else:
        logger.warning(f"Unsupported quantization type '{quant_type}' specified in config. Performing default conversion.")

    # Convert Model
    logger.info("Converting model to TFLite...")
    try:
        tflite_model = converter.convert()
        logger.info("Model converted successfully.")
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}", exc_info=True)
        raise RuntimeError("TFLite conversion process failed.")

    # Save TFLite Model
    os.makedirs(tflite_export_dir, exist_ok=True)
    try:
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"TFLite model saved to: {output_tflite_path}")
        logger.info(f"Model size: {os.path.getsize(output_tflite_path) / (1024 * 1024):.2f} MB")
    except IOError as e:
        logger.error(f"Failed to write TFLite model file: {e}")
        raise RuntimeError("Failed to save TFLite model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export a trained segmentation model to TFLite using a configuration file.")
    parser.add_argument('--config', type=str, default='models/segmentation/config.yaml',
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        export_model(args.config)
        logger.info("TFLite export process completed successfully.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Export script failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        exit(1)

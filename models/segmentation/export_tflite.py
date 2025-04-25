import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_segmentation_model(model_path: str, output_tflite_path: str, quantization_arg: str = 'default'):
    """
    Export a trained segmentation Keras model to TensorFlow Lite format with optional quantization based on config.

    Args:
        model_path: Path to the saved Keras model file (e.g., .h5).
        output_tflite_path: Path to save the TFLite model.
        quantization_arg: Command-line argument for quantization type (default, int8, float16), overrides config if provided.

    Raises:
        FileNotFoundError: If model file is missing.
        RuntimeError: If export fails.
    """
    try:
        # Load config for quantization settings
        config_path = 'models/segmentation/config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            quant_type = quantization_arg if quantization_arg != 'default' else config.get('quantization', 'default')  # Prioritize argument over config
        except FileNotFoundError:
            logger.warning("Config file not found, defaulting to no quantization.")
            quant_type = 'default'
        
        model = keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quant_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Applying 8-bit quantization.")
        elif quant_type == 'float16':
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.inference_input_type = tf.float16
            converter.inference_output_type = tf.float16
            logger.info("Applying float16 quantization.")
        else:
            logger.info("No quantization applied.")
        
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Segmentation model exported to TFLite at '{output_tflite_path}' with '{quant_type}' quantization")
    except (FileNotFoundError, ValueError, tf.errors.InvalidArgumentError) as e:
        logger.error(f"Export error: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export segmentation model to TensorFlow Lite.")
    parser.add_argument('--model_path', required=True, help="Path to saved Keras model (e.g., trained_models/segmentation/model.h5).")
    parser.add_argument('--output_tflite_path', required=True, help="Path to save TFLite model (e.g., trained_models/segmentation/model.tflite).")
    parser.add_argument('--quantization', type=str, default='default', help='Quantization type (default, int8, float16), overrides config if provided')
    args = parser.parse_args()
    
    try:
        export_segmentation_model(args.model_path, args.output_tflite_path, args.quantization)
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)

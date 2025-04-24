import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_model_to_tflite(model_path:str, output_tflite_path:str,
                           quantization:str = "default") -> None:
    """
    Export a trained Keras model to TensorFlow Lite format with optional quantization.

    Args:
        model_path: Path to the saved Keras model file (e.g., .h5).
        output_tflite_path: Path to save the TFLite model.
        quantization: Type of quantization ('default' for no quantization, 'float16' for FP16, 'int8' for full integer).

    Raises:
        FileNotFoundError: If model file is missing.
        ValueError: If invalid quantization option.
        RuntimeError: If export fails.
    """
    try:
        if quantization not in ["default", "float16", "int8"]:
            raise ValueError("Invalid quantization option. Choose from 'default', 'float16', or 'int8' ")
        model = keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantization == "float16":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                   tf.lite.OpsSet.SELECT_TF_OPS]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == 'int8':
            # For int8 quantization, a representative dataset is required
            def representative_dataset():
                for _ in range(100): # Small subset for calibration
                    yield [np.random.rand(1, model.input_shape[1], model.input_shape[2], 3).astype(np.float32)]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(output_tflite_path), exist_ok = True)
        with open(output_tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"Model exported to TFLITE at '{output_tflite_path}' with '{quantization}' quantization")
    except (FileNotFoundError, ValueError, tf.errors.InvalidArgumentError) as e:
        logger.error(f"Export error: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export classification model to TensorFlow Lite.")
    parser.add_argument('--model_path', required=True, help="Path to saved Keras model (e.g., trained_models/classification/model.h5).")
    parser.add_argument('--output_tflite_path', required=True, help="Path to save TFLite model (e.g., trained_models/exported/model.tflite).")
    parser.add_argument('--quantization', choices=['default', 'float16', 'int8'], default='default', help="Quantization method for optimization.")
    args = parser.parse_args()
    
    try:
        export_model_to_tflite(args.model_path, args.output_tflite_path, args.quantization)
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)

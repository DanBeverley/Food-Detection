# scripts/convert_classification_to_tflite.py
import tensorflow as tf
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_h5_to_tflite(h5_model_path: str, tflite_output_path: str):
    """
    Converts a Keras classification model saved in .h5 format to TensorFlow Lite (.tflite) format.

    Args:
        h5_model_path (str): Path to the input .h5 model file.
        tflite_output_path (str): Path where the output .tflite model file will be saved.
    """
    if not os.path.exists(h5_model_path):
        logging.error(f"Input Keras model file not found: {h5_model_path}")
        return False

    try:
        logging.info(f"Loading Keras model from: {h5_model_path}")
        model = tf.keras.models.load_model(h5_model_path, compile=False) # Don't need compilation state
        logging.info("Keras model loaded successfully.")

        logging.info("Initializing TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optional: Add optimizations (e.g., default optimization)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Optional: Specify supported ops if needed for specific hardware targets
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        logging.info("Converting model to TFLite format...")
        tflite_model = converter.convert()
        logging.info("Model converted successfully.")

        # Ensure output directory exists
        output_dir = os.path.dirname(tflite_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

        logging.info(f"Saving TFLite model to: {tflite_output_path}")
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        logging.info("TFLite model saved successfully.")
        return True

    except Exception as e:
        logging.exception(f"An error occurred during conversion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras .h5 classification model to TFLite format.")
    parser.add_argument(
        "--h5_model",
        type=str,
        required=True,
        help="Path to the input Keras classification model (.h5 file)."
    )
    parser.add_argument(
        "--tflite_output",
        type=str,
        required=True,
        help="Path to save the output TFLite classification model (.tflite file)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if not args.h5_model.lower().endswith(".h5"):
        logging.warning("Input model file does not have a .h5 extension.")
    if not args.tflite_output.lower().endswith(".tflite"):
        logging.warning("Output model file does not have a .tflite extension.")

    success = convert_h5_to_tflite(args.h5_model, args.tflite_output)

    if success:
        logging.info("Conversion process completed successfully.")
    else:
        logging.error("Conversion process failed.")
        exit(1)

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
import yaml
import json

try:
    from .data import load_classification_data, _get_project_root, _get_preprocess_fn
except ImportError:
    from data import load_classification_data, _get_project_root, _get_preprocess_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_tflite_model(tflite_model_path: str, config: dict):
    logger.info(f"Starting TFLite model evaluation for: {tflite_model_path}")

    # 1. Load Validation Data
    # Use the full config to ensure all paths and settings are correct for data loading
    logger.info("Loading validation dataset...")
    try:
        # We need val_dataset for evaluation, and index_to_label for understanding output
        # train_ds is loaded as well but not used here, val_ds is what we need.
        _, val_dataset, index_to_label = load_classification_data(config)
        if val_dataset is None:
            logger.error("Validation dataset could not be loaded. Please check configuration and data.")
            return
    except Exception as e:
        logger.error(f"Error loading validation data: {e}", exc_info=True)
        return
    
    num_classes = len(index_to_label)
    logger.info(f"Validation dataset loaded. Number of classes: {num_classes}")

    # 2. Initialize TFLite Interpreter
    logger.info(f"Initializing TFLite interpreter from: {tflite_model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        logger.error(f"Failed to initialize TFLite interpreter: {e}", exc_info=True)
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    logger.info(f"TFLite model input details: {input_details}")
    logger.info(f"TFLite model output details: {output_details}")

    # 3. Get model-specific preprocessing function (from Keras application)
    # This preprocesses to float32, matching what the original Keras model expected *before* TFLite conversion's int8 input stage.
    try:
        architecture = config.get('model', {}).get('architecture')
        if not architecture:
            logger.error("Model architecture not found in config. Cannot determine preprocessing function.")
            return
        preprocess_fn = _get_preprocess_fn(architecture)
        logger.info(f"Using preprocessing function for architecture: {architecture}")
    except Exception as e:
        logger.error(f"Error getting preprocessing function: {e}", exc_info=True)
        return

    # 4. Evaluation Loop
    correct_predictions = 0
    total_predictions = 0
    
    # Check if the TFLite model expects quantized input/output
    is_quantized_input = input_details['dtype'] == np.int8 or input_details['dtype'] == np.uint8
    is_quantized_output = output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8

    if is_quantized_input:
        input_scale, input_zero_point = input_details['quantization']
        if not isinstance(input_scale, (list, np.ndarray)) or not input_scale: # check if empty or not a list/array
            input_scale = [input_details['quantization_parameters']['scales'][0]] if input_details['quantization_parameters']['scales'].size > 0 else [1.0]
            input_zero_point = [input_details['quantization_parameters']['zero_points'][0]] if input_details['quantization_parameters']['zero_points'].size > 0 else [0]
        logger.info(f"TFLite model expects quantized input. Scale: {input_scale[0]}, Zero Point: {input_zero_point[0]}")
        
    if is_quantized_output:
        output_scale, output_zero_point = output_details['quantization']
        if not isinstance(output_scale, (list, np.ndarray)) or not output_scale:
            output_scale = [output_details['quantization_parameters']['scales'][0]] if output_details['quantization_parameters']['scales'].size > 0 else [1.0]
            output_zero_point = [output_details['quantization_parameters']['zero_points'][0]] if output_details['quantization_parameters']['zero_points'].size > 0 else [0]
        logger.info(f"TFLite model provides quantized output. Scale: {output_scale[0]}, Zero Point: {output_zero_point[0]}")

    logger.info("Starting evaluation loop...")
    for i, (images_batch, labels_batch) in enumerate(val_dataset):
        if i % 10 == 0 and i > 0: # Log more frequently if batch size is large
            logger.info(f"Processing dataset batch {i}...")

        # Convert batch to NumPy once if not already
        images_batch_np = images_batch.numpy() if tf.is_tensor(images_batch) else images_batch
        labels_batch_np = labels_batch.numpy() if tf.is_tensor(labels_batch) else labels_batch

        for j in range(images_batch_np.shape[0]): # Loop through each image in the batch
            single_image_np = images_batch_np[j] # Get one image
            true_label = labels_batch_np[j]

            # Preprocess_fn expects a batch. Create a batch of 1 for the single image.
            # The image at this stage is likely (H, W, C)
            single_image_batch_np = np.expand_dims(single_image_np, axis=0)
            images_float_preprocessed_single = preprocess_fn(single_image_batch_np) # Preprocess the batch of 1

            # If TFLite model expects quantized input, quantize the float32 preprocessed data
            if is_quantized_input:
                # images_float_preprocessed_single is already a NumPy array from preprocess_fn or cast
                images_quantized_single = np.round(images_float_preprocessed_single / input_scale[0] + input_zero_point[0])
                images_prepared_single = images_quantized_single.astype(input_details['dtype'])
            else:
                images_prepared_single = images_float_preprocessed_single # Should be NumPy already
            
            # Ensure images_prepared_single is exactly what the model expects (e.g., batch of 1)
            # This should now be (1, H, W, C) matching the model's expected input shape if batch is 1
            if images_prepared_single.shape[0] != 1 and input_details['shape'][0] == 1:
                # This case should ideally not be hit if preprocess_fn handles batching correctly
                logger.warning(f"Unexpected shape before set_tensor. Expected batch of 1, got {images_prepared_single.shape}. Attempting to reshape.")
                # Potentially reshape or re-batch, but this indicates an issue in pipeline logic above.
                # For now, we assume images_prepared_single is (1, H, W, C)

            interpreter.set_tensor(input_details['index'], images_prepared_single)
            interpreter.invoke()
            predictions_raw = interpreter.get_tensor(output_details['index'])

            if is_quantized_output:
                predictions_float_single = (predictions_raw.astype(np.float32) - output_zero_point[0]) * output_scale[0]
            else:
                predictions_float_single = predictions_raw

            # predictions_float_single is now for a single image, shape (1, num_classes)
            predicted_class_index = np.argmax(predictions_float_single, axis=1)[0] # Get the single prediction
            
            if predicted_class_index == true_label:
                correct_predictions += 1
            total_predictions += 1

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        logger.info(f"Evaluation Complete: Accuracy = {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Save metrics
        eval_metrics = {
            'tflite_model_path': tflite_model_path,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'quantized_input': is_quantized_input,
            'quantized_output': is_quantized_output
        }
        metrics_filename = os.path.splitext(tflite_model_path)[0] + "_evaluation_metrics.json"
        try:
            with open(metrics_filename, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            logger.info(f"TFLite evaluation metrics saved to: {metrics_filename}")
        except Exception as e:
            logger.error(f"Failed to save TFLite evaluation metrics: {e}")

    else:
        logger.warning("No predictions made. Cannot calculate accuracy. Check dataset and loop.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a TFLite classification model.")
    parser.add_argument('--tflite_model_path', type=str, required=True, help="Path to the .tflite model file.")
    parser.add_argument('--config_path', type=str, default='models/classification/config.yaml', help="Path to the main YAML configuration file for data loading specs.")
    args = parser.parse_args()

    if not os.path.exists(args.tflite_model_path):
        logger.error(f"TFLite model file not found: {args.tflite_model_path}")
        exit(1)
    if not os.path.exists(args.config_path):
        logger.error(f"Configuration file not found: {args.config_path}")
        exit(1)

    try:
        with open(args.config_path, 'r') as f:
            main_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML configuration from {args.config_path}: {e}", exc_info=True)
        exit(1)

    evaluate_tflite_model(args.tflite_model_path, main_config)

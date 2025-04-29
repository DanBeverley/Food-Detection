import os
import argparse
import logging
import yaml
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from data import _get_project_root 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """Loads and preprocesses an image for TFLite inference."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0 # Normalize to [0, 1]
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise

def predict(config_path: str, image_path: str):
    """Loads TFLite model and performs classification prediction on an image."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return

    paths_config = config.get('paths', {})
    data_config = config.get('data', {})
    export_config = config.get('export', {})
    project_root = _get_project_root()

    tflite_dir = os.path.join(project_root, paths_config.get('tflite_export_dir', 'trained_models/classification/exported/'))
    tflite_filename = export_config.get('tflite_filename', 'model.tflite')
    tflite_model_path = os.path.join(tflite_dir, tflite_filename)
    
    label_map_filename = paths_config.get('label_map_filename', 'label_map.json')
    label_map_path = os.path.join(tflite_dir, label_map_filename)

    # Load Label Map
    try:
        with open(label_map_path, 'r') as f:
            # Keys might be strings '0', '1', etc. Convert them to integers.
            index_to_label_map_str_keys = json.load(f)
            index_to_label_map = {int(k): v for k, v in index_to_label_map_str_keys.items()}
        logger.info(f"Loaded label map from: {label_map_path}")
        num_classes = len(index_to_label_map)
        logger.info(f"Number of classes found in label map: {num_classes}")
    except FileNotFoundError:
        logger.error(f"Label map not found at: {label_map_path}")
        return
    except (json.JSONDecodeError, ValueError) as e:
         logger.error(f"Error loading or parsing label map: {e}")
         return

    #Load TFLite Model and Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        logger.info(f"TFLite model loaded successfully from: {tflite_model_path}")
    except Exception as e:
        logger.error(f"Failed to load TFLite model or allocate tensors: {e}")
        return

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")

    expected_input_shape = input_details[0]['shape'] # e.g., [1, 224, 224, 3]
    image_size_from_model = tuple(expected_input_shape[1:3]) # (height, width)
    config_image_size = tuple(data_config.get('image_size', [224, 224]))
    if image_size_from_model != config_image_size:
       logger.warning(f"Model input height/width {image_size_from_model} differs from config {config_image_size}")
    
    # Preprocess Image 
    try:
        # Use image size from config for consistency with training
        image_size_hw = tuple(data_config.get('image_size', [224, 224]))
        # PIL uses (width, height)
        image_size_wh = (image_size_hw[1], image_size_hw[0]) 
        input_data = preprocess_image(image_path, image_size_wh)
        
        # Handle different input types (float32, int8, etc.)
        input_type = input_details[0]['dtype']
        if input_type == np.int8 or input_type == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (input_data / input_scale) + input_zero_point
            input_data = tf.cast(input_data, dtype=input_type)
            logger.info("Quantizing input data for INT8 model.")
        else:
             input_data = tf.cast(input_data, dtype=input_type)

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        return

    # Run Inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.info("Inference completed.")
    except Exception as e:
        logger.error(f"Error during TFLite inference: {e}")
        return

    #Postprocess Output 
    # Output is usually shape (1, num_classes)
    predictions = output_data[0]

    # Dequantize if necessary (for INT8 models)
    output_type = output_details[0]['dtype']
    if output_type == np.int8 or output_type == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale
        logger.info("Dequantizing INT8 model output.")

    # Apply softmax if the model output are logits (check final_activation in config or model structure)
    # Assuming output are probabilities if final_activation was softmax
    # Or apply softmax manually if output are logits
    # predictions = tf.nn.softmax(predictions).numpy()

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_label = index_to_label_map.get(predicted_index, "Unknown")

    logger.info("--- Prediction Result ---")
    logger.info(f"Image: {os.path.basename(image_path)}")
    logger.info(f"Predicted Class: {predicted_label}")
    logger.info(f"Confidence: {confidence:.4f}")

    # Optional: Print top N predictions
    top_k = 5
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    logger.info(f"\nTop {top_k} Predictions:")
    for i in top_indices:
        label = index_to_label_map.get(i, "Unknown")
        score = predictions[i]
        logger.info(f"  - {label}: {score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run food classification prediction using a TFLite model.")
    parser.add_argument('--config', type=str, default='models/classification/config.yaml', help="Path to the YAML configuration file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    predict(args.config, args.image)

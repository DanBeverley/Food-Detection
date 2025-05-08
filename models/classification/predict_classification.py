import argparse
import os
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
from pathlib import Path
import json
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utilities
def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

# Core Logic

def load_classification_model(tflite_model_path: str, labels_path: str = None):
    """
    Loads the TFLite classification model interpreter and labels.

    Args:
        tflite_model_path (str): Path to the .tflite model file.
        labels_path (str, optional): Path to the labels file (e.g., labels.txt or labels.json).
                                     If None, attempts to find labels.txt/json near model.

    Returns:
        tuple: (interpreter, input_details, output_details, class_labels list)
               Returns None for class_labels if labels file not found/parsable.
    """
    # Load Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.info(f"TFLite classification model loaded: {tflite_model_path}")
        logging.debug(f"Input details: {input_details}")
        logging.debug(f"Output details: {output_details}")
    except Exception as e:
        logging.error(f"Failed to load TFLite interpreter from {tflite_model_path}: {e}")
        raise

    # Load Labels
    class_labels = None
    if labels_path is None:
        # Auto-detect labels file near model
        base_path = os.path.splitext(tflite_model_path)[0]
        if os.path.exists(base_path + ".txt"):
            labels_path = base_path + ".txt"
        elif os.path.exists(base_path + ".json"):
            labels_path = base_path + ".json"
        else:
            logging.warning(f"Labels file not specified and not found near model ({base_path}.txt/.json).")

    if labels_path and os.path.exists(labels_path):
        try:
            ext = os.path.splitext(labels_path)[1].lower()
            if ext == '.txt':
                with open(labels_path, 'r') as f:
                    class_labels = [line.strip() for line in f.readlines() if line.strip()]
            elif ext == '.json':
                with open(labels_path, 'r') as f:
                    loaded_json = json.load(f)
                    if isinstance(loaded_json, list):
                        class_labels = loaded_json
                    elif isinstance(loaded_json, dict):
                        logging.info(f"Attempting to parse labels from dictionary in {labels_path}")
                        try:
                            # Convert keys to int for sorting, then get values in order
                            # Create a list of the correct size, initially with Nones
                            # Example: if keys are "0", "1", "3", max_key = 3, list size = 4
                            int_keys = sorted([int(k) for k in loaded_json.keys()])
                            if not int_keys or int_keys[0] != 0: # Ensure keys start from 0 if expecting dense list
                                logging.warning(f"Label dictionary keys in {labels_path} do not seem to start from 0 or are non-sequential in a way that might be problematic.")
                                # Decide on behavior: error, or try to build sparse list, or use as is if order doesn't strictly matter for model output mapping
                            
                            # Assuming we want a dense list corresponding to model output indices 0...N-1
                            # If keys are e.g. 0, 1, 3 -> creates list of size 4 with labels[0], labels[1], labels[3] filled.
                            # Model output must align with this interpretation.
                            max_idx = -1
                            if int_keys:
                                max_idx = int_keys[-1]
                            
                            temp_labels = [None] * (max_idx + 1)
                            valid_parse = True
                            for key_str, value_str in loaded_json.items():
                                try:
                                    idx = int(key_str)
                                    if 0 <= idx <= max_idx:
                                        if temp_labels[idx] is not None:
                                            logging.warning(f"Duplicate key {idx} in label map {labels_path}. Overwriting.")
                                        temp_labels[idx] = value_str
                                    else:
                                        logging.warning(f"Key {key_str} is out of expected range [0, {max_idx}] in {labels_path}. Skipping.")
                                        valid_parse = False # Or handle as error
                                except ValueError:
                                    logging.warning(f"Non-integer key '{key_str}' found in label map {labels_path}. Skipping.")
                                    valid_parse = False # Or handle as error
                            
                            # Check if all positions were filled (e.g. if keys were "0", "2", labels[1] would be None)
                            # For a model outputting indices 0 to N-1, we need a dense list.
                            if None in temp_labels and valid_parse: # Only warn if parse itself was valid but resulted in gaps
                                logging.warning(f"Parsed labels from dict in {labels_path}, but some indices are missing (resulting in None). This might be an issue if model outputs these indices.")
                                # Option: filter out Nones if model output range is smaller or sparse
                                # class_labels = [lbl for lbl in temp_labels if lbl is not None]
                                # For now, keep Nones to match potential model output range
                                class_labels = temp_labels
                            elif valid_parse:
                                class_labels = temp_labels
                                logging.info(f"Successfully parsed labels from dictionary in {labels_path}")
                            else:
                                class_labels = None # Parsing had issues with keys/values
                                logging.warning(f"Failed to create a consistent label list from dictionary in {labels_path} due to key issues.")

                        except (ValueError, TypeError) as e_parse:
                            logging.warning(f"Error trying to parse dictionary from {labels_path} into an ordered list: {e_parse}")
                            class_labels = None
                    else:
                        logging.warning(f"Could not find a list or dictionary of labels in JSON file: {labels_path}")

            if class_labels:
                logging.info(f"Loaded {len(class_labels)} class labels from {labels_path}")
            else:
                 logging.warning(f"Failed to parse labels from {labels_path}")

        except Exception as e:
            logging.error(f"Error loading or parsing labels file {labels_path}: {e}")
            class_labels = None # Ensure it's None on error
    else:
        logging.warning(f"Labels file not found: {labels_path}")

    return interpreter, input_details, output_details, class_labels

def preprocess_classification_image(image_path: str, target_size_hw: tuple) -> np.ndarray:
    """Loads and preprocesses an image for classification."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Resize using target_size (PIL uses W, H)
        target_size_wh = (target_size_hw[1], target_size_hw[0])
        img_resized = img.resize(target_size_wh, Image.BILINEAR) # Or other interpolation like ANTIALIAS

        # Ensure image is float32 in [0, 255] range before preprocessing
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Apply EfficientNetV2 preprocessing
        preprocessed_img = preprocess_input(img_array)

        # Add batch dimension
        input_data = np.expand_dims(preprocessed_img, axis=0)
        return input_data

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing classification image {image_path}: {e}")
        raise

def run_classification_inference(
    interpreter: tf.lite.Interpreter,
    input_details: list,
    output_details: list,
    image_path: str,
    model_input_size_hw: tuple,
    class_labels: list = None
) -> tuple[str | int, float] | tuple[None, None]:
    """
    Runs classification inference on a single image.

    Args:
        interpreter: Loaded TFLite interpreter.
        input_details: Interpreter input details.
        output_details: Interpreter output details.
        image_path: Path to the input image.
        model_input_size_hw: Expected input size (H, W) of the TFLite model.
        class_labels (list, optional): List of class labels corresponding to output indices.

    Returns:
        tuple: (predicted_label, confidence_score) or (None, None) on failure.
               Label is the string from class_labels if provided, otherwise the class index.
    """
    try:
        input_data = preprocess_classification_image(image_path, model_input_size_hw)

        # Check input tensor type and scale if necessary (e.g., for UINT8 models)
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.uint8:
             scale, zero_point = input_details[0]['quantization']
             input_data = (input_data / scale + zero_point).astype(input_dtype)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output probabilities/logits
        output_data = interpreter.get_tensor(output_details[0]['index'])[0] # Remove batch dim

        # Handle quantized output if necessary
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            if scale != 0:
                 output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                 logging.warning("Output tensor is quantized (UINT8) but scale is zero. Using raw values.")
                 output_data = output_data.astype(np.float32)

        if not np.isclose(np.sum(output_data), 1.0, atol=0.1):
             logging.debug("Output doesn't sum to 1, applying softmax.")
             output_data = tf.nn.softmax(output_data).numpy()

        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index]) # Convert numpy float

        # Get label string if available
        if class_labels:
            if 0 <= predicted_index < len(class_labels):
                predicted_label = class_labels[predicted_index]
            else:
                logging.warning(f"Predicted index {predicted_index} out of range for labels list (size {len(class_labels)}).")
                predicted_label = predicted_index # Return index as fallback
        else:
            predicted_label = predicted_index # Return index if no labels

        return predicted_label, confidence

    except Exception as e:
        logging.exception(f"Error during classification inference for {image_path}: {e}")
        return None, None

# Standalone Execution

def predict_standalone(config_path: str, image_path: str, top_k: int = 1):
    """Loads TFLite model and performs classification prediction (Standalone mode)."""
     # Load Configuration
    try:
        project_root = _get_project_root()
        if not os.path.isabs(config_path):
             config_path = os.path.join(project_root, config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return
    except Exception as e:
        logging.error(f"Error reading or parsing configuration file {config_path}: {e}")
        return

    # Extract necessary info from config (assuming pipeline config format)
    try:
        tflite_model_rel_path = config['models']['classification_tflite']
        model_input_size_hw = tuple(config['model_params']['classification_input_size'])
        # Define where labels file should be (e.g., relative to model)
        labels_rel_path = config.get('models', {}).get('classification_labels', None) # Or derive from model path

    except KeyError as e:
         logging.error(f"Missing key in configuration file {config_path}: {e}")
         return

    tflite_model_path = os.path.join(project_root, tflite_model_rel_path)
    labels_path = os.path.join(project_root, labels_rel_path) if labels_rel_path else None

    if not os.path.exists(tflite_model_path):
        logging.error(f"TFLite model file not found: {tflite_model_path}")
        return

    # Load Model and Labels
    try:
        interpreter, input_details, output_details, class_labels = load_classification_model(tflite_model_path, labels_path)
        if class_labels is None:
             logging.warning("Proceeding without class labels. Output will be indices.")
    except Exception:
        return

    # Preprocess Image
    try:
        input_data = preprocess_classification_image(image_path, model_input_size_hw)
    except Exception:
        return 

    # Check input tensor type and scale if necessary
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(input_dtype)

    # Run Inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Handle quantized output if necessary
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            if scale != 0:
                 output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                 output_data = output_data.astype(np.float32)

        # Apply softmax if needed (as in run_inference)
        if not np.isclose(np.sum(output_data), 1.0, atol=0.1):
             output_data = tf.nn.softmax(output_data).numpy()

        # Get top K predictions
        # Use partition instead of sort for efficiency if K is small
        # top_k_indices = np.argsort(output_data)[-top_k:][::-1]
        num_classes = len(output_data)
        k = min(top_k, num_classes)
        top_k_indices = np.argpartition(output_data, -k)[-k:]
        # Sort the top-k indices by confidence
        top_k_indices = top_k_indices[np.argsort(output_data[top_k_indices])][::-1]


        print(f"\nTop {k} predictions for: {os.path.basename(image_path)}")
        print("-----------------------------")
        for i in top_k_indices:
            confidence = float(output_data[i])
            if class_labels:
                 if 0 <= i < len(class_labels):
                     label = class_labels[i]
                 else:
                     label = f"Index {i} (out of range)"
            else:
                label = f"Index {i}"
            print(f"{label:<20} : {confidence:.4f}")
        print("-----------------------------")

    except Exception as e:
        logging.exception(f"Error during classification inference: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an image using a TFLite model (Standalone).')
    # Point to the central pipeline config now
    parser.add_argument('--config', type=str, default='config_pipeline.yaml',
                        help="Path to the pipeline YAML configuration file.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to display.')
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    predict_standalone(args.config, args.image, args.top_k)

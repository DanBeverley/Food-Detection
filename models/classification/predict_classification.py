import argparse
import os
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
from pathlib import Path
import json
from tensorflow.keras.applications import mobilenet_v2, efficientnet_v2
try:
    from .data import _get_preprocess_fn
except ImportError:
    from data import _get_preprocess_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utilities
def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def _get_preprocess_fn(architecture: str):
    """
    Returns the correct preprocessing function for a given architecture.
    For EfficientNetV2, this is an identity function as preprocessing is built-in.
    For MobileNetV2, it scales pixels to [-1, 1].
    """
    if architecture.startswith("EfficientNetV2B0"):
        return lambda x: x
    elif architecture == "MobileNetV2":
        return lambda x: (x / 127.5) - 1.0
    else:
        logging.warning(f"Architecture '{architecture}' has no specific preprocess function. Defaulting to scale by /255.")
        return lambda x: x / 255.0


def apply_tta_transforms(image: np.ndarray, depth: np.ndarray = None, transform_idx: int = 0):
    """
    Apply test-time augmentation transform to image and optionally depth.
    
    Transforms:
        0: Original
        1: Horizontal flip
        2: Vertical flip
        3: Horizontal + Vertical flip
    
    Args:
        image: Input image array (H, W, C)
        depth: Optional depth array (H, W, 1)
        transform_idx: Index of transform to apply (0-3)
    
    Returns:
        Tuple of (transformed_image, transformed_depth)
    """
    if transform_idx == 0:
        return image, depth
    elif transform_idx == 1:
        # Horizontal flip
        img_out = np.flip(image, axis=1).copy()
        depth_out = np.flip(depth, axis=1).copy() if depth is not None else None
    elif transform_idx == 2:
        # Vertical flip
        img_out = np.flip(image, axis=0).copy()
        depth_out = np.flip(depth, axis=0).copy() if depth is not None else None
    elif transform_idx == 3:
        # Both flips
        img_out = np.flip(np.flip(image, axis=1), axis=0).copy()
        depth_out = np.flip(np.flip(depth, axis=1), axis=0).copy() if depth is not None else None
    else:
        return image, depth
    
    return img_out, depth_out


def run_inference_with_tta(
    interpreter: tf.lite.Interpreter,
    input_details: list,
    output_details: list,
    model_input_size_hw: tuple,
    architecture: str,
    image_data: np.ndarray,
    depth_data: np.ndarray = None,
    num_tta_transforms: int = 2
) -> np.ndarray:
    """
    Run inference with Test-Time Augmentation.
    
    Runs the model multiple times with different augmentations and averages
    the predictions to improve accuracy.
    
    Args:
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        model_input_size_hw: Expected input size (H, W)
        architecture: Model architecture string
        image_data: Original image array (H, W, C)
        depth_data: Optional depth array
        num_tta_transforms: Number of TTA transforms to use (1-4, default 2 = original + h-flip)
    
    Returns:
        Averaged prediction probabilities
    """
    all_predictions = []
    
    for t_idx in range(min(num_tta_transforms, 4)):
        aug_image, aug_depth = apply_tta_transforms(image_data.copy(), 
                                                     depth_data.copy() if depth_data is not None else None,
                                                     t_idx)
        
        # Preprocess augmented image
        preprocess_fn = _get_preprocess_fn(architecture)
        input_img = Image.fromarray(aug_image.astype(np.uint8) if aug_image.max() > 1 else (aug_image * 255).astype(np.uint8))
        input_img = input_img.resize((model_input_size_hw[1], model_input_size_hw[0]), Image.LANCZOS)
        input_array = np.array(input_img, dtype=np.float32)
        input_array = preprocess_fn(input_array)
        input_array = np.expand_dims(input_array, axis=0)
        
        # Handle single vs dual input models
        if len(input_details) > 1 and aug_depth is not None:
            for inp in input_details:
                if 'depth' in inp['name'].lower():
                    expected_channels = inp['shape'][-1]
                    depth_resized = np.array(Image.fromarray(aug_depth.squeeze()).resize(
                        (model_input_size_hw[1], model_input_size_hw[0]), Image.LANCZOS))
                    depth_norm = depth_resized.astype(np.float32) / 2000.0
                    depth_norm = np.clip(depth_norm, 0.0, 1.0)
                    if expected_channels == 3:
                        depth_norm = np.stack([depth_norm] * 3, axis=-1)
                    else:
                        depth_norm = np.expand_dims(depth_norm, axis=-1)
                    interpreter.set_tensor(inp['index'], np.expand_dims(depth_norm, axis=0))
                elif 'rgb' in inp['name'].lower() or inp['shape'][-1] == 3:
                    interpreter.set_tensor(inp['index'], input_array)
        else:
            interpreter.set_tensor(input_details[0]['index'], input_array)
        
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Apply softmax if needed
        if not np.isclose(np.sum(output), 1.0, atol=0.1):
            output = tf.nn.softmax(output).numpy()
        
        all_predictions.append(output)
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    logging.info(f"TTA: Averaged {len(all_predictions)} predictions")
    
    return avg_predictions


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
    # Load TFLite Interpreter
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

    class_labels = None
    if labels_path is None:
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
                            int_keys = sorted([int(k) for k in loaded_json.keys()])
                            if not int_keys or int_keys[0] != 0: # Ensure keys start from 0 if expecting dense list
                                logging.warning(f"Label dictionary keys in {labels_path} do not seem to start from 0 or are non-sequential in a way that might be problematic.")
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
                            
                            if None in temp_labels and valid_parse: # Only warn if parse itself was valid but resulted in gaps
                                logging.warning(f"Parsed labels from dict in {labels_path}, but some indices are missing (resulting in None). This might be an issue if model outputs these indices.")
                                class_labels = temp_labels
                            elif valid_parse:
                                class_labels = temp_labels
                                logging.info(f"Successfully parsed labels from dictionary in {labels_path}")
                            else:
                                class_labels = None 
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
            class_labels = None 
    else:
        logging.warning(f"Labels file not found: {labels_path}")

    return interpreter, input_details, output_details, class_labels

def preprocess_classification_image(
    architecture: str, 
    image_path: str = None, 
    image_data: np.ndarray = None, 
    target_size_hw: tuple = None
) -> np.ndarray:
    """Loads and preprocesses an image for classification."""
    if image_data is None and image_path is None:
        raise ValueError("Either image_path or image_data must be provided.")
    if target_size_hw is None:
        raise ValueError("target_size_hw must be provided.")
    if not architecture:
        raise ValueError("architecture must be provided for dynamic preprocessing.")

    try:
        if image_data is not None:
            img = Image.fromarray(image_data.astype(np.uint8))
        else:
            img = Image.open(image_path).convert('RGB')
        
        target_size_wh = (target_size_hw[1], target_size_hw[0])
        img_resized = img.resize(target_size_wh, Image.BILINEAR) # Or other interpolation like ANTIALIAS

        img_array = np.array(img_resized, dtype=np.float32)
        preprocess_fn = _get_preprocess_fn(architecture)
        preprocessed_img = preprocess_fn(img_array)

        input_data = np.expand_dims(preprocessed_img, axis=0)
        return input_data

    except FileNotFoundError: 
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing classification image: {e}")
        raise

def preprocess_classification_depth(
    depth_data: np.ndarray = None,
    target_size_hw: tuple = None,
    expected_channels: int = 1
) -> np.ndarray:
    """Preprocesses depth map for classification."""
    if depth_data is None:
        raise ValueError("depth_data must be provided.")
    
    # Resize
    img = Image.fromarray(depth_data)
    # If float, Image.fromarray might fail or need mode 'F'.    
    # Simple resize using cv2 is safer for depth
    import cv2
    target_size_wh = (target_size_hw[1], target_size_hw[0])
    depth_resized = cv2.resize(depth_data, target_size_wh, interpolation=cv2.INTER_NEAREST)  
    depth_norm = depth_resized.astype(np.float32)
    if depth_norm.max() > 1.0:
        depth_norm = depth_norm / 2000.0
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        
    depth_norm = np.expand_dims(depth_norm, axis=-1)
    
    if expected_channels == 3:
        depth_norm = np.concatenate([depth_norm, depth_norm, depth_norm], axis=-1)
        
    input_data = np.expand_dims(depth_norm, axis=0) # (B, H, W, C)
    return input_data

def run_classification_inference(
    interpreter: tf.lite.Interpreter,
    input_details: list,
    output_details: list,
    model_input_size_hw: tuple,
    architecture: str, 
    class_labels: list = None,
    image_path: str = None, 
    image_data: np.ndarray = None,
    depth_data: np.ndarray = None 
) -> tuple[str | int, float] | tuple[None, None]:
    """
    Runs classification inference on a single image.

    Args:
        interpreter: Loaded TFLite interpreter.
        input_details: Interpreter input details.
        output_details: Interpreter output details.
        model_input_size_hw: Expected input size (H, W) of the TFLite model.
        architecture (str): The model architecture string (e.g., 'MobileNetV2', 'EfficientNetV2B0').
        class_labels (list, optional): List of class labels corresponding to output indices.
        image_path (str, optional): Path to the input image.
        image_data (np.ndarray, optional): NumPy array of the image (RGB format).

    Returns:
        tuple: (predicted_label, confidence_score) or (None, None) on failure.
               Label is the string from class_labels if provided, otherwise the class index.
    """
    if image_data is None and image_path is None:
        logging.error("No image data or image path provided for classification inference.")
        return None, None
    if not architecture:
        logging.error("Architecture not provided for preprocessing in classification inference.")
        return None, None

    try:
        input_data = preprocess_classification_image(
            architecture=architecture, 
            image_path=image_path, 
            image_data=image_data, 
            target_size_hw=model_input_size_hw
        )

        # Handle Dual Input (RGB + Depth) or Single Input
        if len(input_details) > 1 and depth_data is not None:
             # Find indices. Usually sorted by name or check shape
             # Simple heuristic: 3 channels = rgb, 1 or 3 channels = depth (check name)
             for inp in input_details:
                 if 'depth' in inp['name'].lower():
                     expected_channels = inp['shape'][-1]
                     depth_input = preprocess_classification_depth(depth_data, model_input_size_hw, expected_channels)
                     interpreter.set_tensor(inp['index'], depth_input)
                 elif 'rgb' in inp['name'].lower() or 'image' in inp['name'].lower() or inp['shape'][-1] == 3:
                     # RGB (preprocessing already done above)
                     # Re-verify quantization for RGB branch specifically if needed?
                     # preprocess_classification_image returns float32 usually.
                     # If quantized model, need to quantize.
                     if inp['dtype'] == np.uint8:
                         scale, zero_point = inp['quantization']
                         rgb_q = (input_data / scale + zero_point).astype(np.uint8)
                         interpreter.set_tensor(inp['index'], rgb_q)
                     else:
                         interpreter.set_tensor(inp['index'], input_data)
        else:
            # Single Input (Legacy or RGB-only)
            input_dtype = input_details[0]['dtype']
            if input_dtype == np.uint8:
                 scale, zero_point = input_details[0]['quantization']
                 input_data = (input_data / scale + zero_point).astype(input_dtype)

            interpreter.set_tensor(input_details[0]['index'], input_data)
        
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        output_dtype = output_details[0]['dtype']
        if output_dtype == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            if scale != 0:
                 output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                 logging.warning("Output tensor is quantized (UINT8) but scale is zero. Using raw values.")
                 output_data = output_data.astype(np.float32)

        logging.info(f"Raw model output stats: min={np.min(output_data):.4f}, max={np.max(output_data):.4f}, sum={np.sum(output_data):.4f}")
        top_5_indices = np.argsort(output_data)[-5:][::-1]
        top_5_scores = output_data[top_5_indices]
        logging.info(f"Top 5 predictions: indices={top_5_indices}, scores={top_5_scores}")

        if not np.isclose(np.sum(output_data), 1.0, atol=0.1):
             logging.debug("Output doesn't sum to 1, applying softmax.")
             output_data = tf.nn.softmax(output_data).numpy()

        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index])

        if class_labels:
            if 0 <= predicted_index < len(class_labels):
                predicted_label = class_labels[predicted_index]
            else:
                logging.warning(f"Predicted index {predicted_index} out of range for labels list (size {len(class_labels)}).")
                predicted_label = predicted_index 
        else:
            predicted_label = predicted_index 

        return predicted_label, confidence

    except Exception as e:
        logging.exception(f"Error during classification inference for {image_path}: {e}")
        return None, None

def predict_standalone(config_path: str, image_path: str, top_k: int = 1):
    """Loads TFLite model and performs classification prediction (Standalone mode)."""
    project_root = _get_project_root()
    if not os.path.isabs(config_path):
        config_path = str(project_root / config_path)

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        return

    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading or parsing YAML configuration from {config_path}: {e}")
        return
    models_cfg = config.get('models', {})
    model_params_cfg = config.get('model_params', {})

    tflite_model_rel_path = models_cfg.get('classification_tflite')
    label_map_rel_path = model_params_cfg.get('classification_label_map')
    classification_config_rel_path = model_params_cfg.get('classification_config_path', 'models/classification/config.yaml') # Path to actual classification config
    classification_config_abs_path = project_root / classification_config_rel_path
    
    clf_architecture = "Unknown"
    if os.path.exists(classification_config_abs_path):
        with open(classification_config_abs_path, 'r') as f_clf_cfg:
            clf_cfg_content = yaml.safe_load(f_clf_cfg)
            clf_architecture = clf_cfg_content.get('model',{}).get('architecture', 'Unknown')
    else:
        logging.warning(f"Classification config file not found at {classification_config_abs_path}, cannot determine architecture for preprocessing.")

    if not tflite_model_rel_path or not label_map_rel_path:
        logging.error("TFLite model path or label map path not found in pipeline config.")
        return

    tflite_model_abs_path = str(project_root / tflite_model_rel_path)
    label_map_abs_path = str(project_root / label_map_rel_path)

    input_size_hw = tuple(model_params_cfg.get('classification_input_size', [224, 224]))

    logging.info(f"Using TFLite model: {tflite_model_abs_path}")
    logging.info(f"Using label map: {label_map_abs_path}")
    logging.info(f"Using input size (H,W): {input_size_hw}")
    logging.info(f"Using architecture for preprocessing: {clf_architecture}")

    try:
        interpreter, input_details, output_details, class_labels = load_classification_model(
            tflite_model_abs_path, label_map_abs_path
        )
        if not interpreter or not class_labels:
            logging.error("Failed to load model or labels. Exiting.")
            return

        predicted_label, confidence = run_classification_inference(
            interpreter, input_details, output_details,
            model_input_size_hw=input_size_hw,
            architecture=clf_architecture, 
            class_labels=class_labels,
            image_path=image_path
        )

        if predicted_label is not None:
            print(f"\nPredicted label: {predicted_label} (confidence: {confidence:.4f})")

    except Exception as e:
        logging.exception(f"Error during classification inference: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an image using a TFLite model (Standalone).')
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

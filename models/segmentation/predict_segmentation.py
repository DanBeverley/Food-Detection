"""Script loads exported TFLite segmentation model, run inference on a given image and save the resulting mask"""
import argparse
import os
import yaml
import numpy as numpy
import tensorflow as tf
from PIL import Image
import logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utilities
def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def load_and_process_image(image_path:str, target_size_hw:tuple) -> tuple[np.ndarray, tuple]:
    """
    Loads and preprocesses an image for TFLite inference.

    Args:
        image_path: Path to the input image.
        target_size_hw: Target size as (height, width).

    Returns:
        A tuple containing:
         - Preprocessed image as a numpy array (batch dim added, normalized).
         - Original image dimensions (height, width).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        original_size_wh = img.size # (width, height)
        original_size_hw = (original_size_wh[1], original_size_wh[0])
        # Resize the model input size (PIL resize uses width, height)
        target_size_wh = (target_size_hw[1], target_size_hw[0])
        img_resized = img.resize(target_size_wh, Image.BILINEAR)
        img_array = np.array(img_resized, dtype = np.float32)
        img_array = img_array / 255.0 # Normalize to [0, 1]
        # Add batch dimension
        input_data = np.expand_dims(img_array, axis=0)
        return input_data, original_size_hw
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading or preprocessing image {image_path}: {e}")
        raise

def postprocess_mask(raw_mask:np.ndarray, original_size_hw:tuple, num_classes:int, threshold:float=0.5) -> np.ndarray:
    """
    Postprocess the raw mask output from TFLite model.

    Args:
        raw_mask: Raw mask output from TFLite model.
        original_size_hw: Original image dimensions (height, width).
        num_classes: Number of classes in the segmentation task.
        threshold: Threshold for binarization.

    Returns:
        Postprocessed mask (resized, threshold/argmaxed) as a numpy array.
    """
    # Remove batch dimension if present
    if raw_mask.ndim > 3:
        raw_mask = np.squeeze(raw_mask, axis = 0) # (H, W, C) or (H, W)
    model_output_size_hw = raw_mask.shape[:2]
    # Process based on number of classes
    if num_classes == 2 or raw_mask.shape[-1] == 1:
       # Binary classification (or single channel output) - Apply threshold
       processed_mask = (raw_mask > threshold).astype(np.uint8)
       if processed_mask.ndim > 2 and processed_mask.shape[-1] == 1:
          processed_mask = np.squeeze(processed_mask, axis=-1) # Shape (H, W)
    elif num_classes > 2:
        # Multi-class classification - Argmax
        processed_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8) # Shape (H, W)
    else: # Should not happen if num_classes is configured
        logger.error(f"Invalid num_classes ({num_classes}) for mask postprocessing")
        # Defauting to thresholding single channel if available
        processed_mask = (raw_mask > threshold).astype(np.uint8)
        if processed_mask.ndim > 2 and processed_mask.shape[-1] == 1:
            processed_mask = np.squeeze(processed_mask, axis =- 1)
    # Resize mask back to original image size using NEAREST interpolation
    # OpenCV resize takes (width, height)
    original_size_wh = (original_size_hw[1], original_size_hw[0])
    final_mask = cv2.resize(processed_mask, original_size_wh, interpolation=cv2.INTER_NEAREST)

    return final_mask # Shape (original_H, original_W)

def overlay_mask_on_image(image_path: str, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays the predicted mask on the original image."""
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error(f"Failed to read original image for overlay: {image_path}")
            return None
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # Ensure RGB

        # Ensure mask is single channel 0 or 1 for simple overlay
        if mask.max() > 1: # If mask has class indices > 1, just use non-zero as mask area
            binary_mask = (mask > 0).astype(np.uint8)
        else:
            binary_mask = mask.astype(np.uint8)

        # Create colored mask (e.g., green)
        colored_mask = np.zeros_like(original_image)
        colored_mask[binary_mask == 1] = [0, 255, 0] # Green color for mask

        # Blend images
        # Ensure sizes match (postprocess_mask should handle this)
        if original_image.shape[:2] != mask.shape[:2]:
             logger.warning(f"Original image size {original_image.shape[:2]} and mask size {mask.shape[:2]} differ. Resizing mask again.")
             mask_resized_for_overlay = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
             colored_mask = np.zeros_like(original_image)
             colored_mask[mask_resized_for_overlay == 1] = [0, 255, 0] # Green

        overlayed_image = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
        return overlayed_image
    except Exception as e:
        logger.error(f"Error overlaying mask: {e}")
        return None


# --- Main Prediction Function ---

def predict(config_path: str, image_path: str, output_path: str = None, show_overlay: bool = False):
    """Loads TFLite model and performs segmentation prediction."""
    # Load Configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except Exception as e:
        logger.error(f"Error reading or parsing configuration file {config_path}: {e}")
        return

    project_root = _get_project_root()
    paths_config = config.get('paths', {})
    export_config = config.get('export', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})

    # Paths
    tflite_dir_rel = paths_config.get('tflite_export_dir', 'trained_models/segmentation/exported/')
    tflite_dir = os.path.join(project_root, tflite_dir_rel)
    tflite_filename = export_config.get('tflite_filename', 'segmentation_model.tflite')
    tflite_model_path = os.path.join(tflite_dir, tflite_filename)

    if not os.path.exists(tflite_model_path):
        logger.error(f"TFLite model file not found: {tflite_model_path}")
        return

    # Model/Data Info
    target_size_hw = tuple(data_config.get('image_size', [256, 256]))
    num_classes = model_config.get('num_classes', 2) # Default to binary if not specified

    # Load TFLite Model and Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info(f"TFLite model loaded: {tflite_model_path}")
        logger.info(f"Input details: {input_details}")
        logger.info(f"Output details: {output_details}")
        # Check model input size matches config if needed
        model_input_shape = input_details[0]['shape'] # e.g., [1, 256, 256, 3]
        model_input_size_hw = tuple(model_input_shape[1:3])
        if model_input_size_hw != target_size_hw:
            logger.warning(f"Model input size {model_input_size_hw} differs from config size {target_size_hw}. Using model size for preprocessing.")
            target_size_hw = model_input_size_hw

    except Exception as e:
        logger.error(f"Failed to load TFLite interpreter: {e}")
        return

    # Preprocess Image
    try:
        input_data, original_size_hw = load_and_preprocess_image(image_path, target_size_hw)
    except Exception as e:
        # Error already logged in helper function
        return

    # Run Inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_mask = interpreter.get_tensor(output_details[0]['index'])
        logger.info("Inference completed.")
    except Exception as e:
        logger.error(f"Error during TFLite inference: {e}")
        return

    # Postprocess Output Mask
    try:
        final_mask = postprocess_mask(raw_mask, original_size_hw, num_classes)
        logger.info(f"Mask postprocessed to size: {final_mask.shape}")
        # Mask values should be class indices (0, 1, ... N-1)
        logger.info(f"Unique values in final mask: {np.unique(final_mask)}")
    except Exception as e:
        logger.error(f"Error during mask postprocessing: {e}")
        return

    # Handle Output
    if output_path:
        try:
            # Save the mask as an image (e.g., grayscale)
            # Scale values if needed for visibility (e.g., if max class index is low)
            save_mask = final_mask.astype(np.uint8)
            if num_classes > 1: # Scale to make classes visible in grayscale
                 save_mask = (save_mask * (255 // (num_classes - 1 if num_classes > 1 else 1))).astype(np.uint8)
            else: # Binary mask 0 or 1 -> 0 or 255
                 save_mask = (save_mask * 255).astype(np.uint8)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(output_path, save_mask)
            logger.info(f"Segmentation mask saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output mask to {output_path}: {e}")

    if show_overlay:
        overlayed_image = overlay_mask_on_image(image_path, final_mask)
        if overlayed_image is not None:
             # Display using matplotlib or OpenCV
             try:
                 import matplotlib.pyplot as plt
                 plt.imshow(overlayed_image)
                 plt.title("Segmentation Overlay")
                 plt.axis('off')
                 plt.show()
             except ImportError:
                 logger.warning("Matplotlib not installed. Cannot show overlay inline. Try saving the output.")
                 # Fallback: Save overlay image if output path was also given
                 if output_path:
                     overlay_output_path = os.path.splitext(output_path)[0] + "_overlay.png"
                     try:
                         # Convert back to BGR for cv2.imwrite
                         cv2.imwrite(overlay_output_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
                         logger.info(f"Overlay image saved to: {overlay_output_path}")
                     except Exception as e:
                         logger.error(f"Failed to save overlay image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run segmentation prediction using a TFLite model.")
    parser.add_argument('--config', type=str, default='models/segmentation/config.yaml',
                        help="Path to the YAML configuration file.")
    parser.add_argument('--image', type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save the output mask image.")
    parser.add_argument('--show', action='store_true',
                        help="Show the original image with the mask overlayed (requires matplotlib).")

    args = parser.parse_args()

    predict(args.config, args.image, args.output, args.show)